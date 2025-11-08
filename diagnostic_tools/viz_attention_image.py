"""
Visualization tool for attention heatmaps over images in LLaVA models.

This module provides functions to extract and visualize attention weights
from vision-language models, specifically showing which image regions the
model attends to when generating specific tokens.

Example usage:

    # Single model visualization with saving
    from viz_attention_image import visualize_single_attention
    result, fig = visualize_single_attention(
        model_path="checkpoints/llava-v1.5-7b",
        image_path="image.jpg",
        prompt_text="Describe the image",
        target_word="bicycle",
        save_path="comparison.png",  # Save comparison figure
        save_overlay_path="overlay.png"  # Save attention overlay
    )

    # Extract attention data only
    from viz_attention_image import extract_attention_for_token, save_attention_overlay
    result = extract_attention_for_token(
        model_path="checkpoints/llava-v1.5-7b",
        image_path="image.jpg",
        prompt_text="Describe the image",
        target_word="bicycle"
    )
    # Access: result.attention_map, result.image_array, result.normalized_entropy
    # Save overlay as PNG
    save_attention_overlay(result, "attention_overlay.png")

    # Compare multiple models and save
    from viz_attention_image import extract_attention_for_token, plot_attention_comparison
    results = []
    for model_path in model_paths:
        result = extract_attention_for_token(model_path, image_path, prompt, target_word="bicycle")
        results.append((model_name, result))
    plot_attention_comparison(results, save_path="comparison.png")
"""

import os
import sys
sys.path.append("../")
sys.path.append("../VLM-Visualizer")

from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch

from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, 
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import (
    tokenizer_image_token, 
    get_model_name_from_path
)

from utils import (
    load_image, 
    aggregate_llm_attention,
    heterogenous_stack,
)

# Local helper for image processing
def _expand2square(pil_img: Image.Image, background_color: Tuple[int, ...]) -> Image.Image:
    """Expand a PIL image to square by padding with background color."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def _process_images_for_viz(
    images: List[Image.Image], 
    image_processor, 
    model_cfg
) -> Tuple[torch.Tensor, List[Image.Image]]:
    """
    Process images for visualization, handling aspect ratio padding.
    
    Args:
        images: List of PIL Images
        image_processor: Model's image processor
        model_cfg: Model configuration
        
    Returns:
        Tuple of (processed_image_tensors, raw_images)
    """
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_raw_images = []
    new_images = []
    
    if image_aspect_ratio == 'pad':
        for image in images:
            image = _expand2square(
                image, 
                tuple(int(x * 255) for x in image_processor.image_mean)
            )
            new_raw_images.append(image)
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        raise NotImplementedError("anyres image aspect ratio not supported yet")
    else:
        raise NotImplementedError(f"Image aspect ratio '{image_aspect_ratio}' not supported")
    
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    
    return new_images, new_raw_images


@dataclass
class AttentionVisualizationResult:
    """Result container for attention visualization."""
    generated_text: str
    attention_map: np.ndarray  # 2D attention heatmap (grid_size x grid_size)
    image_array: np.ndarray  # Original image as numpy array
    normalized_entropy: float  # Normalized entropy of attention distribution
    token_position: int  # Position of the target token in the sequence
    target_word: str  # The word that was visualized


def extract_attention_for_token(
    model_path: str,
    image_path: str,
    prompt_text: str,
    target_word: Optional[str] = None,
    token_position: Optional[int] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 512,
    conv_mode: str = "llava_v1",
    verbose: bool = False,
) -> AttentionVisualizationResult:
    """
    Extract attention weights for a specific token when generating a response.
    
    Args:
        model_path: Path to the model checkpoint
        image_path: Path or URL to the input image
        prompt_text: The prompt text for evaluation
        target_word: Word to find and visualize (searches for first occurrence)
        token_position: Direct token position to visualize (overrides target_word)
        device: Device to use ('cuda' or 'cpu'). Auto-detected if None
        max_new_tokens: Maximum number of tokens to generate
        conv_mode: Conversation template mode (default: 'llava_v1')
        verbose: Whether to print debug information
        
    Returns:
        AttentionVisualizationResult containing attention map, image, and metadata
        
    Raises:
        ValueError: If target_word is not found or token_position is invalid
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    disable_torch_init()
    
    # Load model
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, 
        None,  # model_base
        model_name, 
        load_8bit=False, 
        load_4bit=False, 
        device=device
    )
    
    try:
        # Setup conversation
        conv = conv_templates[conv_mode].copy()
        
        # Load and process image
        image = load_image(image_path)
        image_tensor, images = _process_images_for_viz([image], image_processor, model.config)
        image = images[0]
        image_size = image.size
        
        if isinstance(image_tensor, list):
            image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        # Construct input prompt
        if model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
        else:
            inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Remove system prompt if present
        system_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. "
        prompt = prompt.replace(system_prompt, "")
        
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(model.device)
        
        # Generate with attention
        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_attentions=True,
            )
        
        result_text = tokenizer.decode(outputs["sequences"][0]).strip()
        if verbose:
            print("Generated Text:", result_text)
        
        # Aggregate prompt attentions
        aggregated_prompt_attention = []
        for layer in outputs["attentions"][0]:
            layer_attns = layer.squeeze(0)  # (num_heads, seq_len, seq_len)
            attns_per_head = layer_attns.mean(dim=0)  # (seq_len, seq_len)
            cur = attns_per_head[:-1].cpu().clone()
            cur[1:, 0] = 0.0  # Zero out attention to <bos> for tokens after first
            cur[1:] = cur[1:] / cur[1:].sum(dim=-1, keepdim=True)
            aggregated_prompt_attention.append(cur)
        aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)
        
        # Build complete attention matrix
        llm_attn_matrix = heterogenous_stack(
            [torch.tensor([1])] +
            list(aggregated_prompt_attention) +
            list(map(aggregate_llm_attention, outputs["attentions"]))
        )
        
        # Determine vision token indices
        num_patches = model.get_vision_tower().num_patches
        input_token_len = num_patches + len(input_ids[0]) - 1
        vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors="pt")["input_ids"][0])
        vision_token_end = vision_token_start + num_patches
        
        # Extract attention weights over vision tokens for each generated token
        overall_attn_vectors = []
        for row in llm_attn_matrix[input_token_len:]:
            overall_attn_vectors.append(row[vision_token_start:vision_token_end])
        
        # Find token position
        if token_position is None:
            if target_word is None:
                raise ValueError("Either target_word or token_position must be provided")
            
            # Find first occurrence of target word
            target_word_token = tokenizer.encode(target_word, add_special_tokens=False)
            if not target_word_token:
                raise ValueError(f"Could not encode target_word '{target_word}'")
            
            target_token_id = target_word_token[0]  # Use first token
            if verbose:
                print(f"Target word token: {target_token_id} ({tokenizer.decode([target_token_id])})")
            
            token_position = -1
            for i, token_id in enumerate(outputs["sequences"][0]):
                if token_id == target_token_id:
                    token_position = i
                    break
            
            if token_position < 0:
                raise ValueError(
                    f"Target word '{target_word}' (token {target_token_id}) not found in generated sequence"
                )
        else:
            # Use provided token position, but still try to decode the word
            if target_word is None:
                target_word = tokenizer.decode([outputs["sequences"][0][token_position]])
        
        if verbose:
            print(f"Token position: {token_position}")
            print(f"Token at position: {tokenizer.decode([outputs['sequences'][0][token_position]])}")
            if token_position > 0:
                print(f"Adjacent tokens: {tokenizer.decode(outputs['sequences'][0][token_position-1:token_position+4])}")
        
        # Validate token position
        if token_position < 0 or token_position >= len(overall_attn_vectors):
            raise ValueError(
                f"token_position {token_position} is out of bounds "
                f"(should be between 0 and {len(overall_attn_vectors)-1})"
            )
        
        # Extract and normalize attention weights
        attn_weights = overall_attn_vectors[token_position]
        
        # Normalize to [0, 1]
        min_val = attn_weights.min()
        max_val = attn_weights.max()
        normalized_attn_weights = (attn_weights - min_val) / (max_val - min_val + 1e-10)
        
        # Reshape to square grid
        grid_size = int(np.sqrt(normalized_attn_weights.numel()))
        attn_map = normalized_attn_weights.reshape(grid_size, grid_size).detach().cpu().numpy()
        
        # Calculate normalized entropy
        epsilon = 1e-10
        normalized_weights = attn_weights / attn_weights.sum()
        normalized_weights = normalized_weights + epsilon
        normalized_weights = normalized_weights / normalized_weights.sum()
        entropy = -torch.sum(normalized_weights * torch.log2(normalized_weights))
        max_entropy = torch.log2(torch.tensor(float(num_patches)))
        normalized_entropy = (entropy / max_entropy).item()
        
        image_np = np.array(image)
        
        return AttentionVisualizationResult(
            generated_text=result_text,
            attention_map=attn_map,
            image_array=image_np,
            normalized_entropy=normalized_entropy,
            token_position=token_position,
            target_word=target_word or tokenizer.decode([outputs["sequences"][0][token_position]])
        )
    
    finally:
        # Clean up
        del model
        del image_processor
        del tokenizer
        torch.cuda.empty_cache()


def visualize_attention_overlay(
    result: AttentionVisualizationResult,
    alpha: float = 0.7,
    colormap: int = cv2.COLORMAP_RAINBOW,
    image_alpha: float = 0.4,
) -> np.ndarray:
    """
    Create an overlay visualization of attention on the image.
    
    Args:
        result: AttentionVisualizationResult from extract_attention_for_token
        alpha: Opacity of attention overlay (0-1)
        colormap: OpenCV colormap constant (default: RAINBOW)
        image_alpha: Opacity of base image (0-1)
        
    Returns:
        RGB image array with attention overlay
    """
    # Resize attention map to image dimensions
    h, w = result.image_array.shape[:2]
    attention_resized = cv2.resize(result.attention_map, (w, h))
    
    # Scale to 0-255 and apply colormap
    attention_norm = (attention_resized * 255).astype(np.uint8)
    attention_colormap = cv2.applyColorMap(attention_norm, colormap)
    # attention_colormap = cv2.cvtColor(attention_colormap, cv2.COLOR_BGR2RGB)
    
    # Ensure image is RGB
    if len(result.image_array.shape) == 2:
        image_rgb = cv2.cvtColor(result.image_array, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = cv2.cvtColor(result.image_array, cv2.COLOR_BGR2RGB)
    
    # Blend image and attention
    overlay = cv2.addWeighted(image_rgb, image_alpha, attention_colormap, alpha, 0)
    
    return overlay


def save_attention_overlay(
    result: AttentionVisualizationResult,
    save_path: str,
    alpha: float = 0.7,
    colormap: int = cv2.COLORMAP_RAINBOW,
    image_alpha: float = 0.4,
) -> None:
    """
    Save an attention overlay visualization as a PNG image.
    
    Args:
        result: AttentionVisualizationResult from extract_attention_for_token
        save_path: Path to save the PNG file (will add .png if not present)
        alpha: Opacity of attention overlay (0-1)
        colormap: OpenCV colormap constant (default: RAINBOW)
        image_alpha: Opacity of base image (0-1)
    """
    overlay = visualize_attention_overlay(result, alpha, colormap, image_alpha)
    
    # Ensure save_path ends with .png
    if not save_path.lower().endswith('.png'):
        save_path = save_path + '.png'
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Save as PNG
    Image.fromarray(overlay.astype(np.uint8)).save(save_path)


def plot_attention_comparison(
    results: List[Tuple[str, AttentionVisualizationResult]],
    target_word: Optional[str] = None,
    figsize: Tuple[int, int] = (24, 10),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot a comparison of attention visualizations for multiple models.
    
    Args:
        results: List of (model_name, AttentionVisualizationResult) tuples
        target_word: Word being visualized (for title)
        figsize: Figure size (width, height)
        save_path: Optional path to save the figure
        show: Whether to display the figure
        
    Returns:
        matplotlib Figure object
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models + 1, figsize=figsize, constrained_layout=True)
    
    # Show original image
    first_result = results[0][1]
    axes[0].imshow(first_result.image_array)
    target_display = target_word or first_result.target_word
    axes[0].set_title(f"Original Image\nConditioned Word: {target_display}", fontsize=14)
    axes[0].axis("off")
    
    # Show each model's attention
    for idx, (model_name, result) in enumerate(results, start=1):
        overlay = visualize_attention_overlay(result)
        axes[idx].imshow(overlay)
        axes[idx].set_title(
            f"{model_name}\nEntropy: {result.normalized_entropy:.2f}",
            fontsize=14
        )
        axes[idx].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        # Ensure save_path ends with .png
        if not save_path.lower().endswith('.png'):
            save_path = save_path + '.png'
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', format='png')
        print(f"Saved comparison plot to: {save_path}")
    
    if show:
        plt.show()
    
    return fig


# Convenience function for single model visualization
def visualize_single_attention(
    model_path: str,
    image_path: str,
    prompt_text: str,
    target_word: Optional[str] = None,
    token_position: Optional[int] = None,
    model_name: Optional[str] = None,
    save_path: Optional[str] = None,
    save_overlay_path: Optional[str] = None,
    show: bool = True,
    **kwargs
) -> Tuple[AttentionVisualizationResult, plt.Figure]:
    """
    Convenience function to extract and visualize attention for a single model.
    
    Args:
        model_path: Path to model checkpoint
        image_path: Path to input image
        prompt_text: Prompt for the model
        target_word: Word to visualize (or use token_position)
        token_position: Token position to visualize (or use target_word)
        model_name: Display name for the model (defaults to basename of model_path)
        save_path: Optional path to save comparison figure (PNG)
        save_overlay_path: Optional path to save attention overlay image (PNG)
        show: Whether to display the figure
        **kwargs: Additional arguments passed to extract_attention_for_token
        
    Returns:
        Tuple of (AttentionVisualizationResult, matplotlib Figure)
    """
    result = extract_attention_for_token(
        model_path, image_path, prompt_text, 
        target_word=target_word, 
        token_position=token_position,
        **kwargs
    )
    
    if model_name is None:
        model_name = os.path.basename(model_path.rstrip('/'))
    
    # Save overlay if requested
    if save_overlay_path:
        save_attention_overlay(result, save_overlay_path)
    
    fig = plot_attention_comparison(
        [(model_name, result)],
        target_word=result.target_word,
        save_path=save_path,
        show=show,
    )
    
    return result, fig


if __name__ == "__main__":
    # Example usage
    model_paths = [
        "../checkpoints/llava-v1.5-7b",
        "../checkpoints/llava-v1.5-7b-normWmean",
        "../checkpoints/llava-v1.5-7b-multilayerNorm",
    ]
    
    image_path = "../Data/COCO/val2017/000000445658.jpg"
    question = "Where is the teddy bear"
    target_word = "teddy"
    
    # Extract attention for each model
    results = []
    for model_path in model_paths:
        model_name = os.path.basename(model_path.rstrip('/'))
        result = extract_attention_for_token(
            model_path, 
            image_path, 
            question, 
            target_word=target_word,
            verbose=True
        )
        results.append((model_name, result))
    
    # Plot comparison and save
    plot_attention_comparison(
        results, 
        target_word=target_word,
        save_path="attention_comparison.png",
        show=True
    )
    
    # Optionally save individual overlays
    for model_name, result in results:
        overlay_path = f"attention_overlay_{model_name.replace('/', '_')}.png"
        save_attention_overlay(result, overlay_path)
        print(f"Saved overlay to: {overlay_path}")
