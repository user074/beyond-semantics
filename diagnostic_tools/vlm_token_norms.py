"""
vlm_token_norms.py

Compute and visualize L2-norm distributions for:
  - Vision tokens (CLIP -> projector -> LLM-input features)
  - Text tokens (LLM input embeddings)

Tested with LLaVA 1.5 7B-style layouts.

Example:
python vlm_token_norms.py \
    --model_path ../checkpoints/llava-v1.5-7b \
    --vision_tower_path ../checkpoints/clip-vit-large-patch14-336 \
    --coco_root ../Data/COCO \
    --device cuda \
    --weights_type bin \
    --max_images 20 \
    --save_vectors \
    --out_dir outputs/token_norms
"""


import os
import sys
sys.path.append("../")
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Literal
from io import BytesIO

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import requests
from pycocotools.coco import COCO
from safetensors.torch import load_file
from transformers import LlamaForCausalLM, LlamaTokenizer

from LLaVA.llava.model.multimodal_projector.builder import build_vision_projector
from LLaVA.llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from LLaVA.llava.mm_utils import process_images


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration class for model paths and settings"""
    model_path: str
    vision_tower_path: str
    device: str = 'cuda'
    dtype: torch.dtype = torch.float16
    weights_type: Literal['bin', 'safetensors'] = 'bin'
    safetensors_index: Optional[str] = None


class ConfigWrapper:
    """Wrapper for JSON config dictionary"""
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


# ============================================================================
# Model Loading and Processing
# ============================================================================

def load_config(model_path: str) -> ConfigWrapper:
    """Load and parse model configuration"""
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return ConfigWrapper(config_dict)


class LLaVAImageProcessor:
    """Class to handle image processing with LLaVA models"""
    
    def __init__(self, model_config: ModelConfig):
        """Initialize the image processor with models"""
        self.model_config = model_config
        self.config = load_config(model_config.model_path)
        self.mm_projector = self._init_mm_projector()
        self.vision_tower = self._init_vision_tower()
    
    def _init_mm_projector(self) -> torch.nn.Module:
        """Initialize and load the multimodal projector"""
        mm_projector = build_vision_projector(self.config)
        
        if self.model_config.weights_type == 'bin':
            weights_path = os.path.join(
                self.model_config.model_path, 
                'pytorch_model-00002-of-00002.bin'
            )
            mm_projector_weights = torch.load(weights_path, map_location='cpu')
            mm_projector_weights = {
                k: v.to(self.model_config.dtype) 
                for k, v in mm_projector_weights.items()
            }
            mm_projector_weights = {
                k.split('mm_projector.')[1]: v 
                for k, v in mm_projector_weights.items() 
                if 'mm_projector' in k
            }
        elif self.model_config.weights_type == 'safetensors':
            if not self.model_config.safetensors_index:
                raise ValueError("safetensors_index required for safetensors weights")
            
            safetensor_file = os.path.join(
                self.model_config.model_path,
                f'model-{self.model_config.safetensors_index}.safetensors'
            )
            state_dict = load_file(safetensor_file, device='cpu')
            mm_projector_weights = {
                k.split("mm_projector.")[1]: v 
                for k, v in state_dict.items() 
                if "mm_projector" in k
            }
        else:
            raise ValueError(f"Unsupported weights type: {self.model_config.weights_type}")
        
        mm_projector = mm_projector.to(self.model_config.dtype)
        mm_projector.load_state_dict(mm_projector_weights)
        return mm_projector.to(self.model_config.device)
    
    def _init_vision_tower(self) -> CLIPVisionTower:
        """Initialize and load the CLIP vision tower"""
        vision_tower = CLIPVisionTower(
            self.model_config.vision_tower_path,
            args=self.config
        )
        return vision_tower.to(self.model_config.device)
    
    @staticmethod
    def load_image(image_path: str) -> Image.Image:
        """Load an image from a file path or URL"""
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
        return image
    
    @staticmethod
    def resize_image(image: Image.Image, scale: float) -> Image.Image:
        """Resize image by a scale factor"""
        width, height = image.size
        new_width = int(width * scale)
        new_height = int(height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def process_image(self, image: Image.Image) -> torch.Tensor:
        """Process a single image through vision tower"""
        image_tensor = process_images(
            [image], 
            self.vision_tower.image_processor, 
            self.config
        )
        
        if isinstance(image_tensor, list):
            image_tensor = [
                img.to(self.model_config.device, dtype=self.model_config.dtype) 
                for img in image_tensor
            ]
        else:
            image_tensor = image_tensor.to(
                self.model_config.device, 
                dtype=self.model_config.dtype
            )
        
        with torch.no_grad():
            image_features = self.vision_tower(image_tensor)
        return image_features
    
    def get_llm_input(self, image_path: str, scale: float = 3) -> torch.Tensor:
        """Process an image and get LLM input tensor"""
        image = self.load_image(image_path)
        image = self.resize_image(image, scale)
        image_features = self.process_image(image)
        
        with torch.no_grad():
            llm_input = self.mm_projector(
                image_features.to(self.model_config.device)
            )
        return llm_input


# ============================================================================
# Token Norm Extraction
# ============================================================================

def extract_vision_token_norms(
    processor: LLaVAImageProcessor,
    coco_caps: COCO,
    image_dir: str,
    max_images: int = 0,
    scale: float = 3.0
) -> list:
    """Extract L2 norms from vision tokens"""
    all_token_norms = []
    image_ids = coco_caps.getImgIds()
    
    if max_images == 0:
        max_images = len(image_ids)
    for i, img_id in enumerate(image_ids[:max_images]):
        img_info = coco_caps.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])
        
        llm_input = processor.get_llm_input(img_path, scale=scale)
        
        if llm_input.dim() == 3:
            token_norms = llm_input.norm(p=2, dim=2).flatten().cpu().tolist()
        else:
            token_norms = llm_input.norm(p=2, dim=1).cpu().tolist()
        
        all_token_norms.extend(token_norms)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1} vision images...")
    
    return all_token_norms


def extract_text_token_norms(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizer,
    coco_caps: COCO,
    max_images: int = 0
) -> list:
    """Extract L2 norms from text token embeddings"""
    all_token_norms = []
    image_ids = coco_caps.getImgIds()
    
    if max_images == 0:
        max_images = len(image_ids)
        
    for i, img_id in enumerate(image_ids[:max_images]):
        ann_ids = coco_caps.getAnnIds(imgIds=img_id)
        captions = coco_caps.loadAnns(ann_ids)
        
        for ann in captions:
            caption_text = ann['caption']
            inputs = tokenizer(caption_text, return_tensors='pt')
            input_ids = inputs['input_ids'].to(model.device)
            
            with torch.no_grad():
                raw_embeddings = model.model.embed_tokens(input_ids)
                token_norms = raw_embeddings.norm(p=2, dim=2).flatten().cpu().tolist()
            
            all_token_norms.extend(token_norms)
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1} text images...")
    
    return all_token_norms


# ============================================================================
# Visualization
# ============================================================================

def plot_token_norms_comparison(
    vision_norms: list,
    text_norms: list,
    output_prefix: str = 'token_norms_comparison'
):
    """Create overlayed violin plot of token norm distributions"""
    sns.set(style="whitegrid", font_scale=0.9)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    
    data = pd.DataFrame({
        'Norm': list(vision_norms) + list(text_norms),
        'Token Type': (
            ['Vision Token Norms'] * len(vision_norms) + 
            ['Text Token Norms'] * len(text_norms)
        ),
        'All': 'All'
    })
    
    palette = sns.color_palette("Set2")
    token_colors = {
        'Vision Token Norms': palette[1],
        'Text Token Norms': palette[0]
    }
    
    plt.figure(figsize=(5.5, 3))
    
    ax = sns.violinplot(
        data=data,
        x='Norm',
        y='All',
        hue='Token Type',
        split=False,
        inner='quartile',
        linewidth=1,
        dodge=False,
        palette=token_colors,
        alpha=0.7
    )
    
    ax.set_xscale('log')
    ax.set_ylabel('')
    ax.set_yticks([])
    ax.set_xlabel('L2 Norm (Log Scale)', fontsize=9, labelpad=10)
    
    plt.title(
        'Overlayed Violin Plot of Token Norm Distributions',
        size=10.5,
        y=1.02,
        fontweight='bold'
    )
    
    legend = plt.legend(
        title=None,
        loc='upper right',
        frameon=True,
        framealpha=0.9,
        edgecolor='lightgray',
        fontsize=8.5
    )
    legend.get_frame().set_linewidth(0.8)
    
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.tight_layout(pad=1.2)
    
    plt.savefig(f'{output_prefix}.pdf', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_prefix}.png', dpi=600, bbox_inches='tight')
    print(f"Plots saved as {output_prefix}.pdf and {output_prefix}.png")
    plt.show()


def save_vectors(txt_vector: list, img_vector: list, filename: str = 'vectors.npz'):
    """Save token norm vectors to a file"""
    np.savez(filename, txt=txt_vector, img=img_vector)
    print(f"Vectors saved to {filename}")


def load_vectors(filename: str = 'vectors.npz') -> tuple:
    """Load token norm vectors from a file"""
    data = np.load(filename)
    return data['txt'], data['img']


# ============================================================================
# Argument Parsing
# ============================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser"""
    parser = argparse.ArgumentParser(
        description="Compute and visualize L2-norm distributions for vision and text tokens in LLaVA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with bin weights:
  python vlm_token_norms.py \\
    --model_path /path/to/llava-v1.5-7b \\
    --vision_tower_path /path/to/clip-vit-large-patch14-336 \\
    --coco_root /path/to/COCO \\
    --out_dir outputs/token_norms

  # With safetensors weights:
  python vlm_token_norms.py \\
    --model_path /path/to/llava-v1.5-7b \\
    --vision_tower_path /path/to/clip-vit-large-patch14-336 \\
    --coco_root /path/to/COCO \\
    --weights_type safetensors \\
    --safetensors_index 00001 \\
    --out_dir outputs/token_norms
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the LLaVA model directory'
    )
    parser.add_argument(
        '--vision_tower_path',
        type=str,
        required=True,
        help='Path to the CLIP vision tower directory'
    )
    parser.add_argument(
        '--coco_root',
        type=str,
        required=True,
        help='Root directory of COCO dataset'
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--out_dir',
        type=str,
        default='outputs/token_norms',
        help='Output directory for results (default: outputs/token_norms)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for computation (default: cuda)'
    )
    parser.add_argument(
        '--max_images',
        type=int,
        default=0,
        help='Maximum number of images to process (default: 20)'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=3.0,
        help='Image scaling factor (default: 3.0)'
    )
    parser.add_argument(
        '--weights_type',
        type=str,
        default='bin',
        choices=['bin', 'safetensors'],
        help='Type of model weights (default: bin)'
    )
    parser.add_argument(
        '--safetensors_index',
        type=str,
        default=None,
        help='Index for safetensors file (e.g., "00001" for model-00001.safetensors)'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float16', 'float32', 'bfloat16'],
        help='Data type for model computation (default: float16)'
    )
    parser.add_argument(
        '--caption_file',
        type=str,
        default=None,
        help='Path to COCO captions file (default: {coco_root}/annotations/captions_val2017.json)'
    )
    parser.add_argument(
        '--image_dir',
        type=str,
        default=None,
        help='Path to COCO images directory (default: {coco_root}/val2017)'
    )
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='token_norms_comparison',
        help='Prefix for output files (default: token_norms_comparison)'
    )
    parser.add_argument(
        '--save_vectors',
        action='store_true',
        help='Save token norm vectors to .npz file'
    )
    parser.add_argument(
        '--vectors_file',
        type=str,
        default='vectors.npz',
        help='Filename for saved vectors (default: vectors.npz)'
    )
    
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command-line arguments and check file existence"""
    # Check model path
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    
    # Check vision tower path
    if not os.path.exists(args.vision_tower_path):
        raise FileNotFoundError(f"Vision tower path does not exist: {args.vision_tower_path}")
    
    # Check COCO root
    if not os.path.exists(args.coco_root):
        raise FileNotFoundError(f"COCO root directory does not exist: {args.coco_root}")
    
    # Set default paths if not provided
    if args.caption_file is None:
        args.caption_file = os.path.join(args.coco_root, 'annotations', 'captions_val2017.json')
    if args.image_dir is None:
        args.image_dir = os.path.join(args.coco_root, 'val2017')
    
    # Check caption file
    if not os.path.exists(args.caption_file):
        raise FileNotFoundError(f"Caption file does not exist: {args.caption_file}")
    
    # Check image directory
    if not os.path.exists(args.image_dir):
        raise FileNotFoundError(f"Image directory does not exist: {args.image_dir}")
    
    # Validate safetensors arguments
    if args.weights_type == 'safetensors' and args.safetensors_index is None:
        raise ValueError("safetensors_index is required when weights_type is 'safetensors'")
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution function"""
    # Parse command-line arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    try:
        validate_arguments(args)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    
    # Convert dtype string to torch dtype
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16
    }
    torch_dtype = dtype_map[args.dtype]
    
    print("Initializing models...")
    print(f"Model path: {args.model_path}")
    print(f"Vision tower path: {args.vision_tower_path}")
    print(f"Device: {args.device}")
    print(f"Max images: {args.max_images}")
    print(f"Scale: {args.scale}")
    print(f"Weights type: {args.weights_type}")
    
    # Initialize vision processor
    model_config = ModelConfig(
        model_path=args.model_path,
        vision_tower_path=args.vision_tower_path,
        device=args.device,
        dtype=torch_dtype,
        weights_type=args.weights_type,
        safetensors_index=args.safetensors_index
    )
    vision_processor = LLaVAImageProcessor(model_config)
    
    # Initialize text model
    print("Loading text model...")
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path, use_fast=False)
    text_model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        ignore_mismatched_sizes=True
    )
    text_model.eval()
    
    # Load COCO dataset
    print(f"Loading COCO dataset from {args.caption_file}...")
    coco_caps = COCO(args.caption_file)
    
    # Extract vision token norms
    print(f"\nExtracting vision token norms from {args.image_dir}...")
    vision_norms = extract_vision_token_norms(
        vision_processor, coco_caps, args.image_dir, args.max_images, args.scale
    )
    print(f"Total vision tokens: {len(vision_norms)}")
    
    # Extract text token norms
    print("\nExtracting text token norms...")
    text_norms = extract_text_token_norms(
        text_model, tokenizer, coco_caps, args.max_images
    )
    print(f"Total text tokens: {len(text_norms)}")
    
    # Save vectors if requested
    if args.save_vectors:
        vectors_path = os.path.join(args.out_dir, args.vectors_file)
        print(f"\nSaving vectors to {vectors_path}...")
        save_vectors(text_norms, vision_norms, vectors_path)
    
    # Create visualization
    print(f"\nCreating visualization in {args.out_dir}...")
    output_prefix = os.path.join(args.out_dir, args.output_prefix)
    plot_token_norms_comparison(vision_norms, text_norms, output_prefix)
    
    print("\nAnalysis complete!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)