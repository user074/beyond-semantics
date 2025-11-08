from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from transformers.models.llama.modeling_llama import LlamaForCausalLM

# Import from absolute path
import sys
sys.path.append("/home/jianingqi/Github/VLM/models")  # Add models directory to path

# Now import your hybrid models
try:
    from llava.twoDmodel.LlamaHybridModel import LlamaHybridModel
    from llava.twoDmodel.LlamaHybridForCausalLM import LlamaHybridForCausalLM
    from llava.twoDmodel.LlamaHybridRotaryEmbedding import LlamaHybridRotaryEmbedding
except ImportError:
    print("Failed to import LlamaHybridModel and LlamaHybridForCausalLM")

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaHybridConfig(LlamaConfig):

    model_type = "llava_llama2d"
    patches_per_side = 24


class LlavaLlamaHybridModel(LlavaMetaModel, LlamaHybridModel):
    config_class = LlavaHybridConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)

        

class LlavaLlamaHybridForCausalLM(LlamaHybridForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaHybridConfig

    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the model using the superclass method
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        
        device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        dtype = kwargs.get('torch_dtype', torch.float32)
        
        # Re-initialize rotary embeddings
        for module in model.modules():
            if isinstance(module, LlamaHybridRotaryEmbedding):
                # print(f"Reinitializing rotary embeddings on device {device} with dtype {dtype}")
                module.initialize_buffers(device=device, dtype=dtype)
        
        return model
    
    def __init__(self, config):
        super().__init__(config)
        self.model = LlavaLlamaHybridModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Tuple[torch.Tensor], List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        batch_img_token_mask: Optional[torch.BoolTensor] = None,  # Add this line
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.training:
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels,
                    batch_img_token_mask,
                ) = self.prepare_inputs_labels_for_multimodal(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    image_sizes,
                )
            else:
                batch_img_token_mask = None
        else:
            if inputs_embeds is None:
                if images is not None:
                    (
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        inputs_embeds,
                        labels,
                        batch_img_token_mask,
                    ) = self.prepare_inputs_labels_for_multimodal(
                        input_ids,
                        position_ids,
                        attention_mask,
                        past_key_values,
                        labels,
                        images,
                        image_sizes,
                    )
                    # print("Images are present")
                    # print("Batch_img_token_mask shape:", batch_img_token_mask.shape)
                    # print("inputs_embeds shape:", inputs_embeds.shape)
                    # print("input_ids shape:", input_ids.shape)
                else:
                    inputs_embeds = self.get_model().embed_tokens(input_ids)
                    batch_img_token_mask = None
                    input_ids = None  # Set input_ids to None when inputs_embeds is set
            
        
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            batch_img_token_mask=batch_img_token_mask,
            return_dict=return_dict,
        )

        # print('outputs logits:', outputs.logits)
        
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                input_ids,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                batch_img_token_mask,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
            )
            # print("Images are present")
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            batch_img_token_mask = None
            # print("No images present")

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            batch_img_token_mask=batch_img_token_mask,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        batch_img_token_mask = kwargs.pop("batch_img_token_mask", None)

        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if batch_img_token_mask is not None:
            inputs["batch_img_token_mask"] = batch_img_token_mask
        return inputs

# Register the configuration and model classes
AutoConfig.register("llava_llama2d", LlavaHybridConfig)
AutoModelForCausalLM.register(LlavaHybridConfig, LlavaLlamaHybridForCausalLM)