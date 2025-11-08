#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)
            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type
        self.config.add_spatial_coordinates = getattr(model_args, 'add_spatial_coordinates', False)
        self.config.compression_after_projection = getattr(model_args, 'compression_after_projection', False)
        self.config.final_spatial_size = getattr(model_args, 'final_spatial_size', 16)
        self.config.add_registers = getattr(model_args, 'add_registers', False)
        self.config.num_registers = getattr(model_args, 'num_registers', 4)
        self.config.normalize_projector = getattr(model_args, 'normalize_projector', False)
        self.config.use_multilayer_features = getattr(model_args, 'use_multilayer_features', False)
        self.config.num_vision_tokens = getattr(model_args, 'num_vision_tokens', 576)

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]
    return unpadded_tensor

class SpatialTokenInterleaver(nn.Module):
    def __init__(self, tokenizer, embedding_layer, grid_size=24, sparsity=4):
        """
        Initializes the SpatialTokenInterleaver.

        Args:
            tokenizer: The tokenizer to convert text to tokens.
            embedding_layer: The embedding layer of the language model.
            grid_size (int): The size of the image grid (e.g., 24 for a 24x24 grid).
            sparsity (int): The interval at which to interleave spatial tokens.
                            For example, sparsity=4 means after every 4 image tokens.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.grid_size = grid_size
        self.sparsity = sparsity  # New parameter for sparsity

    def get_position_description(self, idx):
        row = idx // self.grid_size
        col = idx % self.grid_size
        description = f"{row},{col}"  # Recommended concise format
        return description

    def get_spatial_embeddings(self, device, num_patches):
        spatial_embeddings = []
        for idx in range(num_patches):
            description = self.get_position_description(idx)
            desc_tokens = self.tokenizer(
                description,
                add_special_tokens=False,
                return_tensors="pt"
            )["input_ids"].squeeze(0).to(device)
            desc_embeddings = self.embedding_layer(desc_tokens)
            spatial_embeddings.append(desc_embeddings)
        return spatial_embeddings

    def forward(self, hidden_states, img_token_mask):
        """
        Interleaves spatial embeddings into the hidden states based on sparsity.

        Args:
            hidden_states (torch.Tensor): Shape (batch_size, seq_length, hidden_dim)
            img_token_mask (torch.Tensor): Shape (batch_size, seq_length) indicating image token positions.

        Returns:
            new_hidden_states (torch.Tensor): Interleaved hidden states.
            new_mask (torch.Tensor): Updated mask indicating image tokens.
        """
        batch_size, seq_length, hidden_dim = hidden_states.shape
        new_hidden_states = []
        new_masks = []

        for b in range(batch_size):
            cur_hidden = []
            cur_mask = []
            img_positions = torch.where(img_token_mask[b])[0]
            spatial_embeddings = self.get_spatial_embeddings(hidden_states.device, len(img_positions))
            last_pos = 0

            for i, img_pos in enumerate(img_positions):
                # Append tokens before the image token
                if img_pos > last_pos:
                    cur_hidden.append(hidden_states[b, last_pos:img_pos])
                    cur_mask.extend([False] * (img_pos - last_pos))

                # Append image token
                cur_hidden.append(hidden_states[b, img_pos:img_pos+1])
                cur_mask.append(True)

                # Interleave spatial tokens based on sparsity
                if (i + 1) % self.sparsity == 0:
                    # Interleave spatial tokens
                    spatial_emb = spatial_embeddings[i]
                    cur_hidden.append(spatial_emb)
                    cur_mask.extend([False] * spatial_emb.size(0))

                last_pos = img_pos + 1

            # Append remaining tokens after the last image token
            if last_pos < seq_length:
                cur_hidden.append(hidden_states[b, last_pos:])
                cur_mask.extend([False] * (seq_length - last_pos))

            # Concatenate for the current batch
            cur_hidden = torch.cat(cur_hidden, dim=0)
            cur_mask = torch.tensor(cur_mask, device=hidden_states.device)

            new_hidden_states.append(cur_hidden)
            new_masks.append(cur_mask)

        # Pad sequences to the same length
        max_len = max(h.size(0) for h in new_hidden_states)
        padded_states = torch.zeros((batch_size, max_len, hidden_dim), device=hidden_states.device)
        padded_masks = torch.zeros((batch_size, max_len), dtype=torch.bool, device=hidden_states.device)

        for i in range(batch_size):
            seq_len = new_hidden_states[i].size(0)
            padded_states[i, :seq_len] = new_hidden_states[i]
            padded_masks[i, :seq_len] = new_masks[i]

        return padded_states, padded_masks
    
class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def rearrange_image_features(self, image_features, patch_size=3):
        # image_features shape: (batch_size, num_patches, hidden_dim)
        batch_size, num_patches, hidden_dim = image_features.shape
        grid_size = int(num_patches ** 0.5)  # Should be 24 for a 24x24 grid

        # Reshape to (batch_size, grid_size, grid_size, hidden_dim)
        image_features = image_features.view(batch_size, grid_size, grid_size, hidden_dim)

        # Initialize a list to collect rearranged patches
        rearranged_patches = []

        # Loop over patches
        for p_row in range(0, grid_size, patch_size):
            for p_col in range(0, grid_size, patch_size):
                # Extract the patch
                patch = image_features[:, p_row:p_row+patch_size, p_col:p_col+patch_size, :]
                # Flatten the patch to (batch_size, patch_size*patch_size, hidden_dim)
                patch = patch.contiguous().view(batch_size, -1, hidden_dim)
                # Append to the list
                rearranged_patches.append(patch)

        # Concatenate all patches along the sequence dimension
        image_features = torch.cat(rearranged_patches, dim=1)  # Shape: (batch_size, 576, hidden_dim)

        return image_features
    def extract_multilayer_features(self, vision_tower, images):
        """
        Extract and average features from all layers of the CLIP vision tower.
        
        Args:
            vision_tower: The CLIP vision tower model
            images: Input images to be processed
            
        Returns:
            torch.Tensor: Averaged features across all layers before projection
        """
        vision_tower = vision_tower.vision_tower  # Get the actual CLIP vision model
        
        # Get embeddings
        hidden_states = vision_tower.vision_model.embeddings(images)
        hidden_states = vision_tower.vision_model.pre_layrnorm(hidden_states)
        
        # Collect outputs from all layers
        all_layer_outputs = []
        for layer in vision_tower.vision_model.encoder.layers:
            residual = hidden_states
            
            # Self attention
            hidden_states = layer.layer_norm1(hidden_states)
            hidden_states = layer.self_attn(hidden_states)[0]
            hidden_states = residual + hidden_states
            
            # MLP
            residual = hidden_states
            hidden_states = layer.layer_norm2(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states
            
            # Apply post layer norm and collect output
            layer_output = vision_tower.vision_model.post_layernorm(hidden_states)
            all_layer_outputs.append(layer_output)
        
        # Stack and average all layer outputs
        # Shape: [batch_size, num_layers, num_patches, hidden_dim]
        stacked_outputs = torch.stack(all_layer_outputs, dim=1)
        
        # Average across layers
        # Shape: [batch_size, num_patches, hidden_dim]
        # averaged_outputs = torch.mean(stacked_outputs, dim=1)
        
        #need to remove the cls token from the CLIP output
        stacked_outputs = stacked_outputs[:, :, 1:, :]
        
        return stacked_outputs

    def encode_images(self, images, permutation=False):
        vision_tower = self.get_model().get_vision_tower()
    
        # Get configuration for feature extraction method
        use_multilayer = getattr(self.config, 'use_multilayer_features', False)
        # use_multilayer = True
        
        if use_multilayer:
            # Extract and average features from all layers
            multilayer_image_features = self.extract_multilayer_features(vision_tower, images)
            # print("Using multi-layer features")
            #use 12, 16, 20, 24 layers and concatenate them
            # Extract features from specific layers (12, 16, 20, 24)
            image_features = torch.cat([
                multilayer_image_features[:, 11, :, :],  # layer 12
                multilayer_image_features[:, 15, :, :],  # layer 16
                multilayer_image_features[:, 19, :, :],  # layer 20
                multilayer_image_features[:, 23, :, :]   # layer 24
            ], dim=-1)  # Concatenate along the last dimension
            # print(f"After concatenation shape: {image_features.shape}")
            
        else:
            # Use standard single-layer features
            image_features = vision_tower(images)
        #to same dtype
        image_features = image_features.to(self.dtype)
        
        if getattr(self.config, 'add_registers', False):
            projected_features, orig_length = self.get_model().mm_projector(image_features)
            image_features = projected_features[:, :orig_length, :]
        else:
            image_features = self.get_model().mm_projector(image_features)
        #permute the image features dimension 1 to random order (uncomment to use this)
        if permutation == True:
            # print("----------WARNING: Permuting vision token order----------")
            image_features = image_features[:, torch.randperm(image_features.shape[1]), :]
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None,
        tokenizer=None, permutation=False
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images, permutation=permutation)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images, permutation=permutation)
        
        rearrange_image_features = getattr(self.config, 'rearrange_image_features', False)
        if rearrange_image_features:
            image_features = self.rearrange_image_features(image_features)
        
        interleave = getattr(self.config, 'interleave', False)
        
        if interleave:
            # print("Interleaving spatial embeddings with image patches")
            sparsity = getattr(self.config, 'interleave_sparsity', 4)  # Default sparsity=4
            # Initialize the SpatialTokenInterleaver
            spatial_interleaver = SpatialTokenInterleaver(
                tokenizer= tokenizer,
                embedding_layer=self.get_model().embed_tokens,
            )


        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        batch_img_token_mask_list = []  # New list to track image token positions
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                # Add mask for no images
                batch_img_token_mask_list.append(torch.zeros(cur_input_embeds.size(0), dtype=torch.bool, device=cur_input_embeds.device))
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_img_token_mask = []  # Track image tokens for current batch

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                # Add mask for text tokens
                cur_img_token_mask.append(torch.zeros(cur_input_embeds_no_im[i].size(0), dtype=torch.bool, device=cur_input_embeds_no_im[i].device))
                
                
                if i < num_images:
                    if interleave:
                        # Get the image features for the current image
                        cur_image_features = image_features[cur_image_idx]
                        num_patches = cur_image_features.size(0)

                        # Interleave spatial embeddings sparsely
                        interleaved_features = []
                        interleaved_labels = []
                        interleaved_img_mask = []

                        for patch_idx in range(num_patches):
                            # Append image patch embedding
                            interleaved_features.append(cur_image_features[patch_idx:patch_idx+1])
                            interleaved_labels.append(torch.tensor([IGNORE_INDEX], device=cur_image_features.device))
                            interleaved_img_mask.append(torch.tensor([True], device=cur_image_features.device))

                            # Check if this patch should have a spatial token based on sparsity
                            if (patch_idx + 1) % spatial_interleaver.sparsity == 0:
                                # Append spatial embedding
                                spatial_embedding = spatial_interleaver.get_position_description(patch_idx)
                                desc_tokens = tokenizer(
                                    spatial_embedding,
                                    add_special_tokens=False,
                                    return_tensors="pt"
                                )["input_ids"].squeeze(0).to(cur_image_features.device)
                                desc_tokens = desc_tokens[1:]
                                desc_embeddings = self.get_model().embed_tokens(desc_tokens)
                                interleaved_features.append(desc_embeddings)
                                interleaved_labels.append(torch.full((desc_embeddings.size(0),), IGNORE_INDEX, device=cur_image_features.device))
                                interleaved_img_mask.append(torch.zeros(desc_embeddings.size(0), dtype=torch.bool, device=cur_image_features.device))

                        # Concatenate interleaved features, labels, and masks
                        if interleaved_features:
                            interleaved_features = torch.cat(interleaved_features, dim=0)
                            interleaved_labels = torch.cat(interleaved_labels, dim=0)
                            interleaved_img_mask = torch.cat(interleaved_img_mask, dim=0)

                            cur_new_input_embeds.append(interleaved_features)
                            cur_new_labels.append(interleaved_labels)
                            cur_img_token_mask.append(interleaved_img_mask)

                        cur_image_idx += 1
                    else:
                        #original code:
                        cur_image_features = image_features[cur_image_idx]
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        # Add mask for image tokens
                        cur_img_token_mask.append(torch.ones(cur_image_features.size(0), dtype=torch.bool, device=cur_image_features.device))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_img_token_mask = torch.cat(cur_img_token_mask) # Concatenate image token masks

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            batch_img_token_mask_list.append(cur_img_token_mask) # Append image token masks

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            batch_img_token_mask_list = [x[:tokenizer_model_max_length] for x in batch_img_token_mask_list] # Truncate image token masks

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        batch_img_token_mask_padded = torch.zeros((batch_size, max_len), dtype=torch.bool, device=self.device) # Initialize image token mask

        for i, (cur_new_embed, cur_new_labels, cur_img_token_mask) in enumerate(zip(new_input_embeds, new_labels, batch_img_token_mask_list)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    batch_img_token_mask_padded[i, -cur_len:] = cur_img_token_mask # Update image token mask
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
                    batch_img_token_mask_padded[i, :cur_len] = cur_img_token_mask # Update image token mask

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, batch_img_token_mask_padded
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False


    