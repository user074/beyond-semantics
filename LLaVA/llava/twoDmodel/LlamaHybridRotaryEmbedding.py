import math

import torch
from torch import nn

from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


class LlamaHybridRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        rope_type="default",
        image_size=None,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.rope_type = rope_type
        self.image_size = 24
        self.device = device

        if self.dim % 4 != 0:
            raise ValueError("Embedding dimension must be divisible by 4 for 2D rotary embeddings.")

        self.text_dim = self.dim
        self.img_dim = self.dim
        if self.device.type != 'meta':
            self.initialize_buffers(device=self.device, dtype=torch.get_default_dtype())

        # self._init_inv_freq()
        # self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=device, dtype=torch.get_default_dtype())

    def _init_inv_freq(self):
        device = self.device
        self.inv_freq_text = 1.0 / (self.base ** (torch.arange(0, self.text_dim, 2).float().to(device) / self.text_dim))
        self.register_buffer("inv_freq", self.inv_freq_text, persistent=True)

        self.inv_freq_image = self.inv_freq_text  # Adjust if needed
        self.register_buffer("inv_freq_img", self.inv_freq_image, persistent=True)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=True)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=True)

        img_seq_len = self.image_size
        t_img = torch.arange(img_seq_len, device=device, dtype=self.inv_freq_img.dtype)
        freqs_img = torch.outer(t_img, self.inv_freq_img)
        emb_img = torch.cat((freqs_img, freqs_img), dim=-1)
        self.register_buffer("cos_cached_img", emb_img.cos().to(dtype), persistent=True)
        self.register_buffer("sin_cached_img", emb_img.sin().to(dtype), persistent=True)

    def forward(self, x, seq_len=None, batch_img_token_mask=None, position_ids=None):
        if seq_len is None:
            seq_len = x.shape[2]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        batch_size, num_heads, seq_len, head_dim = x.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0)
        # Ensure position_ids has shape [batch_size, seq_len]
        if position_ids.shape[0] != batch_size:
            position_ids = position_ids.expand(batch_size, seq_len)

        # Compute the maximum position id to define the size of cos and sin tensors
        max_position_id = position_ids.max().item()
        if max_position_id >= self.max_seq_len_cached:
            # Ensure cos_cached and sin_cached are large enough
            self._set_cos_sin_cache(seq_len=max_position_id + 1, device=x.device, dtype=x.dtype)

        cos = torch.zeros(max_position_id + 1, head_dim, device=x.device, dtype=x.dtype)
        sin = torch.zeros_like(cos)

        if batch_img_token_mask is None or not batch_img_token_mask.any():
            # All tokens are text tokens
            position_ids_flat = position_ids.reshape(-1)
            cos[position_ids_flat] = self.cos_cached[position_ids_flat]
            sin[position_ids_flat] = self.sin_cached[position_ids_flat]
        else:
            # Text tokens
            text_mask = ~batch_img_token_mask
            if text_mask.any():
                position_ids_text = position_ids[text_mask]
                cos[position_ids_text] = self.cos_cached[position_ids_text]
                sin[position_ids_text] = self.sin_cached[position_ids_text]
            # Image tokens
            if batch_img_token_mask.any():
                position_ids_img = position_ids[batch_img_token_mask]
                cos_img, sin_img = self._get_2d_cos_sin(position_ids_img)
                cos[position_ids_img] = cos_img
                sin[position_ids_img] = sin_img

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _get_2d_cos_sin(self, position_ids_img):
        max_position_id = self.image_size * self.image_size - 1
        
        #use the relative position since image might not start at 0
        img_start_pos = position_ids_img.min()
        relative_pos = position_ids_img - img_start_pos
        
        if relative_pos.max() > max_position_id:
            print(f"exceed the max images size: {relative_pos.max()} > {max_position_id}")
            raise ValueError("position_ids_img contains invalid positions for image tokens.")
        row_ids = (relative_pos // self.image_size).long()
        col_ids = (relative_pos % self.image_size).long()

        cos_row = self.cos_cached_img[row_ids]
        sin_row = self.sin_cached_img[row_ids]
        cos_col = self.cos_cached_img[col_ids]
        sin_col = self.sin_cached_img[col_ids]

        half_head_dim = self.dim // 2
        cos_img = torch.cat([cos_row[:, :half_head_dim], cos_col[:, :half_head_dim]], dim=-1)
        sin_img = torch.cat([sin_row[:, :half_head_dim], sin_col[:, :half_head_dim]], dim=-1)

        return cos_img, sin_img
    
    def initialize_buffers(self, device=None, dtype=None):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = torch.get_default_dtype()
            
        self.device = device
        self._init_inv_freq()
        self._set_cos_sin_cache(seq_len=self.max_position_embeddings, device=device, dtype=dtype)