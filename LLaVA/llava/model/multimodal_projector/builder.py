import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    
    
import torch
import torch.nn as nn

class PoolingProjector(nn.Module):
    def __init__(self, projector, config):
        super().__init__()
        self.input_dim = config.mm_hidden_size
        self.hidden_size = config.hidden_size
        num_vision_tokens = getattr(config, 'num_vision_tokens', 576)
        self.input_size = int(num_vision_tokens ** 0.5)
        self.output_size = config.final_spatial_size  # Assuming you want to reduce by a factor of 2

        # Linear projection
        self.proj = projector
        self.pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))
        print(f"PoolingProjector initialized")

    def forward(self, x):
        batch_size = x.size(0)

        # Project to hidden size
        x = self.proj(x)  # Shape: (B, N_in, hidden_size)

        # Reshape to (B, hidden_size, H, W)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, self.hidden_size, self.input_size, self.input_size)

       # Apply adaptive average pooling
        x = self.pool(x)  # (B, hidden_size, H_out, W_out)

        # Flatten back to (B, N_out, hidden_size)
        x = x.view(batch_size, self.hidden_size, -1)
        x = x.transpose(1, 2)  # Shape: (B, N_out, hidden_size)

        return x

class VisionProjectorWithRegisters(nn.Module):
    def __init__(self, projector, config):
        super().__init__()
        self.projector = projector
        self.num_registers = getattr(config, 'num_registers', 4)  # Default 4 registers
        self.hidden_size = config.hidden_size
        
        # Initialize learnable register tokens
        self.registers = nn.Parameter(torch.zeros(1, self.num_registers, self.hidden_size))
        nn.init.normal_(self.registers, std=0.02)  # Initialize with small random values
        print(f"VisionProjectorWithRegisters initialized with {self.num_registers} registers")

    def forward(self, image_features):
        batch_size = image_features.shape[0]
        
        # Project the input features
        projected_features = self.projector(image_features)  # [B, L, hidden_size]
        
        # Expand registers to batch size
        registers = self.registers.expand(batch_size, -1, -1)  # [B, num_registers, hidden_size]
        
        # Concatenate registers with projected features
        features_with_registers = torch.cat([projected_features, registers], dim=1)
        
        return features_with_registers, projected_features.shape[1]

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm/LlamaRMSNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class NormalizedProjector(nn.Module):
    def __init__(self, projector, config):
        super().__init__()
        self.projector = projector
        self.norm = RMSNorm(config.hidden_size)
        
        # Target statistics from LLM embeddings
        self.target_mean = 0.83 # Based on LLM embedding mean
        self.max_norm = 1.22  # Based on LLM embedding max
        
    def forward(self, x):
        # Project first
        x = self.projector(x)
        
        # Apply RMSNorm
        x = self.norm(x)
        
        # Get current norms
        norms = torch.norm(x, dim=-1, keepdim=True)
        
        # Clip to max norm if needed
        scale = torch.minimum(self.max_norm / norms, torch.ones_like(norms))
        x = x * scale
        
        # Scale to match target mean
        x = x * self.target_mean
        
        return x


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    add_spatial_coordinates = getattr(config, 'add_spatial_coordinates', False)
    print(f'spatial coordinates: {add_spatial_coordinates}')
    add_registers = getattr(config, 'add_registers', False)
    use_multilayer = getattr(config, 'use_multilayer_features', False)
    
    mm_hidden_size = config.mm_hidden_size
    hidden_size = config.hidden_size

    if add_spatial_coordinates:
        mm_hidden_size += 2  # Increase input dimension by 2 for (x, y)
    if use_multilayer:
        mm_hidden_size *= 4  # Multiply by 4 since we're concatenating 4 layers
        print(f'Using multilayer features. Input size adjusted to: {mm_hidden_size}')
        
    if projector_type == 'linear':
        projector = nn.Linear(mm_hidden_size, hidden_size)
    elif projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(mm_hidden_size, hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(hidden_size, hidden_size))
            projector = nn.Sequential(*modules)
        else:
            raise ValueError(f'Unknown projector type: {projector_type}')
    elif projector_type == 'identity':
        projector = IdentityMap()
    else:
        raise ValueError(f'Unknown projector type: {projector_type}')

    if add_registers:
        projector = VisionProjectorWithRegisters(projector, config)
    
    if getattr(config, 'normalize_projector', False):
        projector = NormalizedProjector(projector, config)
        
    # Wrap the projector to include spatial coordinate concatenation
    if getattr(config, 'compression_after_projection', False):
        pooling_projector = PoolingProjector(projector, config)
        return pooling_projector
    else:
        if add_spatial_coordinates:
            return VisionProjectorWithSpatial(projector, config)
        else:
            return projector

class VisionProjectorWithSpatial(nn.Module):
    def __init__(self, projector, config):
        super().__init__()
        self.projector = projector
        self.config = config
        print(f"VisionProjectorWithSpatial initialized")

    def forward(self, image_features):
        batch_size, num_patches, mm_hidden_size = image_features.shape

        # Generate normalized coordinates
        height = width = int(num_patches ** 0.5)  # Assuming a square grid
        x_coords = torch.linspace(-1, 1, steps=width, device=image_features.device)
        y_coords = torch.linspace(-1, 1, steps=height, device=image_features.device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords)
        x_grid = x_grid.reshape(-1)  # [num_patches]
        y_grid = y_grid.reshape(-1)  # [num_patches]
        coords = torch.stack((x_grid, y_grid), dim=1)  # [num_patches, 2]
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_patches, 2]
        #turn them into the dtype of the image features
        coords = coords.to(image_features.dtype)

        # Concatenate coordinates to image features
        image_features = torch.cat([image_features, coords], dim=2)  # [batch_size, num_patches, mm_hidden_size + 2]

        # Pass through the projector
        projected_features = self.projector(image_features)  # [batch_size, num_patches, hidden_size]

        return projected_features