import inspect
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


@dataclass
class VisionConfig:
    model_type: str = "glm4v_vision"
    num_hidden_layers: int = 24
    hidden_size: int = 1536
    intermediate_size: int = 4096
    merger_intermediate_size: int = 13696
    num_attention_heads: int = 12
    attention_bias: bool = False
    hidden_dropout_prob: float = 0.0
    initializer_range: float = 0.02
    image_size: int = 336
    patch_size: int = 14
    out_hidden_size: int = 4096
    rms_norm_eps: float = 1e-05
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    num_channels: int = 3
    layer_norm_eps: float = 1e-6

    @classmethod
    def from_dict(cls, params):
        params['intermediate_size'] = 4096
        return cls(**{k: v for k, v in params.items() if hasattr(cls, k)})


class VisualAttention(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.QuantizedLinear(self.embed_dim, 3 * self.embed_dim, bias=config.attention_bias)
        self.proj = nn.QuantizedLinear(self.embed_dim, self.embed_dim, bias=config.attention_bias)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, D = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, D)
        return self.proj(output)


class VisualMLP(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.QuantizedLinear(config.hidden_size, config.intermediate_size, bias=False) 
        self.down_proj = nn.QuantizedLinear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class VisualBlock(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.attn = VisualAttention(config)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.mlp = VisualMLP(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        return mx.flatten(x, start_axis=1, end_axis=2)


class VisualEmbeddings(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embedding = nn.QuantizedEmbedding(self.num_patches, config.hidden_size)

    def __call__(self, patch_embeds: mx.array) -> mx.array:
        position_ids = mx.array(np.arange(self.num_patches)[None, :])
        return patch_embeds + self.position_embedding(position_ids)


class VisualMerger(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        input_size = config.out_hidden_size
        self.gate_proj = nn.QuantizedLinear(input_size, config.merger_intermediate_size, bias=False)
        self.up_proj = nn.QuantizedLinear(input_size, config.merger_intermediate_size, bias=False)
        self.down_proj = nn.QuantizedLinear(config.merger_intermediate_size, config.out_hidden_size, bias=False)
        self.proj = nn.QuantizedLinear(input_size, config.out_hidden_size, bias=False)
        self.post_projection_norm = nn.LayerNorm(config.out_hidden_size, eps=config.rms_norm_eps, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        gate = nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        mlp_out = self.down_proj(gate * up)
        
        linear_out = self.proj(x)
        
        out = mlp_out + linear_out
        return self.post_projection_norm(out)


class VisionModel(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.patch_embed = PatchEmbed(config)
        self.embeddings = VisualEmbeddings(config)
        self.blocks = [VisualBlock(config) for _ in range(config.num_hidden_layers)]
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        # --- THIS IS THE CORRECTED LINE ---
        self.post_conv_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        
        self.downsample = nn.Conv2d(
            config.hidden_size,
            config.out_hidden_size,
            kernel_size=2,
            stride=2,
            bias=True
        )
        
        self.merger = VisualMerger(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.patch_embed(x)
        x = self.embeddings(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.post_layernorm(x)
        x = self.post_conv_layernorm(x)
        
        B, L, D = x.shape
        grid_size = int(L ** 0.5)
        x = x.reshape(B, grid_size, grid_size, D)
        x = x.transpose(0, 3, 1, 2)
        
        x = self.downsample(x)
        
        B, D_out, H_out, W_out = x.shape
        x = x.transpose(0, 2, 3, 1)
        x = x.reshape(B, H_out * W_out, D_out)
        
        x = self.merger(x)
        
        return x