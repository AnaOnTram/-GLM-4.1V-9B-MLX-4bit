import inspect
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    create_attention_mask,
    scaled_dot_product_attention,
)


@dataclass
class TextConfig:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    attention_bias: bool
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    partial_rotary_factor: float
    rope_theta: float
    rope_traditional: bool = True
    max_position_embeddings: int = 65536

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class Glm4MLP(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.gate_up_proj = nn.QuantizedLinear(
            args.hidden_size, 2 * args.intermediate_size, bias=False
        )
        self.down_proj = nn.QuantizedLinear(
            args.intermediate_size, args.hidden_size, bias=False
        )

    def __call__(self, x) -> mx.array:
        x = self.gate_up_proj(x)
        gate, up_states = mx.split(x, 2, axis=-1)
        return self.down_proj(nn.silu(gate) * up_states)


class Glm4Attention(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.scale = self.head_dim ** -0.5

        bias = args.attention_bias
        q_out = args.num_attention_heads * self.head_dim
        kv_out = args.num_key_value_heads * self.head_dim

        self.q_proj = nn.QuantizedLinear(args.hidden_size, q_out, bias=bias)
        self.k_proj = nn.QuantizedLinear(args.hidden_size, kv_out, bias=bias)
        self.v_proj = nn.QuantizedLinear(args.hidden_size, kv_out, bias=bias)
        self.o_proj = nn.QuantizedLinear(q_out, args.hidden_size, bias=False)

        self.rope = nn.RoPE(
            dims=int(self.head_dim * args.partial_rotary_factor),
            base=args.rope_theta,
            traditional=args.rope_traditional,
        )

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class Glm4DecoderLayer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.self_attn = Glm4Attention(args=args)
        self.mlp = Glm4MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_self_attn_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )
        self.post_mlp_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None
    ) -> mx.array:
        x = x + self.post_self_attn_layernorm(
            self.self_attn(self.input_layernorm(x), mask, cache)
        )
        residual = x
        x = (
            self.post_mlp_layernorm(self.mlp(self.post_attention_layernorm(x)))
            + residual
        )
        return x


class Glm4Model(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.embed_tokens = nn.QuantizedEmbedding(args.vocab_size, args.hidden_size)
        self.layers = [
            Glm4DecoderLayer(args=args) for _ in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(inputs)

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = Glm4Model(config)
        self.lm_head = nn.QuantizedLinear(config.hidden_size, config.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
    ):
        out = self.model(inputs, inputs_embeds=inputs_embeds, mask=mask, cache=cache)
        out = self.lm_head(out)
        return out

    @property
    def layers(self):
        return self.model.layers