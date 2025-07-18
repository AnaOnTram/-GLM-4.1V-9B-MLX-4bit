import inspect
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Import CausalLMOutput from language.py where it is now defined
from .language import LanguageModel, TextConfig, CausalLMOutput
from .vision import VisionConfig, VisionModel


@dataclass
class ModelConfig:
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    vocab_size: int = 151338
    ignore_index: int = -100
    image_token_id: int = 151343
    video_token_id: int = 151344
    image_start_token_id: int = 151339
    image_end_token_id: int = 151340
    video_start_token_id: int = 151341
    video_end_token_id: int = 151342
    pad_token_id: int = 151329
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


class GLM4VMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.mm_input_projection_weight = mx.zeros(
            (config.vision_config.out_hidden_size, config.text_config.hidden_size)
        )
        self.mm_soft_emb_norm = nn.LayerNorm(
            config.vision_config.out_hidden_size, eps=config.vision_config.rms_norm_eps
        )


def masked_scatter(
    final_embedding: mx.array,
    image_mask_expanded: mx.array,
    scaled_image_features: mx.array,
):
    final_embedding_shape = final_embedding.shape
    scaled_image_features_flattened = mx.flatten(scaled_image_features)
    final_embedding_flattened = mx.flatten(final_embedding)
    image_mask_expanded_flattened = mx.flatten(image_mask_expanded)

    image_positions = mx.array(np.where(image_mask_expanded_flattened)[0], mx.uint32)
    final_embedding_flattened[image_positions] = scaled_image_features_flattened

    return mx.reshape(final_embedding_flattened, final_embedding_shape)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model_type = config.model_type
        self.config = config

        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config)
        self.multi_modal_projector = GLM4VMultiModalProjector(config)

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ):
        if pixel_values is None:
            # This case is for text-only generation in a loop
            return self.language_model.model.embed_tokens(input_ids), None

        # This case is for the first multi-modal input
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        vision_outputs = self.vision_tower(
            pixel_values.transpose(0, 2, 3, 1).astype(inputs_embeds.dtype)
        )

        vision_outputs = mx.einsum(
            "btm,md->btd",
            self.multi_modal_projector.mm_soft_emb_norm(vision_outputs),
            self.multi_modal_projector.mm_input_projection_weight,
        )

        final_inputs_embeds = self.prepare_inputs_for_multimodal(
            vision_outputs, inputs_embeds, input_ids
        )
        return final_inputs_embeds, None

    @staticmethod
    def prepare_inputs_for_multimodal(vision_outputs, inputs_embeds, input_ids):
        _, _, embed_dim = vision_outputs.shape
        batch_size, sequence_length = input_ids.shape

        scaled_vision_features = vision_outputs / (embed_dim**0.5)
        final_embedding = mx.zeros((batch_size, sequence_length, embed_dim))

        text_mask = (input_ids != ModelConfig.image_token_id) & (
            input_ids != ModelConfig.pad_token_id
        )
        image_mask = input_ids == ModelConfig.image_token_id
        pad_mask = input_ids == ModelConfig.pad_token_id

        text_mask_expanded = mx.repeat(mx.expand_dims(text_mask, -1), embed_dim, axis=-1)
        pad_mask_expanded = mx.repeat(mx.expand_dims(pad_mask, -1), embed_dim, axis=-1)
        image_mask_expanded = mx.repeat(
            mx.expand_dims(image_mask, -1), embed_dim, axis=-1
        )

        final_embedding = mx.where(text_mask_expanded, inputs_embeds, final_embedding)
        final_embedding = mx.where(
            pad_mask_expanded, mx.zeros_like(final_embedding), final_embedding
        )

        final_embedding = masked_scatter(
            final_embedding, image_mask_expanded, scaled_vision_features
        )

        return final_embedding.astype(inputs_embeds.dtype)

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,  # <-- THIS IS THE FINAL FIX
    ):
        if inputs_embeds is None:
            inputs_embeds, _ = self.get_input_embeddings(
                input_ids, pixel_values, mask
            )
        
        return self.language_model(
            inputs=input_ids,
            cache=cache,
            inputs_embeds=inputs_embeds,
        )

    def sanitize(self, weights):
        final_weights = {}
        for k, v in weights.items():
            new_k = k
            if k.startswith("model.visual."):
                new_k = k.replace("model.visual.", "vision_tower.")
            elif k.startswith("model.language_model."):
                new_k = k.replace("model.language_model.", "language_model.model.")
            elif k.startswith("lm_head"):
                new_k = k.replace("lm_head", "language_model.lm_head")
            final_weights[new_k] = v

        if "multi_modal_projector.mm_input_projection_weight" not in final_weights:
            final_weights["multi_modal_projector.mm_input_projection_weight"] = self.multi_modal_projector.mm_input_projection_weight
        if "multi_modal_projector.mm_soft_emb_norm.weight" not in final_weights:
            final_weights["multi_modal_projector.mm_soft_emb_norm.weight"] = self.multi_modal_projector.mm_soft_emb_norm.weight
        if "multi_modal_projector.mm_soft_emb_norm.bias" not in final_weights:
            final_weights["multi_modal_projector.mm_soft_emb_norm.bias"] = self.multi_modal_projector.mm_soft_emb_norm.bias
            
        return final_weights
