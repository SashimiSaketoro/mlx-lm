# Copyright © 2024 Apple Inc.

"""Model wrapper that streams transformer layers from disk."""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask

from .layer_loader import RollingWindowLoader


class StreamingModelWrapper:
    """
    Wraps an mlx-lm model and loads layer weights on demand.

    Compatible with ``mlx_lm.generate`` — exposes the same ``__call__`` signature
    and delegates KV cache handling to the underlying architecture.

    The wrapped model is stored privately (not as an nn.Module child) to avoid
    conflicting with MLX parameter tracking during per-layer load_weights.
    """

    def __init__(self, model: nn.Module, loader: RollingWindowLoader):
        self._model = model
        self.loader = loader
        self._streaming_enabled = self._detect_streaming_support()

    def _detect_streaming_support(self) -> bool:
        inner = getattr(self._model, "model", None)
        return inner is not None and hasattr(inner, "layers")

    @property
    def args(self):
        return self._model.args

    def make_cache(self):
        if hasattr(self._model, "make_cache"):
            return self._model.make_cache()
        inner = self._model.model
        return [None] * len(inner.layers)

    def sanitize(self, weights):
        if hasattr(self._model, "sanitize"):
            return self._model.sanitize(weights)
        return weights

    def set_dtype(self, dtype):
        if hasattr(self._model, "set_dtype"):
            self._model.set_dtype(dtype)
        return self

    def __getattr__(self, name):
        if name in ("_model", "loader", "_streaming_enabled"):
            raise AttributeError(name)
        return getattr(self._model, name)

    def get_stats(self) -> dict:
        """Memory and window statistics for streaming inference."""
        usage = self.loader.get_memory_usage()
        return {
            "streaming": {
                "window_size": usage["window_size"],
                "loaded_layers": usage["loaded_layers"],
                "layer_indices": usage["layer_indices"],
                "total_mb": usage["total_mb"],
            },
        }

    def _forward_streaming(
        self,
        inputs: mx.array,
        cache: Optional[List],
        input_embeddings: Optional[mx.array],
    ) -> mx.array:
        m = self._model.model
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = m.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(m.layers)

        fa_idx = getattr(m, "fa_idx", 0)
        swa_idx = getattr(m, "swa_idx", None)
        sliding_window = getattr(m, "sliding_window", None)

        fa_mask = create_attention_mask(h, cache[fa_idx])
        swa_mask = None
        if swa_idx is not None:
            swa_mask = create_attention_mask(
                h, cache[swa_idx], window_size=sliding_window
            )

        for i, (layer, layer_cache) in enumerate(zip(m.layers, cache)):
            layer_weights = self.loader.get_layer(i)
            layer.load_weights(list(layer_weights.weights.items()), strict=False)
            mx.eval(layer.parameters())

            mask = swa_mask if getattr(layer, "use_sliding", False) else fa_mask
            h = layer(h, mask, cache=layer_cache)

        h = m.norm(h)

        if self._model.args.tie_word_embeddings:
            return m.embed_tokens.as_linear(h)
        return self._model.lm_head(h)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if self._streaming_enabled:
            return self._forward_streaming(inputs, cache, input_embeddings)
        return self._model(inputs, cache=cache, input_embeddings=input_embeddings)