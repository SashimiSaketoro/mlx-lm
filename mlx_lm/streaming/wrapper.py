# Copyright © 2024 Apple Inc.

"""Model wrapper that streams transformer layers from disk."""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask

from .layer_loader import RollingWindowLoader


class StreamingModelWrapper(nn.Module):
    """
    Wraps an mlx-lm model and loads layer weights on demand.

    Compatible with ``mlx_lm.generate`` — exposes the same ``__call__`` signature
    and delegates KV cache handling to the underlying architecture.
    """

    def __init__(self, model: nn.Module, loader: RollingWindowLoader):
        super().__init__()
        self.inner = model
        object.__setattr__(self, "loader", loader)
        self._streaming_enabled = self._detect_streaming_support()

    def _detect_streaming_support(self) -> bool:
        inner = getattr(self.inner, "model", None)
        return inner is not None and hasattr(inner, "layers")

    @property
    def args(self):
        return self.inner.args

    def make_cache(self):
        if hasattr(self.inner, "make_cache"):
            return self.inner.make_cache()
        inner = self.inner.model
        return [None] * len(inner.layers)

    def sanitize(self, weights):
        if hasattr(self.inner, "sanitize"):
            return self.inner.sanitize(weights)
        return weights

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def _forward_streaming(
        self,
        inputs: mx.array,
        cache: Optional[List],
        input_embeddings: Optional[mx.array],
    ) -> mx.array:
        m = self.inner.model
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

        if self.inner.args.tie_word_embeddings:
            return m.embed_tokens.as_linear(h)
        return self.inner.lm_head(h)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if self._streaming_enabled:
            return self._forward_streaming(inputs, cache, input_embeddings)
        return self.inner(inputs, cache=cache, input_embeddings=input_embeddings)