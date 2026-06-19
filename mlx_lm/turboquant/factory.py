# Copyright © 2025 Bonsai Demo contributors.

from __future__ import annotations

from typing import List, Optional

import mlx.nn as nn

from mlx_lm.turboquant.cache import AsymmetricTurboQuantCache


def _layer_count(model: nn.Module) -> int:
    if hasattr(model, "layers"):
        return len(model.layers)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    raise ValueError("Could not determine transformer layer count for TurboQuant cache")


def make_turboquant_cache(
    model: nn.Module,
    *,
    max_kv_size: Optional[int] = None,
    k_bits: int = 4,
    v_bits: int = 3,
    fp16_layers: int = 4,
    head_dim: int = 128,
    seed: int = 42,
) -> List:
    """Build a per-layer cache list with TurboQuant on middle layers.

    First and last ``fp16_layers`` use standard FP16 ``KVCache`` (or
    ``RotatingKVCache`` when ``max_kv_size`` is set). Middle layers use
    ``AsymmetricTurboQuantCache``.
    """
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    num_layers = _layer_count(model)
    if fp16_layers < 0:
        raise ValueError("fp16_layers must be >= 0")
    if 2 * fp16_layers >= num_layers:
        raise ValueError(
            f"fp16_layers={fp16_layers} leaves no middle layers for TurboQuant "
            f"(model has {num_layers} layers)"
        )
    caches = []
    for i in range(num_layers):
        if i < fp16_layers or i >= num_layers - fp16_layers:
            if max_kv_size is not None:
                caches.append(RotatingKVCache(max_size=max_kv_size, keep=4))
            else:
                caches.append(KVCache())
        else:
            caches.append(
                AsymmetricTurboQuantCache(
                    head_dim=head_dim,
                    k_bits=k_bits,
                    v_bits=v_bits,
                    seed=seed + i,
                )
            )
    return caches