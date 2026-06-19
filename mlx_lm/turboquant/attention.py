# Copyright © 2025 Bonsai Demo contributors.

"""Fused TurboQuant attention — packed KV, no persistent dequant buffers."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import mlx.core as mx

from mlx_lm.turboquant.codebooks import get_codebook
from mlx_lm.turboquant.kernels import (
    decode_mse_metal,
    metal_available,
    tq_sdpa_metal,
)
from mlx_lm.turboquant.packing import unpack_indices
from mlx_lm.turboquant.quantize import decode_mse, decode_prod

_QJL_SCALE = math.sqrt(math.pi / 2.0)


def qk_scores_vectorized(
    queries: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    k_qjl_signs: mx.array,
    k_gamma: mx.array,
    rotation: mx.array,
    s_matrix: mx.array,
    k_bits: int,
    dim: int,
    scale: float,
) -> mx.array:
    """Compute QK logits from packed keys without materializing K (for tests)."""
    mse_bits = k_bits - 1
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = k_packed.shape[1]
    S = k_packed.shape[2]
    n_repeats = n_q_heads // n_kv_heads

    q_rot = queries @ rotation.T
    q_s = queries @ s_matrix.T

    if n_repeats > 1:
        q_rot = q_rot.reshape(B, n_kv_heads, n_repeats, L, D)
        q_s = q_s.reshape(B, n_kv_heads, n_repeats, L, D)

    indices = unpack_indices(k_packed, mse_bits, dim)
    centroids, _ = get_codebook(mse_bits)
    rot_c = centroids[indices.astype(mx.int32)]

    mse_dot = mx.sum(
        q_rot[..., None, :] * rot_c[:, :, None, None, :, :], axis=-1
    )
    signs = unpack_indices(k_qjl_signs, 1, dim).astype(mx.float32) * 2.0 - 1.0
    qjl_dot = mx.sum(
        q_s[..., None, :] * signs[:, :, None, None, :, :], axis=-1
    )

    kn = k_norms[..., 0][:, :, None, None, :]
    kg = k_gamma[..., 0][:, :, None, None, :]
    scores = kn * (mse_dot + (_QJL_SCALE / dim) * kg * qjl_dot)

    if n_repeats > 1:
        scores = scores.reshape(B, n_q_heads, L, S)
    return scores * scale


def av_weighted_sum_vectorized(
    attn: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    rotation: mx.array,
    v_bits: int,
    dim: int,
) -> mx.array:
    """Reference: attn @ dequant(V) without materializing full V."""
    B, n_q_heads, L, S = attn.shape
    n_kv_heads = v_packed.shape[1]
    n_repeats = n_q_heads // n_kv_heads

    values_deq = decode_mse(v_packed, v_norms, rotation, v_bits, dim)
    if n_repeats > 1:
        values_deq = mx.repeat(values_deq, n_repeats, axis=1)
    return mx.matmul(attn, values_deq)


def _qk_scores_reference(
    queries: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    k_qjl_signs: mx.array,
    k_gamma: mx.array,
    rotation: mx.array,
    s_matrix: mx.array,
    k_bits: int,
    dim: int,
    scale: float,
) -> mx.array:
    keys = decode_prod(
        k_packed, k_norms, k_qjl_signs, k_gamma, rotation, s_matrix, k_bits, dim
    )
    n_repeats = queries.shape[1] // keys.shape[1]
    if n_repeats > 1:
        keys = mx.repeat(keys, n_repeats, axis=1)
    return mx.matmul(queries, keys.transpose(0, 1, 3, 2)) * scale


def _decode_values(
    v_packed: mx.array,
    v_norms: mx.array,
    rotation: mx.array,
    v_bits: int,
    dim: int,
    dtype: mx.dtype,
) -> mx.array:
    if metal_available():
        return decode_mse_metal(v_packed, v_norms, rotation, v_bits, dim, dtype=dtype)
    return decode_mse(v_packed, v_norms, rotation, v_bits, dim).astype(dtype)


def _sdpa_decode_fallback(
    queries: mx.array,
    keys: Tuple[mx.array, mx.array, mx.array, mx.array],
    values: Tuple[mx.array, mx.array],
    cache,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    """MLX fallback: ephemeral K/V decode + fast SDPA."""
    k_packed, k_norms, k_qjl_signs, k_gamma = keys
    v_packed, v_norms = values
    dtype = queries.dtype
    dim = queries.shape[-1]
    n_repeats = queries.shape[1] // k_packed.shape[1]

    keys_deq = decode_prod(
        k_packed, k_norms, k_qjl_signs, k_gamma,
        cache._k_rotation, cache._k_qjl, cache.k_bits, dim,
    ).astype(dtype)
    values_deq = _decode_values(
        v_packed, v_norms, cache._v_rotation, cache.v_bits, dim, dtype,
    )
    if n_repeats > 1:
        keys_deq = mx.repeat(keys_deq, n_repeats, axis=1)
        values_deq = mx.repeat(values_deq, n_repeats, axis=1)

    return mx.fast.scaled_dot_product_attention(
        queries, keys_deq, values_deq, scale=scale, mask=mask
    )


def turboquant_scaled_dot_product_attention(
    queries: mx.array,
    keys: Tuple[mx.array, mx.array, mx.array, mx.array],
    values: Tuple[mx.array, mx.array],
    cache,
    scale: float,
    mask: Optional[mx.array],
) -> mx.array:
    """SDPA from packed TurboQuant KV without persistent dequant buffers."""
    if metal_available():
        k_packed, k_norms, k_qjl_signs, k_gamma = keys
        v_packed, v_norms = values
        dim = queries.shape[-1]
        L = queries.shape[2]
        do_causal = mask == "causal" and L > 1
        return tq_sdpa_metal(
            queries,
            k_packed,
            k_norms,
            k_qjl_signs,
            k_gamma,
            v_packed,
            v_norms,
            cache._k_rotation,
            cache._v_rotation,
            cache._k_qjl,
            cache.k_bits,
            cache.v_bits,
            dim,
            scale,
            do_causal,
        )
    return _sdpa_decode_fallback(queries, keys, values, cache, scale, mask)