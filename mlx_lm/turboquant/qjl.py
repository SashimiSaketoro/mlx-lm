# Copyright © 2025 Bonsai Demo contributors.

"""Quantized Johnson-Lindenstrauss (QJL) 1-bit residual quantizer."""

from __future__ import annotations

import math

import mlx.core as mx

from mlx_lm.turboquant.packing import pack_indices, packed_dim, unpack_indices


def make_qjl_matrix(dim: int, seed: int) -> mx.array:
    mx.random.seed(seed + 1_000_003)
    with mx.stream(mx.cpu):
        s = mx.random.normal(shape=(dim, dim)).astype(mx.float32)
        mx.eval(s)
    return s


def qjl_packed_dim(dim: int) -> int:
    """Packed uint32 words for 1-bit QJL signs along ``dim``."""
    return packed_dim(dim, 1)


def quantize_qjl(residual: mx.array, s_matrix: mx.array) -> mx.array:
    """Pack 1-bit QJL signs; last axis is ``qjl_packed_dim(dim)`` uint32 words."""
    dim = residual.shape[-1]
    projected = residual @ s_matrix.T
    bits = (projected >= 0).astype(mx.uint8)
    return pack_indices(bits, bits=1)


def dequantize_qjl(
    packed_signs: mx.array, gamma: mx.array, s_matrix: mx.array, dim: int
) -> mx.array:
    """Unbiased QJL reconstruction scaled by residual norm gamma."""
    scale = math.sqrt(math.pi / 2.0) / dim
    out_shape = packed_signs.shape[:-1] + (dim,)
    bits = unpack_indices(packed_signs, bits=1, dim=dim)
    flat_signs = bits.reshape(-1, dim).astype(mx.float32) * 2.0 - 1.0
    flat_gamma = gamma.reshape(-1, 1)
    out = scale * flat_gamma * (flat_signs @ s_matrix)
    return out.reshape(out_shape)