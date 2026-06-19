# Copyright © 2025 Bonsai Demo contributors.

"""Bit packing for low-bit TurboQuant index streams."""

from __future__ import annotations

import math

import mlx.core as mx

_BITS_PER_WORD = 32


def packed_dim(dim: int, bits: int) -> int:
    return math.ceil(dim * bits / _BITS_PER_WORD)


def pack_indices(indices: mx.array, bits: int) -> mx.array:
    """Pack unsigned indices along the last axis into uint32 words."""
    if bits not in (1, 2, 3, 4):
        raise ValueError(f"pack_indices supports 1-4 bits, got {bits}")
    *batch, dim = indices.shape
    pdim = packed_dim(dim, bits)
    flat = indices.reshape(-1, dim).astype(mx.uint32)
    n = flat.shape[0]
    words = mx.zeros((n, pdim), dtype=mx.uint32)
    mask = (1 << bits) - 1
    for i in range(dim):
        word = i * bits // _BITS_PER_WORD
        shift = (i * bits) % _BITS_PER_WORD
        chunk = (flat[:, i] & mask) << shift
        words[:, word] = words[:, word] | chunk
        if shift + bits > _BITS_PER_WORD:
            spill = bits - (_BITS_PER_WORD - shift)
            words[:, word + 1] = words[:, word + 1] | ((flat[:, i] & mask) >> (_BITS_PER_WORD - shift))
    return words.reshape(*batch, pdim)


def unpack_indices(packed: mx.array, bits: int, dim: int) -> mx.array:
    if bits not in (1, 2, 3, 4):
        raise ValueError(f"unpack_indices supports 1-4 bits, got {bits}")
    *batch, pdim = packed.shape
    flat = packed.reshape(-1, pdim).astype(mx.uint32)
    n = flat.shape[0]
    out = mx.zeros((n, dim), dtype=mx.uint32)
    mask = (1 << bits) - 1
    for i in range(dim):
        word = i * bits // _BITS_PER_WORD
        shift = (i * bits) % _BITS_PER_WORD
        val = (flat[:, word] >> shift) & mask
        if shift + bits > _BITS_PER_WORD:
            spill = bits - (_BITS_PER_WORD - shift)
            val = val | ((flat[:, word + 1] & ((1 << spill) - 1)) << (_BITS_PER_WORD - shift))
        out[:, i] = val
    return out.reshape(*batch, dim).astype(mx.uint8)