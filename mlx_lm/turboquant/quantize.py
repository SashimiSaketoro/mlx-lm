# Copyright © 2025 Bonsai Demo contributors.

"""TurboQuant encode/decode primitives."""

from __future__ import annotations

import mlx.core as mx

from mlx_lm.turboquant.codebooks import dequantize_coords, quantize_coords
from mlx_lm.turboquant.kernels import (
    decode_mse_metal,
    decode_prod_metal,
    encode_kv_metal,
    encode_mse_metal,
    encode_prod_metal,
    metal_available,
)
from mlx_lm.turboquant.packing import pack_indices, unpack_indices
from mlx_lm.turboquant.qjl import dequantize_qjl, qjl_packed_dim, quantize_qjl

_USE_METAL = metal_available()


def _decode_mse_impl(
    packed: mx.array,
    norms: mx.array,
    rotation: mx.array,
    bits: int,
    dim: int,
    dtype: mx.dtype,
) -> mx.array:
    if _USE_METAL:
        return decode_mse_metal(packed, norms, rotation, bits, dim, dtype)
    out_shape = packed.shape[:-1] + (dim,)
    indices = unpack_indices(packed, bits, dim)
    flat = indices.reshape(-1, dim)
    rotated = dequantize_coords(flat, bits)
    unit = _inverse_rotate(rotated, rotation)
    scaled = unit * norms.reshape(-1, 1)
    return scaled.reshape(out_shape)


def _decode_prod_impl(
    mse_packed: mx.array,
    norms: mx.array,
    qjl_signs: mx.array,
    gamma: mx.array,
    rotation: mx.array,
    s_matrix: mx.array,
    bits: int,
    dim: int,
    dtype: mx.dtype,
) -> mx.array:
    if _USE_METAL:
        return decode_prod_metal(
            mse_packed, norms, qjl_signs, gamma, rotation, s_matrix, bits, dim, dtype
        )
    mse_bits = bits - 1
    ones = mx.ones(norms.shape, dtype=norms.dtype)
    mse_unit = _decode_mse_impl(mse_packed, ones, rotation, mse_bits, dim, mx.float32)
    qjl_part = dequantize_qjl(qjl_signs, gamma, s_matrix, dim)
    return (norms * (mse_unit + qjl_part)).astype(dtype)


def _rotate(vectors: mx.array, rotation: mx.array) -> mx.array:
    # vectors: (N, d), rotation: (d, d) — row-vector convention: y = x @ R.T
    return vectors @ rotation.T


def _inverse_rotate(vectors: mx.array, rotation: mx.array) -> mx.array:
    return vectors @ rotation


def _encode_mse_ref(vectors: mx.array, rotation: mx.array, bits: int):
    shape = vectors.shape
    dim = shape[-1]
    flat = vectors.astype(mx.float32).reshape(-1, dim)
    norms = mx.linalg.norm(flat, axis=-1, keepdims=True)
    unit = flat / mx.maximum(norms, 1e-8)
    rotated = _rotate(unit, rotation)
    indices = quantize_coords(rotated, bits).reshape(shape[:-1] + (dim,))
    packed = pack_indices(indices, bits)
    return packed, norms.reshape(shape[:-1] + (1,))


def encode_mse(vectors: mx.array, rotation: mx.array, bits: int):
    """Quantize vectors with TurboQuant_mse. Returns packed indices and norms."""
    shape = vectors.shape
    dim = shape[-1]
    if _USE_METAL:
        return encode_mse_metal(vectors, rotation, bits, dim)
    return _encode_mse_ref(vectors, rotation, bits)


def decode_mse(
    packed: mx.array, norms: mx.array, rotation: mx.array, bits: int, dim: int
) -> mx.array:
    return _decode_mse_impl(packed, norms, rotation, bits, dim, mx.float32)


def _encode_prod_ref(vectors: mx.array, rotation: mx.array, s_matrix: mx.array, bits: int):
    if bits < 2:
        raise ValueError("TurboQuant_prod requires bits >= 2")
    mse_bits = bits - 1
    shape = vectors.shape
    dim = shape[-1]
    flat = vectors.astype(mx.float32).reshape(-1, dim)
    norms = mx.linalg.norm(flat, axis=-1, keepdims=True)
    unit = flat / mx.maximum(norms, 1e-8)
    unit_batched = unit.reshape(shape)

    mse_packed, _ = _encode_mse_ref(unit_batched, rotation, mse_bits)
    mse_unit = _decode_mse_impl(
        mse_packed,
        mx.ones_like(norms.reshape(shape[:-1] + (1,))),
        rotation,
        mse_bits,
        dim,
        mx.float32,
    ).reshape(-1, dim)

    residual = unit - mse_unit
    gamma = mx.linalg.norm(residual, axis=-1, keepdims=True)
    qjl_signs = quantize_qjl(
        residual.reshape(shape), s_matrix
    ).reshape(shape[:-1] + (qjl_packed_dim(dim),))

    return (
        mse_packed,
        norms.reshape(shape[:-1] + (1,)),
        qjl_signs,
        gamma.reshape(shape[:-1] + (1,)),
    )


def encode_prod(vectors: mx.array, rotation: mx.array, s_matrix: mx.array, bits: int):
    """TurboQuant_prod: (bits-1) Lloyd-Max on the unit vector + 1-bit QJL residual."""
    if bits < 2:
        raise ValueError("TurboQuant_prod requires bits >= 2")
    dim = vectors.shape[-1]
    if _USE_METAL:
        return encode_prod_metal(vectors, rotation, s_matrix, bits, dim)
    return _encode_prod_ref(vectors, rotation, s_matrix, bits)


def encode_kv(
    keys: mx.array,
    values: mx.array,
    k_rotation: mx.array,
    v_rotation: mx.array,
    k_qjl: mx.array,
    k_bits: int,
    v_bits: int,
):
    """Fused TurboQuant_prod(K) + TurboQuant_mse(V) in one dispatch."""
    if k_bits < 2:
        raise ValueError("TurboQuant_prod requires k_bits >= 2")
    dim = keys.shape[-1]
    if _USE_METAL:
        return encode_kv_metal(
            keys, values, k_rotation, v_rotation, k_qjl, k_bits, v_bits, dim
        )
    k_out = encode_prod(keys, k_rotation, k_qjl, k_bits)
    v_out = encode_mse(values, v_rotation, v_bits)
    return (*k_out, *v_out)


def decode_prod(
    mse_packed: mx.array,
    norms: mx.array,
    qjl_signs: mx.array,
    gamma: mx.array,
    rotation: mx.array,
    s_matrix: mx.array,
    bits: int,
    dim: int,
) -> mx.array:
    return _decode_prod_impl(
        mse_packed, norms, qjl_signs, gamma, rotation, s_matrix, bits, dim, mx.float32
    )