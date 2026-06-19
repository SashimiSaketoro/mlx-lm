# Copyright © 2025 Bonsai Demo contributors.

"""Metal kernels for TurboQuant encode/decode and fused attention."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional

import mlx.core as mx

from mlx_lm.turboquant.codebooks import get_codebook
from mlx_lm.turboquant.packing import packed_dim
from mlx_lm.turboquant.qjl import qjl_packed_dim

_QJL_SCALE = math.sqrt(math.pi / 2.0)


def metal_available() -> bool:
    return mx.metal.is_available()


@lru_cache(maxsize=None)
def _decode_mse_kernel(bits: int, dim: int):
    pdim = packed_dim(dim, bits)
    n_cent = 2**bits
    source = f"""
    constexpr int DIM = {dim};
    constexpr int BITS = {bits};
    constexpr int PDIM = {pdim};
    constexpr int N_CENT = {n_cent};
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;

    uint n = thread_position_in_grid.x;
    float norm = norms[n];
    uint packed_off = n * PDIM;

    float rotated[DIM];
    for (int i = 0; i < DIM; i++) {{
        int word = (i * BITS) / BITS_PER_WORD;
        int shift = (i * BITS) % BITS_PER_WORD;
        uint w = packed[packed_off + word];
        uint val = (w >> shift) & MASK;
        if (shift + BITS > BITS_PER_WORD) {{
            int spill = BITS - (BITS_PER_WORD - shift);
            uint w2 = packed[packed_off + word + 1];
            val = val | ((w2 & ((1u << spill) - 1u)) << (BITS_PER_WORD - shift));
        }}
        rotated[i] = centroids[val];
    }}

    for (int j = 0; j < DIM; j++) {{
        float sum = 0.0f;
        for (int i = 0; i < DIM; i++) {{
            sum += rotated[i] * rotation[i * DIM + j];
        }}
        out[n * DIM + j] = static_cast<T>(norm * sum);
    }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_decode_mse_{bits}b_{dim}d",
        input_names=["packed", "norms", "rotation", "centroids"],
        output_names=["out"],
        source=source,
    )


@lru_cache(maxsize=None)
def _decode_prod_kernel(mse_bits: int, dim: int):
    pdim = packed_dim(dim, mse_bits)
    qjl_pdim = qjl_packed_dim(dim)
    n_cent = 2**mse_bits
    scale = _QJL_SCALE / dim
    source = f"""
    constexpr int DIM = {dim};
    constexpr int MSE_BITS = {mse_bits};
    constexpr int PDIM = {pdim};
    constexpr int QJL_PDIM = {qjl_pdim};
    constexpr int N_CENT = {n_cent};
    constexpr uint MSE_MASK = (1u << MSE_BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;
    constexpr float QJL_SCALE = {scale}f;

    uint n = thread_position_in_grid.x;
    float norm = norms[n];
    float gamma = qjl_gamma[n];
    uint packed_off = n * PDIM;
    uint signs_off = n * QJL_PDIM;

    float mse_unit[DIM];
    for (int i = 0; i < DIM; i++) {{
        int word = (i * MSE_BITS) / BITS_PER_WORD;
        int shift = (i * MSE_BITS) % BITS_PER_WORD;
        uint w = mse_packed[packed_off + word];
        uint val = (w >> shift) & MSE_MASK;
        if (shift + MSE_BITS > BITS_PER_WORD) {{
            int spill = MSE_BITS - (BITS_PER_WORD - shift);
            uint w2 = mse_packed[packed_off + word + 1];
            val = val | ((w2 & ((1u << spill) - 1u)) << (BITS_PER_WORD - shift));
        }}
        mse_unit[i] = centroids[val];
    }}

    float unit[DIM];
    for (int j = 0; j < DIM; j++) {{
        float sum = 0.0f;
        for (int i = 0; i < DIM; i++) {{
            sum += mse_unit[i] * rotation[i * DIM + j];
        }}
        unit[j] = sum;
    }}

    float qjl[DIM];
    for (int j = 0; j < DIM; j++) {{
        float sum = 0.0f;
        for (int i = 0; i < DIM; i++) {{
            int word = i / BITS_PER_WORD;
            int shift = i % BITS_PER_WORD;
            uint bit = (qjl_signs[signs_off + word] >> shift) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            sum += sign * s_matrix[i * DIM + j];
        }}
        qjl[j] = QJL_SCALE * gamma * sum;
    }}

    for (int j = 0; j < DIM; j++) {{
        out[n * DIM + j] = static_cast<T>(norm * (unit[j] + qjl[j]));
    }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_decode_prod_{mse_bits}b_{dim}d",
        input_names=[
            "mse_packed",
            "norms",
            "qjl_signs",
            "qjl_gamma",
            "rotation",
            "centroids",
            "s_matrix",
        ],
        output_names=["out"],
        source=source,
    )


def _launch_decode_mse(
    packed: mx.array,
    norms: mx.array,
    rotation: mx.array,
    bits: int,
    dim: int,
    dtype: mx.dtype,
) -> mx.array:
    shape = packed.shape[:-1]
    n = int(math.prod(shape)) if shape else packed.shape[0]
    flat_packed = packed.reshape(n, packed.shape[-1])
    flat_norms = norms.reshape(n, 1).astype(mx.float32)
    centroids, _ = get_codebook(bits)
    kernel = _decode_mse_kernel(bits, dim)
    out = kernel(
        inputs=[flat_packed, flat_norms, rotation, centroids],
        template=[("T", dtype)],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
        output_shapes=[(n, dim)],
        output_dtypes=[dtype],
        stream=mx.gpu,
    )[0]
    return out.reshape(shape + (dim,))


def _launch_decode_prod(
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
    mse_bits = bits - 1
    shape = mse_packed.shape[:-1]
    n = int(math.prod(shape)) if shape else mse_packed.shape[0]
    flat_mse = mse_packed.reshape(n, mse_packed.shape[-1])
    flat_signs = qjl_signs.reshape(n, qjl_signs.shape[-1])
    flat_norms = norms.reshape(n, 1).astype(mx.float32)
    flat_gamma = gamma.reshape(n, 1).astype(mx.float32)
    centroids, _ = get_codebook(mse_bits)
    kernel = _decode_prod_kernel(mse_bits, dim)
    out = kernel(
        inputs=[
            flat_mse,
            flat_norms,
            flat_signs,
            flat_gamma,
            rotation,
            centroids,
            s_matrix,
        ],
        template=[("T", dtype)],
        grid=(n, 1, 1),
        threadgroup=(min(256, n), 1, 1),
        output_shapes=[(n, dim)],
        output_dtypes=[dtype],
        stream=mx.gpu,
    )[0]
    return out.reshape(shape + (dim,))


def decode_mse_metal(
    packed: mx.array,
    norms: mx.array,
    rotation: mx.array,
    bits: int,
    dim: int,
    dtype: Optional[mx.dtype] = None,
) -> mx.array:
    if not metal_available():
        raise RuntimeError("Metal is not available")
    if dtype is None:
        dtype = mx.float32
    return _launch_decode_mse(packed, norms, rotation, bits, dim, dtype)


def decode_prod_metal(
    mse_packed: mx.array,
    norms: mx.array,
    qjl_signs: mx.array,
    gamma: mx.array,
    rotation: mx.array,
    s_matrix: mx.array,
    bits: int,
    dim: int,
    dtype: Optional[mx.dtype] = None,
) -> mx.array:
    if not metal_available():
        raise RuntimeError("Metal is not available")
    if dtype is None:
        dtype = mx.float32
    return _launch_decode_prod(
        mse_packed, norms, qjl_signs, gamma, rotation, s_matrix, bits, dim, dtype
    )


def _pack_loop_source(
    var: str,
    bits_name: str,
    *,
    mask: str = "MASK",
    packed_buf: str = "packed",
    offset: str = "packed_off",
) -> str:
    """Metal snippet: pack per-coordinate indices into uint32 words."""
    return f"""
    for (int i = 0; i < DIM; i++) {{
        int word = (i * {bits_name}) / BITS_PER_WORD;
        int shift = (i * {bits_name}) % BITS_PER_WORD;
        uint val = {var}[i];
        uint chunk = (val & {mask}) << shift;
        {packed_buf}[{offset} + word] |= chunk;
        if (shift + {bits_name} > BITS_PER_WORD) {{
            int spill = {bits_name} - (BITS_PER_WORD - shift);
            {packed_buf}[{offset} + word + 1] |= (val & {mask}) >> (BITS_PER_WORD - shift);
        }}
    }}
    """


@lru_cache(maxsize=None)
def _encode_mse_kernel(bits: int, dim: int):
    pdim = packed_dim(dim, bits)
    n_cent = 2**bits
    pack_loop = _pack_loop_source("indices", "BITS")
    source = f"""
    constexpr int DIM = {dim};
    constexpr int BITS = {bits};
    constexpr int PDIM = {pdim};
    constexpr int N_CENT = {n_cent};
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;

    uint n = thread_position_in_grid.x;
    uint packed_off = n * PDIM;

    for (int w = 0; w < PDIM; w++) {{
        packed[packed_off + w] = 0u;
    }}

    float norm_sq = 0.0f;
    for (int j = 0; j < DIM; j++) {{
        float v = static_cast<float>(vectors[n * DIM + j]);
        norm_sq += v * v;
    }}
    float norm = sqrt(norm_sq);
    norms[n] = norm;
    float inv_norm = 1.0f / max(norm, 1e-8f);

    float unit[DIM];
    for (int j = 0; j < DIM; j++) {{
        unit[j] = static_cast<float>(vectors[n * DIM + j]) * inv_norm;
    }}

    uint indices[DIM];
    for (int i = 0; i < DIM; i++) {{
        float rot = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            rot += unit[j] * rotation[i * DIM + j];
        }}
        uint best = 0u;
        float best_d = abs(rot - centroids[0]);
        for (int c = 1; c < N_CENT; c++) {{
            float d = abs(rot - centroids[c]);
            if (d < best_d) {{
                best_d = d;
                best = uint(c);
            }}
        }}
        indices[i] = best;
    }}

    {pack_loop}
    """
    return mx.fast.metal_kernel(
        name=f"tq_encode_mse_{bits}b_{dim}d",
        input_names=["vectors", "rotation", "centroids"],
        output_names=["packed", "norms"],
        source=source,
    )


@lru_cache(maxsize=None)
def _encode_prod_kernel(bits: int, dim: int):
    mse_bits = bits - 1
    pdim = packed_dim(dim, mse_bits)
    qjl_pdim = qjl_packed_dim(dim)
    n_cent = 2**mse_bits
    mse_pack = _pack_loop_source(
        "mse_indices",
        "MSE_BITS",
        mask="MSE_MASK",
        packed_buf="mse_packed",
        offset="packed_off",
    )
    source = f"""
    constexpr int DIM = {dim};
    constexpr int MSE_BITS = {mse_bits};
    constexpr int PDIM = {pdim};
    constexpr int QJL_PDIM = {qjl_pdim};
    constexpr int N_CENT = {n_cent};
    constexpr uint MSE_MASK = (1u << MSE_BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;

    uint n = thread_position_in_grid.x;
    uint packed_off = n * PDIM;
    uint signs_off = n * QJL_PDIM;

    for (int w = 0; w < PDIM; w++) {{
        mse_packed[packed_off + w] = 0u;
    }}
    for (int w = 0; w < QJL_PDIM; w++) {{
        qjl_signs[signs_off + w] = 0u;
    }}

    float norm_sq = 0.0f;
    for (int j = 0; j < DIM; j++) {{
        float v = static_cast<float>(vectors[n * DIM + j]);
        norm_sq += v * v;
    }}
    float norm = sqrt(norm_sq);
    norms[n] = norm;
    float inv_norm = 1.0f / max(norm, 1e-8f);

    float unit[DIM];
    for (int j = 0; j < DIM; j++) {{
        unit[j] = static_cast<float>(vectors[n * DIM + j]) * inv_norm;
    }}

    uint mse_indices[DIM];
    float mse_rot[DIM];
    for (int i = 0; i < DIM; i++) {{
        float rot = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            rot += unit[j] * rotation[i * DIM + j];
        }}
        uint best = 0u;
        float best_d = abs(rot - centroids[0]);
        for (int c = 1; c < N_CENT; c++) {{
            float d = abs(rot - centroids[c]);
            if (d < best_d) {{
                best_d = d;
                best = uint(c);
            }}
        }}
        mse_indices[i] = best;
        mse_rot[i] = centroids[best];
    }}

    {mse_pack}

    float mse_unit[DIM];
    for (int j = 0; j < DIM; j++) {{
        float sum = 0.0f;
        for (int i = 0; i < DIM; i++) {{
            sum += mse_rot[i] * rotation[i * DIM + j];
        }}
        mse_unit[j] = sum;
    }}

    float residual[DIM];
    float gamma_sq = 0.0f;
    for (int j = 0; j < DIM; j++) {{
        residual[j] = unit[j] - mse_unit[j];
        gamma_sq += residual[j] * residual[j];
    }}
    qjl_gamma[n] = sqrt(gamma_sq);

    for (int i = 0; i < DIM; i++) {{
        float proj = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            proj += residual[j] * s_matrix[i * DIM + j];
        }}
        uint bit = proj >= 0.0f ? 1u : 0u;
        int word = i / BITS_PER_WORD;
        int shift = i % BITS_PER_WORD;
        qjl_signs[signs_off + word] |= bit << shift;
    }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_encode_prod_{bits}b_{dim}d",
        input_names=["vectors", "rotation", "centroids", "s_matrix"],
        output_names=["mse_packed", "norms", "qjl_signs", "qjl_gamma"],
        source=source,
    )


_ENCODE_SIMD_THREADS = 32


@lru_cache(maxsize=None)
def _encode_mse_simd_kernel(bits: int, dim: int):
    pdim = packed_dim(dim, bits)
    n_cent = 2**bits
    source = f"""
    constexpr int DIM = {dim};
    constexpr int BITS = {bits};
    constexpr int PDIM = {pdim};
    constexpr int N_CENT = {n_cent};
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;
    constexpr int BD = 32;
    constexpr int per_thread = DIM / BD;

    uint tg = threadgroup_position_in_grid.x;
    uint simd_lid = thread_index_in_simdgroup;

    threadgroup float unit_shmem[DIM];
    threadgroup uint indices_shmem[DIM];
    threadgroup float norm_shmem;

    float norm_sq = 0.0f;
    for (int t = 0; t < per_thread; t++) {{
        int j = simd_lid * per_thread + t;
        float v = vectors[tg * DIM + j];
        norm_sq += v * v;
    }}
    norm_sq = simd_sum(norm_sq);
    if (simd_lid == 0) {{
        norm_shmem = sqrt(norm_sq);
        norms[tg] = norm_shmem;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_norm = 1.0f / max(norm_shmem, 1e-8f);
    for (int t = 0; t < per_thread; t++) {{
        int j = simd_lid * per_thread + t;
        unit_shmem[j] = vectors[tg * DIM + j] * inv_norm;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < per_thread; t++) {{
        int i = simd_lid * per_thread + t;
        float rot = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            rot += unit_shmem[j] * rotation[i * DIM + j];
        }}
        uint best = 0u;
        float best_d = abs(rot - centroids[0]);
        for (int c = 1; c < N_CENT; c++) {{
            float d = abs(rot - centroids[c]);
            if (d < best_d) {{
                best_d = d;
                best = uint(c);
            }}
        }}
        indices_shmem[i] = best;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lid == 0) {{
        uint packed_off = tg * PDIM;
        for (int w = 0; w < PDIM; w++) {{
            packed[packed_off + w] = 0u;
        }}
        for (int i = 0; i < DIM; i++) {{
            int word = (i * BITS) / BITS_PER_WORD;
            int shift = (i * BITS) % BITS_PER_WORD;
            uint val = indices_shmem[i];
            packed[packed_off + word] |= (val & MASK) << shift;
            if (shift + BITS > BITS_PER_WORD) {{
                int spill = BITS - (BITS_PER_WORD - shift);
                packed[packed_off + word + 1] |= (val & MASK) >> (BITS_PER_WORD - shift);
            }}
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_encode_mse_simd_{bits}b_{dim}d",
        input_names=["vectors", "rotation", "centroids"],
        output_names=["packed", "norms"],
        source=source,
    )


@lru_cache(maxsize=None)
def _encode_prod_simd_kernel(bits: int, dim: int):
    mse_bits = bits - 1
    pdim = packed_dim(dim, mse_bits)
    qjl_pdim = qjl_packed_dim(dim)
    n_cent = 2**mse_bits
    source = f"""
    constexpr int DIM = {dim};
    constexpr int MSE_BITS = {mse_bits};
    constexpr int PDIM = {pdim};
    constexpr int QJL_PDIM = {qjl_pdim};
    constexpr int N_CENT = {n_cent};
    constexpr uint MSE_MASK = (1u << MSE_BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;
    constexpr int BD = 32;
    constexpr int per_thread = DIM / BD;

    uint tg = threadgroup_position_in_grid.x;
    uint simd_lid = thread_index_in_simdgroup;

    threadgroup float unit_shmem[DIM];
    threadgroup float mse_rot_shmem[DIM];
    threadgroup float residual_shmem[DIM];
    threadgroup uint mse_indices_shmem[DIM];
    threadgroup uint qjl_bits_shmem[DIM];
    threadgroup float norm_shmem;

    float norm_sq = 0.0f;
    for (int t = 0; t < per_thread; t++) {{
        int j = simd_lid * per_thread + t;
        float v = vectors[tg * DIM + j];
        norm_sq += v * v;
    }}
    norm_sq = simd_sum(norm_sq);
    if (simd_lid == 0) {{
        norm_shmem = sqrt(norm_sq);
        norms[tg] = norm_shmem;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float inv_norm = 1.0f / max(norm_shmem, 1e-8f);
    for (int t = 0; t < per_thread; t++) {{
        int j = simd_lid * per_thread + t;
        unit_shmem[j] = vectors[tg * DIM + j] * inv_norm;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < per_thread; t++) {{
        int i = simd_lid * per_thread + t;
        float rot = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            rot += unit_shmem[j] * rotation[i * DIM + j];
        }}
        uint best = 0u;
        float best_d = abs(rot - centroids[0]);
        for (int c = 1; c < N_CENT; c++) {{
            float d = abs(rot - centroids[c]);
            if (d < best_d) {{
                best_d = d;
                best = uint(c);
            }}
        }}
        mse_indices_shmem[i] = best;
        mse_rot_shmem[i] = centroids[best];
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float res_part = 0.0f;
    for (int t = 0; t < per_thread; t++) {{
        int j = simd_lid * per_thread + t;
        float mse_u = 0.0f;
        for (int i = 0; i < DIM; i++) {{
            mse_u += mse_rot_shmem[i] * rotation[i * DIM + j];
        }}
        float r = unit_shmem[j] - mse_u;
        residual_shmem[j] = r;
        res_part += r * r;
    }}
    float gamma_sq = simd_sum(res_part);
    if (simd_lid == 0) {{
        qjl_gamma[tg] = sqrt(gamma_sq);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < per_thread; t++) {{
        int i = simd_lid * per_thread + t;
        float proj = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            proj += residual_shmem[j] * s_matrix[i * DIM + j];
        }}
        qjl_bits_shmem[i] = proj >= 0.0f ? 1u : 0u;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lid == 0) {{
        uint packed_off = tg * PDIM;
        uint signs_off = tg * QJL_PDIM;
        for (int w = 0; w < PDIM; w++) {{
            mse_packed[packed_off + w] = 0u;
        }}
        for (int w = 0; w < QJL_PDIM; w++) {{
            qjl_signs[signs_off + w] = 0u;
        }}
        for (int i = 0; i < DIM; i++) {{
            int word = (i * MSE_BITS) / BITS_PER_WORD;
            int shift = (i * MSE_BITS) % BITS_PER_WORD;
            uint val = mse_indices_shmem[i];
            mse_packed[packed_off + word] |= (val & MSE_MASK) << shift;
            if (shift + MSE_BITS > BITS_PER_WORD) {{
                mse_packed[packed_off + word + 1] |= (val & MSE_MASK) >> (BITS_PER_WORD - shift);
            }}
            int s_word = i / BITS_PER_WORD;
            int s_shift = i % BITS_PER_WORD;
            qjl_signs[signs_off + s_word] |= qjl_bits_shmem[i] << s_shift;
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_encode_prod_simd_{bits}b_{dim}d",
        input_names=["vectors", "rotation", "centroids", "s_matrix"],
        output_names=["mse_packed", "norms", "qjl_signs", "qjl_gamma"],
        source=source,
    )


@lru_cache(maxsize=None)
def _encode_kv_fused_kernel(k_bits: int, v_bits: int, dim: int):
    """Fused K (prod) + V (mse) encode in one dispatch per token."""
    mse_bits = k_bits - 1
    k_pdim = packed_dim(dim, mse_bits)
    v_pdim = packed_dim(dim, v_bits)
    qjl_pdim = qjl_packed_dim(dim)
    k_n_cent = 2**mse_bits
    v_n_cent = 2**v_bits
    source = f"""
    constexpr int DIM = {dim};
    constexpr int MSE_BITS = {mse_bits};
    constexpr int V_BITS = {v_bits};
    constexpr int K_PDIM = {k_pdim};
    constexpr int V_PDIM = {v_pdim};
    constexpr int QJL_PDIM = {qjl_pdim};
    constexpr int K_N_CENT = {k_n_cent};
    constexpr int V_N_CENT = {v_n_cent};
    constexpr uint MSE_MASK = (1u << MSE_BITS) - 1u;
    constexpr uint V_MASK = (1u << V_BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;
    constexpr int BD = 32;
    constexpr int per_thread = DIM / BD;

    uint tg = threadgroup_position_in_grid.x;
    uint simd_lid = thread_index_in_simdgroup;

    threadgroup float k_unit_shmem[DIM];
    threadgroup float v_unit_shmem[DIM];
    threadgroup float k_mse_rot_shmem[DIM];
    threadgroup float k_residual_shmem[DIM];
    threadgroup uint k_mse_indices_shmem[DIM];
    threadgroup uint k_qjl_bits_shmem[DIM];
    threadgroup uint v_indices_shmem[DIM];
    threadgroup float k_norm_shmem;
    threadgroup float v_norm_shmem;

    float k_norm_sq = 0.0f;
    float v_norm_sq = 0.0f;
    for (int t = 0; t < per_thread; t++) {{
        int j = simd_lid * per_thread + t;
        float kv = keys[tg * DIM + j];
        float vv = values[tg * DIM + j];
        k_norm_sq += kv * kv;
        v_norm_sq += vv * vv;
    }}
    k_norm_sq = simd_sum(k_norm_sq);
    v_norm_sq = simd_sum(v_norm_sq);
    if (simd_lid == 0) {{
        k_norm_shmem = sqrt(k_norm_sq);
        v_norm_shmem = sqrt(v_norm_sq);
        k_norms[tg] = k_norm_shmem;
        v_norms[tg] = v_norm_shmem;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float k_inv = 1.0f / max(k_norm_shmem, 1e-8f);
    float v_inv = 1.0f / max(v_norm_shmem, 1e-8f);
    for (int t = 0; t < per_thread; t++) {{
        int j = simd_lid * per_thread + t;
        k_unit_shmem[j] = keys[tg * DIM + j] * k_inv;
        v_unit_shmem[j] = values[tg * DIM + j] * v_inv;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < per_thread; t++) {{
        int i = simd_lid * per_thread + t;
        float k_rot = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            k_rot += k_unit_shmem[j] * k_rotation[i * DIM + j];
        }}
        uint k_best = 0u;
        float k_best_d = abs(k_rot - k_centroids[0]);
        for (int c = 1; c < K_N_CENT; c++) {{
            float d = abs(k_rot - k_centroids[c]);
            if (d < k_best_d) {{
                k_best_d = d;
                k_best = uint(c);
            }}
        }}
        k_mse_indices_shmem[i] = k_best;
        k_mse_rot_shmem[i] = k_centroids[k_best];

        float v_rot = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            v_rot += v_unit_shmem[j] * v_rotation[i * DIM + j];
        }}
        uint v_best = 0u;
        float v_best_d = abs(v_rot - v_centroids[0]);
        for (int c = 1; c < V_N_CENT; c++) {{
            float d = abs(v_rot - v_centroids[c]);
            if (d < v_best_d) {{
                v_best_d = d;
                v_best = uint(c);
            }}
        }}
        v_indices_shmem[i] = v_best;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float res_part = 0.0f;
    for (int t = 0; t < per_thread; t++) {{
        int j = simd_lid * per_thread + t;
        float mse_u = 0.0f;
        for (int i = 0; i < DIM; i++) {{
            mse_u += k_mse_rot_shmem[i] * k_rotation[i * DIM + j];
        }}
        float r = k_unit_shmem[j] - mse_u;
        k_residual_shmem[j] = r;
        res_part += r * r;
    }}
    float gamma_sq = simd_sum(res_part);
    if (simd_lid == 0) {{
        k_gamma[tg] = sqrt(gamma_sq);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < per_thread; t++) {{
        int i = simd_lid * per_thread + t;
        float proj = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            proj += k_residual_shmem[j] * k_qjl[i * DIM + j];
        }}
        k_qjl_bits_shmem[i] = proj >= 0.0f ? 1u : 0u;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_lid == 0) {{
        uint k_off = tg * K_PDIM;
        uint v_off = tg * V_PDIM;
        uint signs_off = tg * QJL_PDIM;
        for (int w = 0; w < K_PDIM; w++) {{
            k_packed[k_off + w] = 0u;
        }}
        for (int w = 0; w < V_PDIM; w++) {{
            v_packed[v_off + w] = 0u;
        }}
        for (int w = 0; w < QJL_PDIM; w++) {{
            k_qjl_signs[signs_off + w] = 0u;
        }}
        for (int i = 0; i < DIM; i++) {{
            int k_word = (i * MSE_BITS) / BITS_PER_WORD;
            int k_shift = (i * MSE_BITS) % BITS_PER_WORD;
            uint k_val = k_mse_indices_shmem[i];
            k_packed[k_off + k_word] |= (k_val & MSE_MASK) << k_shift;
            if (k_shift + MSE_BITS > BITS_PER_WORD) {{
                k_packed[k_off + k_word + 1] |= (k_val & MSE_MASK) >> (BITS_PER_WORD - k_shift);
            }}
            int v_word = (i * V_BITS) / BITS_PER_WORD;
            int v_shift = (i * V_BITS) % BITS_PER_WORD;
            uint v_val = v_indices_shmem[i];
            v_packed[v_off + v_word] |= (v_val & V_MASK) << v_shift;
            if (v_shift + V_BITS > BITS_PER_WORD) {{
                v_packed[v_off + v_word + 1] |= (v_val & V_MASK) >> (BITS_PER_WORD - v_shift);
            }}
            int s_word = i / BITS_PER_WORD;
            int s_shift = i % BITS_PER_WORD;
            k_qjl_signs[signs_off + s_word] |= k_qjl_bits_shmem[i] << s_shift;
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_encode_kv_fused_{k_bits}k_{v_bits}v_{dim}d",
        input_names=[
            "keys",
            "values",
            "k_rotation",
            "v_rotation",
            "k_centroids",
            "v_centroids",
            "k_qjl",
        ],
        output_names=[
            "k_packed",
            "k_norms",
            "k_qjl_signs",
            "k_gamma",
            "v_packed",
            "v_norms",
        ],
        source=source,
    )


def _launch_encode_mse(
    vectors: mx.array,
    rotation: mx.array,
    bits: int,
    dim: int,
) -> tuple[mx.array, mx.array]:
    shape = vectors.shape[:-1]
    n = int(math.prod(shape)) if shape else vectors.shape[0]
    pdim = packed_dim(dim, bits)
    flat = vectors.astype(mx.float32).reshape(n, dim)
    centroids, _ = get_codebook(bits)
    kernel = _encode_mse_simd_kernel(bits, dim)
    packed, norms = kernel(
        inputs=[flat, rotation, centroids],
        template=[("T", mx.float32)],
        grid=(n * _ENCODE_SIMD_THREADS, 1, 1),
        threadgroup=(_ENCODE_SIMD_THREADS, 1, 1),
        output_shapes=[(n, pdim), (n, 1)],
        output_dtypes=[mx.uint32, mx.float32],
        stream=mx.gpu,
    )
    return packed.reshape(shape + (pdim,)), norms.reshape(shape + (1,))


def _launch_encode_prod(
    vectors: mx.array,
    rotation: mx.array,
    s_matrix: mx.array,
    bits: int,
    dim: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    mse_bits = bits - 1
    shape = vectors.shape[:-1]
    n = int(math.prod(shape)) if shape else vectors.shape[0]
    pdim = packed_dim(dim, mse_bits)
    qpdim = qjl_packed_dim(dim)
    flat = vectors.astype(mx.float32).reshape(n, dim)
    centroids, _ = get_codebook(mse_bits)
    kernel = _encode_prod_simd_kernel(bits, dim)
    mse_packed, norms, qjl_signs, qjl_gamma = kernel(
        inputs=[flat, rotation, centroids, s_matrix],
        template=[("T", mx.float32)],
        grid=(n * _ENCODE_SIMD_THREADS, 1, 1),
        threadgroup=(_ENCODE_SIMD_THREADS, 1, 1),
        output_shapes=[(n, pdim), (n, 1), (n, qpdim), (n, 1)],
        output_dtypes=[mx.uint32, mx.float32, mx.uint32, mx.float32],
        stream=mx.gpu,
    )
    return (
        mse_packed.reshape(shape + (pdim,)),
        norms.reshape(shape + (1,)),
        qjl_signs.reshape(shape + (qpdim,)),
        qjl_gamma.reshape(shape + (1,)),
    )


def _launch_encode_kv(
    keys: mx.array,
    values: mx.array,
    k_rotation: mx.array,
    v_rotation: mx.array,
    k_qjl: mx.array,
    k_bits: int,
    v_bits: int,
    dim: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    shape = keys.shape[:-1]
    n = int(math.prod(shape)) if shape else keys.shape[0]
    mse_bits = k_bits - 1
    k_pdim = packed_dim(dim, mse_bits)
    v_pdim = packed_dim(dim, v_bits)
    qpdim = qjl_packed_dim(dim)
    flat_k = keys.astype(mx.float32).reshape(n, dim)
    flat_v = values.astype(mx.float32).reshape(n, dim)
    k_centroids, _ = get_codebook(mse_bits)
    v_centroids, _ = get_codebook(v_bits)
    kernel = _encode_kv_fused_kernel(k_bits, v_bits, dim)
    k_packed, k_norms, k_signs, k_gamma, v_packed, v_norms = kernel(
        inputs=[
            flat_k,
            flat_v,
            k_rotation,
            v_rotation,
            k_centroids,
            v_centroids,
            k_qjl,
        ],
        template=[("T", mx.float32)],
        grid=(n * _ENCODE_SIMD_THREADS, 1, 1),
        threadgroup=(_ENCODE_SIMD_THREADS, 1, 1),
        output_shapes=[
            (n, k_pdim),
            (n, 1),
            (n, qpdim),
            (n, 1),
            (n, v_pdim),
            (n, 1),
        ],
        output_dtypes=[
            mx.uint32,
            mx.float32,
            mx.uint32,
            mx.float32,
            mx.uint32,
            mx.float32,
        ],
        stream=mx.gpu,
    )
    return (
        k_packed.reshape(shape + (k_pdim,)),
        k_norms.reshape(shape + (1,)),
        k_signs.reshape(shape + (qpdim,)),
        k_gamma.reshape(shape + (1,)),
        v_packed.reshape(shape + (v_pdim,)),
        v_norms.reshape(shape + (1,)),
    )


def encode_mse_metal(
    vectors: mx.array,
    rotation: mx.array,
    bits: int,
    dim: int,
) -> tuple[mx.array, mx.array]:
    if not metal_available():
        raise RuntimeError("Metal is not available")
    return _launch_encode_mse(vectors, rotation, bits, dim)


def encode_prod_metal(
    vectors: mx.array,
    rotation: mx.array,
    s_matrix: mx.array,
    bits: int,
    dim: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    if not metal_available():
        raise RuntimeError("Metal is not available")
    return _launch_encode_prod(vectors, rotation, s_matrix, bits, dim)


def encode_kv_metal(
    keys: mx.array,
    values: mx.array,
    k_rotation: mx.array,
    v_rotation: mx.array,
    k_qjl: mx.array,
    k_bits: int,
    v_bits: int,
    dim: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array, mx.array]:
    """Fused K+V encode in one Metal dispatch."""
    if not metal_available():
        raise RuntimeError("Metal is not available")
    return _launch_encode_kv(
        keys, values, k_rotation, v_rotation, k_qjl, k_bits, v_bits, dim
    )


@lru_cache(maxsize=None)
def _qk_scores_kernel(k_bits: int, dim: int):
    """Fused Q·K for TurboQuant_prod keys.

    Math (row-vector convention, ``R`` from ``make_rotation_matrix``):
      unit_mse = c @ R,  c_i = Lloyd-Max centroid of rotated coord i
      q·unit_mse = (q @ R.T)·c = sum_i c_i * (q @ R.T)_i

      qjl = (sqrt(pi/2)/d) * gamma * (signs @ S),  signs in {{±1}}
      q·qjl = (sqrt(pi/2)/d) * gamma * sum_i sign_i * (q @ S.T)_i

      score = norm * (q·unit_mse + q·qjl) * attn_scale
    """
    mse_bits = k_bits - 1
    pdim = packed_dim(dim, mse_bits)
    qjl_pdim = qjl_packed_dim(dim)
    n_cent = 2**mse_bits
    qjl_scale = _QJL_SCALE / dim
    source = f"""
    constexpr int DIM = {dim};
    constexpr int MSE_BITS = {mse_bits};
    constexpr int PDIM = {pdim};
    constexpr int QJL_PDIM = {qjl_pdim};
    constexpr int N_CENT = {n_cent};
    constexpr uint MSE_MASK = (1u << MSE_BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;
    constexpr float QJL_SCALE = {qjl_scale}f;

    uint gid = thread_position_in_grid.x;

    uint tmp = gid;
    uint s = tmp % seq_len;
    tmp /= seq_len;
    uint l = tmp % query_len;
    tmp /= query_len;
    uint h_q = tmp % n_q_heads;
    uint b = tmp / n_q_heads;
    uint h_kv = h_q / n_repeats;

    uint q_base = ((b * n_q_heads + h_q) * query_len + l) * DIM;
    uint kv_slot = (b * n_kv_heads + h_kv) * seq_len + s;
    uint k_packed_off = kv_slot * PDIM;
    uint k_signs_off = kv_slot * QJL_PDIM;

    float kn = k_norms[kv_slot];
    float gamma = k_gamma[kv_slot];

    float qv[DIM];
    for (int j = 0; j < DIM; j++) {{
        qv[j] = static_cast<float>(queries[q_base + j]);
    }}

    float q_rot[DIM];
    float q_s[DIM];
    for (int i = 0; i < DIM; i++) {{
        float rot = 0.0f;
        float ps = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            rot += qv[j] * rotation[i * DIM + j];
            ps += qv[j] * s_matrix[i * DIM + j];
        }}
        q_rot[i] = rot;
        q_s[i] = ps;
    }}

    float mse_dot = 0.0f;
    for (int i = 0; i < DIM; i++) {{
        int word = (i * MSE_BITS) / BITS_PER_WORD;
        int shift = (i * MSE_BITS) % BITS_PER_WORD;
        uint w = mse_packed[k_packed_off + word];
        uint val = (w >> shift) & MSE_MASK;
        if (shift + MSE_BITS > BITS_PER_WORD) {{
            int spill = MSE_BITS - (BITS_PER_WORD - shift);
            uint w2 = mse_packed[k_packed_off + word + 1];
            val = val | ((w2 & ((1u << spill) - 1u)) << (BITS_PER_WORD - shift));
        }}
        mse_dot += centroids[val] * q_rot[i];
    }}

    float qjl_dot = 0.0f;
    for (int i = 0; i < DIM; i++) {{
        int s_word = i / BITS_PER_WORD;
        int s_shift = i % BITS_PER_WORD;
        uint bit = (qjl_signs[k_signs_off + s_word] >> s_shift) & 1u;
        float sign = bit ? 1.0f : -1.0f;
        qjl_dot += sign * q_s[i];
    }}

    scores[gid] = static_cast<T>(kn * (mse_dot + QJL_SCALE * gamma * qjl_dot));
    """
    return mx.fast.metal_kernel(
        name=f"tq_qk_scores_{k_bits}b_{dim}d",
        input_names=[
            "queries",
            "mse_packed",
            "k_norms",
            "qjl_signs",
            "k_gamma",
            "rotation",
            "centroids",
            "s_matrix",
        ],
        output_names=["scores"],
        source=source,
    )


def qk_scores_metal(
    queries: mx.array,
    mse_packed: mx.array,
    k_norms: mx.array,
    qjl_signs: mx.array,
    k_gamma: mx.array,
    rotation: mx.array,
    s_matrix: mx.array,
    k_bits: int,
    dim: int,
    scale: float,
) -> mx.array:
    if not metal_available():
        raise RuntimeError("Metal is not available")
    B, n_q_heads, L, _ = queries.shape
    S = mse_packed.shape[2]
    n_kv_heads = mse_packed.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    mse_bits = k_bits - 1
    pdim = packed_dim(dim, mse_bits)
    qpdim = qjl_packed_dim(dim)
    centroids, _ = get_codebook(mse_bits)
    # Row-contiguous flatten (bitlinear / QuantizedKVCache pattern).
    flat_q = mx.contiguous(queries.reshape(-1, dim))
    flat_packed = mx.contiguous(mse_packed.reshape(-1, pdim))
    flat_signs = mx.contiguous(qjl_signs.reshape(-1, qpdim))
    flat_norms = mx.contiguous(k_norms.reshape(-1).astype(mx.float32))
    flat_gamma = mx.contiguous(k_gamma.reshape(-1).astype(mx.float32))
    n_scores = B * n_q_heads * L * S
    kernel = _qk_scores_kernel(k_bits, dim)
    scores = kernel(
        inputs=[
            flat_q,
            flat_packed,
            flat_norms,
            flat_signs,
            flat_gamma,
            rotation,
            centroids,
            s_matrix,
        ],
        template=[
            ("T", mx.float32),
            ("n_q_heads", n_q_heads),
            ("n_kv_heads", n_kv_heads),
            ("n_repeats", n_repeats),
            ("query_len", L),
            ("seq_len", S),
        ],
        grid=(n_scores, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(n_scores,)],
        output_dtypes=[mx.float32],
        stream=mx.gpu,
    )[0]
    return (scores * scale).reshape(B, n_q_heads, L, S).astype(queries.dtype)


@lru_cache(maxsize=None)
def _av_weighted_sum_kernel(v_bits: int, dim: int):
    """Fused attn @ V for TurboQuant_mse values without materializing V.

    Per output coord (b, h_q, l, d):
      out = sum_s attn[b,h_q,l,s] * norm[s] * sum_i c_i * R[i,d]
    where c_i is the Lloyd-Max centroid from packed indices at coord i.
    """
    pdim = packed_dim(dim, v_bits)
    n_cent = 2**v_bits
    source = f"""
    constexpr int DIM = {dim};
    constexpr int BITS = {v_bits};
    constexpr int PDIM = {pdim};
    constexpr int N_CENT = {n_cent};
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;

    uint gid = thread_position_in_grid.x;

    uint tmp = gid;
    uint d = tmp % DIM;
    tmp /= DIM;
    uint l = tmp % query_len;
    tmp /= query_len;
    uint h_q = tmp % n_q_heads;
    uint b = tmp / n_q_heads;
    uint h_kv = h_q / n_repeats;

    uint attn_base = ((b * n_q_heads + h_q) * query_len + l) * seq_len;
    uint kv_base = (b * n_kv_heads + h_kv) * seq_len;

    float acc = 0.0f;
    for (uint s = 0; s < seq_len; s++) {{
        float w = static_cast<float>(attn[attn_base + s]);
        uint kv_slot = kv_base + s;
        float vn = v_norms[kv_slot];
        uint packed_off = kv_slot * PDIM;

        float vd = 0.0f;
        for (int i = 0; i < DIM; i++) {{
            int word = (i * BITS) / BITS_PER_WORD;
            int shift = (i * BITS) % BITS_PER_WORD;
            uint w_packed = v_packed[packed_off + word];
            uint val = (w_packed >> shift) & MASK;
            if (shift + BITS > BITS_PER_WORD) {{
                int spill = BITS - (BITS_PER_WORD - shift);
                uint w2 = v_packed[packed_off + word + 1];
                val = val | ((w2 & ((1u << spill) - 1u)) << (BITS_PER_WORD - shift));
            }}
            vd += centroids[val] * rotation[i * DIM + d];
        }}
        acc += w * vn * vd;
    }}

    out[gid] = static_cast<T>(acc);
    """
    return mx.fast.metal_kernel(
        name=f"tq_av_weighted_sum_{v_bits}b_{dim}d",
        input_names=["attn", "v_packed", "v_norms", "rotation", "centroids"],
        output_names=["out"],
        source=source,
    )


def av_weighted_sum_metal(
    attn: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    rotation: mx.array,
    v_bits: int,
    dim: int,
    dtype: Optional[mx.dtype] = None,
) -> mx.array:
    """Weighted sum of dequantized TurboQuant_mse values without full V decode."""
    if not metal_available():
        raise RuntimeError("Metal is not available")
    if dtype is None:
        dtype = attn.dtype
    B, n_q_heads, L, S = attn.shape
    n_kv_heads = v_packed.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    pdim = packed_dim(dim, v_bits)
    centroids, _ = get_codebook(v_bits)
    flat_attn = mx.contiguous(attn.reshape(-1, S))
    flat_packed = mx.contiguous(v_packed.reshape(-1, pdim))
    flat_norms = mx.contiguous(v_norms.reshape(-1).astype(mx.float32))
    n_out = B * n_q_heads * L * dim
    kernel = _av_weighted_sum_kernel(v_bits, dim)
    out = kernel(
        inputs=[flat_attn, flat_packed, flat_norms, rotation, centroids],
        template=[
            ("T", dtype),
            ("n_q_heads", n_q_heads),
            ("n_kv_heads", n_kv_heads),
            ("n_repeats", n_repeats),
            ("query_len", L),
            ("seq_len", S),
        ],
        grid=(n_out, 1, 1),
        threadgroup=(1, 1, 1),
        output_shapes=[(n_out,)],
        output_dtypes=[dtype],
        stream=mx.gpu,
    )[0]
    return out.reshape(B, n_q_heads, L, dim)

def _metal_unpack_inline(bits: str, mask: str, packed: str, offset: str, idx: str) -> str:
    return f"""
            int word = ({idx} * {bits}) / BITS_PER_WORD;
            int shift = ({idx} * {bits}) % BITS_PER_WORD;
            uint w = {packed}[{offset} + word];
            uint val = (w >> shift) & {mask};
            if (shift + {bits} > BITS_PER_WORD) {{
                int spill = {bits} - (BITS_PER_WORD - shift);
                uint w2 = {packed}[{offset} + word + 1];
                val = val | ((w2 & ((1u << spill) - 1u)) << (BITS_PER_WORD - shift));
            }}
    """


@lru_cache(maxsize=None)
def _tq_sdpa_vector_kernel(k_bits: int, v_bits: int, dim: int, scale: float):
    """Fused vector SDPA for packed TurboQuant KV (L <= 8)."""
    mse_bits = k_bits - 1
    k_pdim = packed_dim(dim, mse_bits)
    v_pdim = packed_dim(dim, v_bits)
    qjl_pdim = qjl_packed_dim(dim)
    qjl_scale = _QJL_SCALE / dim
    unpack_k_inline = _metal_unpack_inline("MSE_BITS", "MSE_MASK", "k_packed", "k_packed_off", "i")
    unpack_v_inline = _metal_unpack_inline("V_BITS", "V_MASK", "v_packed", "v_packed_off", "i")
    source = f"""
    constexpr int DIM = {dim};
    constexpr int MSE_BITS = {mse_bits};
    constexpr int V_BITS = {v_bits};
    constexpr int K_PDIM = {k_pdim};
    constexpr int V_PDIM = {v_pdim};
    constexpr int QJL_PDIM = {qjl_pdim};
    constexpr uint MSE_MASK = (1u << MSE_BITS) - 1u;
    constexpr uint V_MASK = (1u << V_BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = DIM / BD;
    constexpr int v_per_thread = DIM / BD;
    constexpr float QJL_SCALE = {qjl_scale}f;
    constexpr float ATTN_SCALE = {scale}f;

    uint tg = threadgroup_position_in_grid.x;
    uint q_batch_head_idx = tg / query_len;
    uint q_seq_idx = tg % query_len;
    uint simd_lid = thread_index_in_simdgroup;

    uint b = q_batch_head_idx / n_q_heads;
    uint h_q = q_batch_head_idx % n_q_heads;
    uint h_kv = h_q / n_repeats;
    uint kv_base = (b * n_kv_heads + h_kv) * seq_len;
    uint q_offset = q_batch_head_idx * query_len + q_seq_idx;

    threadgroup float q_shmem[DIM];
    threadgroup float q_rot_shmem[DIM];
    threadgroup float q_s_shmem[DIM];
    threadgroup float tg_max_score;
    threadgroup float tg_sum_exp;
    threadgroup float tg_factor;
    threadgroup float tg_exp_score;

    thread float o[v_per_thread];
    for (int i = 0; i < v_per_thread; i++) {{
        o[i] = 0.0f;
    }}
    if (simd_lid == 0) {{
        tg_max_score = -1e30f;
        tg_sum_exp = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint q_base = q_offset * DIM;
    for (int t = 0; t < qk_per_thread; t++) {{
        int idx = simd_lid * qk_per_thread + t;
        q_shmem[idx] = static_cast<float>(queries[q_base + idx]) * ATTN_SCALE;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = simd_lid; idx < DIM; idx += BD) {{
        float rot = 0.0f;
        float ps = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            rot += q_shmem[j] * k_rotation[idx * DIM + j];
            ps += q_shmem[j] * k_qjl[idx * DIM + j];
        }}
        q_rot_shmem[idx] = rot;
        q_s_shmem[idx] = ps;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ki = 0; ki < seq_len; ki++) {{
        if (do_causal && ki > int(seq_len - query_len + q_seq_idx)) {{
            continue;
        }}

        uint kv_slot = kv_base + ki;
        uint k_packed_off = kv_slot * K_PDIM;
        uint k_signs_off = kv_slot * QJL_PDIM;
        float kn = k_norms[kv_slot];
        float gamma = k_gamma[kv_slot];

        float mse_part = 0.0f;
        float qjl_part = 0.0f;
        for (int t = 0; t < qk_per_thread; t++) {{
            int i = simd_lid * qk_per_thread + t;
            {unpack_k_inline}
            mse_part += centroids_k[val] * q_rot_shmem[i];
            int s_word = i / BITS_PER_WORD;
            int s_shift = i % BITS_PER_WORD;
            uint bit = (qjl_signs[k_signs_off + s_word] >> s_shift) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            qjl_part += sign * q_s_shmem[i];
        }}
        float mse_dot = simd_sum(mse_part);
        float qjl_dot = simd_sum(qjl_part);
        float score = kn * (mse_dot + QJL_SCALE * gamma * qjl_dot);

        if (simd_lid == 0) {{
            float new_max = max(tg_max_score, score);
            tg_factor = metal::fast::exp(tg_max_score - new_max);
            tg_exp_score = metal::fast::exp(score - new_max);
            tg_max_score = new_max;
            tg_sum_exp = tg_sum_exp * tg_factor + tg_exp_score;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint v_packed_off = kv_slot * V_PDIM;
        float vn = v_norms[kv_slot];
        for (int t = 0; t < v_per_thread; t++) {{
            int d = simd_lid * v_per_thread + t;
            float vd = 0.0f;
            for (int i = 0; i < DIM; i++) {{
                {unpack_v_inline}
                vd += centroids_v[val] * v_rotation[i * DIM + d];
            }}
            o[t] = o[t] * tg_factor + tg_exp_score * vn * vd;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    float inv_sum = tg_sum_exp == 0.0f ? 1.0f : (1.0f / tg_sum_exp);
    uint out_base = q_offset * DIM + simd_lid * v_per_thread;
    for (int t = 0; t < v_per_thread; t++) {{
        out[out_base + t] = static_cast<T>(o[t] * inv_sum);
    }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_sdpa_vector_v3_{k_bits}k_{v_bits}v_{dim}d",
        input_names=[
            "queries",
            "k_packed",
            "k_norms",
            "qjl_signs",
            "k_gamma",
            "v_packed",
            "v_norms",
            "k_rotation",
            "v_rotation",
            "centroids_k",
            "centroids_v",
            "k_qjl",
        ],
        output_names=["out"],
        source=source,
    )


def tq_sdpa_vector_metal(
    queries: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    qjl_signs: mx.array,
    k_gamma: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    k_rotation: mx.array,
    v_rotation: mx.array,
    k_qjl: mx.array,
    k_bits: int,
    v_bits: int,
    dim: int,
    scale: float,
    do_causal: bool,
) -> mx.array:
    """Fused vector SDPA from packed TurboQuant KV (no dense K/V/scores)."""
    if not metal_available():
        raise RuntimeError("Metal is not available")
    B, n_q_heads, L, _ = queries.shape
    S = k_packed.shape[2]
    n_kv_heads = k_packed.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    mse_bits = k_bits - 1
    k_pdim = packed_dim(dim, mse_bits)
    v_pdim = packed_dim(dim, v_bits)
    centroids_k, _ = get_codebook(mse_bits)
    centroids_v, _ = get_codebook(v_bits)
    flat_q = mx.contiguous(queries.reshape(-1, dim))
    flat_k = mx.contiguous(k_packed.reshape(-1, k_pdim))
    flat_signs = mx.contiguous(qjl_signs.reshape(-1, qjl_packed_dim(dim)))
    flat_v = mx.contiguous(v_packed.reshape(-1, v_pdim))
    flat_kn = mx.contiguous(k_norms.reshape(-1).astype(mx.float32))
    flat_kg = mx.contiguous(k_gamma.reshape(-1).astype(mx.float32))
    flat_vn = mx.contiguous(v_norms.reshape(-1).astype(mx.float32))
    kernel = _tq_sdpa_vector_kernel(k_bits, v_bits, dim, scale)
    out = kernel(
        inputs=[
            flat_q,
            flat_k,
            flat_kn,
            flat_signs,
            flat_kg,
            flat_v,
            flat_vn,
            k_rotation,
            v_rotation,
            centroids_k,
            centroids_v,
            k_qjl,
        ],
        template=[
            ("T", queries.dtype),
            ("n_q_heads", n_q_heads),
            ("n_kv_heads", n_kv_heads),
            ("n_repeats", n_repeats),
            ("query_len", L),
            ("seq_len", S),
            ("do_causal", do_causal),
        ],
        grid=(B * n_q_heads * L * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(B * n_q_heads * L, dim)],
        output_dtypes=[queries.dtype],
        stream=mx.gpu,
    )[0]
    return out.reshape(B, n_q_heads, L, dim)


@lru_cache(maxsize=None)
def _tq_sdpa_tiled_kernel(k_bits: int, v_bits: int, dim: int, scale: float):
    """Fused tiled SDPA: 32 simdgroups stride over KV (L > 8 prefill)."""
    mse_bits = k_bits - 1
    k_pdim = packed_dim(dim, mse_bits)
    v_pdim = packed_dim(dim, v_bits)
    qjl_pdim = qjl_packed_dim(dim)
    qjl_scale = _QJL_SCALE / dim
    unpack_k_inline = _metal_unpack_inline("MSE_BITS", "MSE_MASK", "k_packed", "k_packed_off", "i")
    unpack_v_inline = _metal_unpack_inline("V_BITS", "V_MASK", "v_packed", "v_packed_off", "i")
    source = f"""
    constexpr int DIM = {dim};
    constexpr int MSE_BITS = {mse_bits};
    constexpr int V_BITS = {v_bits};
    constexpr int K_PDIM = {k_pdim};
    constexpr int V_PDIM = {v_pdim};
    constexpr int QJL_PDIM = {qjl_pdim};
    constexpr uint MSE_MASK = (1u << MSE_BITS) - 1u;
    constexpr uint V_MASK = (1u << V_BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = DIM / BD;
    constexpr int v_per_thread = DIM / BD;
    constexpr float QJL_SCALE = {qjl_scale}f;
    constexpr float ATTN_SCALE = {scale}f;

    uint tg = threadgroup_position_in_grid.x;
    uint q_batch_head_idx = tg / query_len;
    uint q_seq_idx = tg % query_len;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    uint b = q_batch_head_idx / n_q_heads;
    uint h_q = q_batch_head_idx % n_q_heads;
    uint h_kv = h_q / n_repeats;
    uint kv_base = (b * n_kv_heads + h_kv) * seq_len;
    uint q_offset = q_batch_head_idx * query_len + q_seq_idx;

    threadgroup float q_shmem[DIM];
    threadgroup float q_rot_shmem[DIM];
    threadgroup float q_s_shmem[DIM];
    threadgroup float max_scores[BN];
    threadgroup float sum_exp_scores[BN];
    threadgroup float outputs[BN * DIM];

    thread float o[v_per_thread];
    for (int i = 0; i < v_per_thread; i++) {{
        o[i] = 0.0f;
    }}

    uint q_base = q_offset * DIM;
    for (int t = 0; t < qk_per_thread; t++) {{
        int idx = simd_lid * qk_per_thread + t;
        q_shmem[idx] = static_cast<float>(queries[q_base + idx]) * ATTN_SCALE;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = simd_lid; idx < DIM; idx += BD) {{
        float rot = 0.0f;
        float ps = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            rot += q_shmem[j] * k_rotation[idx * DIM + j];
            ps += q_shmem[j] * k_qjl[idx * DIM + j];
        }}
        q_rot_shmem[idx] = rot;
        q_s_shmem[idx] = ps;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float max_score = -1e30f;
    float sum_exp_score = 0.0f;

    for (int ki = simd_gid; ki < seq_len; ki += BN) {{
        if (do_causal && ki > int(seq_len - query_len + q_seq_idx)) {{
            continue;
        }}

        uint kv_slot = kv_base + ki;
        uint k_packed_off = kv_slot * K_PDIM;
        uint k_signs_off = kv_slot * QJL_PDIM;
        float kn = k_norms[kv_slot];
        float gamma = k_gamma[kv_slot];

        float mse_part = 0.0f;
        float qjl_part = 0.0f;
        for (int t = 0; t < qk_per_thread; t++) {{
            int i = simd_lid * qk_per_thread + t;
            {unpack_k_inline}
            mse_part += centroids_k[val] * q_rot_shmem[i];
            int s_word = i / BITS_PER_WORD;
            int s_shift = i % BITS_PER_WORD;
            uint bit = (qjl_signs[k_signs_off + s_word] >> s_shift) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            qjl_part += sign * q_s_shmem[i];
        }}
        float mse_dot = simd_sum(mse_part);
        float qjl_dot = simd_sum(qjl_part);
        float score = kn * (mse_dot + QJL_SCALE * gamma * qjl_dot);

        float new_max = max(max_score, score);
        float factor = metal::fast::exp(max_score - new_max);
        float exp_score = metal::fast::exp(score - new_max);
        max_score = new_max;
        sum_exp_score = sum_exp_score * factor + exp_score;

        uint v_packed_off = kv_slot * V_PDIM;
        float vn = v_norms[kv_slot];
        for (int t = 0; t < v_per_thread; t++) {{
            int d = simd_lid * v_per_thread + t;
            float vd = 0.0f;
            for (int i = 0; i < DIM; i++) {{
                {unpack_v_inline}
                vd += centroids_v[val] * v_rotation[i * DIM + d];
            }}
            o[t] = o[t] * factor + exp_score * vn * vd;
        }}
    }}

    if (simd_lid == 0) {{
        max_scores[simd_gid] = max_score;
        sum_exp_scores[simd_gid] = sum_exp_score;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    max_score = max_scores[simd_lid];
    float global_max = simd_max(max_score);
    float global_factor = metal::fast::exp(max_score - global_max);
    sum_exp_score = simd_sum(sum_exp_scores[simd_lid] * global_factor);

    for (int t = 0; t < v_per_thread; t++) {{
        outputs[simd_lid * DIM + simd_gid] = o[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[t] = simd_sum(outputs[simd_gid * DIM + simd_lid] * global_factor);
        o[t] = sum_exp_score == 0.0f ? o[t] : (o[t] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (simd_lid == 0) {{
        uint out_base = q_offset * DIM + simd_gid * v_per_thread;
        for (int t = 0; t < v_per_thread; t++) {{
            out[out_base + t] = static_cast<T>(o[t]);
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_sdpa_tiled_{k_bits}k_{v_bits}v_{dim}d",
        input_names=[
            "queries",
            "k_packed",
            "k_norms",
            "qjl_signs",
            "k_gamma",
            "v_packed",
            "v_norms",
            "k_rotation",
            "v_rotation",
            "centroids_k",
            "centroids_v",
            "k_qjl",
        ],
        output_names=["out"],
        source=source,
    )


_TQ_TILE_THREADS = 1024


def tq_sdpa_tiled_metal(
    queries: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    qjl_signs: mx.array,
    k_gamma: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    k_rotation: mx.array,
    v_rotation: mx.array,
    k_qjl: mx.array,
    k_bits: int,
    v_bits: int,
    dim: int,
    scale: float,
    do_causal: bool,
) -> mx.array:
    """Fused tiled SDPA from packed TurboQuant KV (simdgroup-parallel over S)."""
    if not metal_available():
        raise RuntimeError("Metal is not available")
    B, n_q_heads, L, _ = queries.shape
    S = k_packed.shape[2]
    n_kv_heads = k_packed.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    mse_bits = k_bits - 1
    k_pdim = packed_dim(dim, mse_bits)
    v_pdim = packed_dim(dim, v_bits)
    centroids_k, _ = get_codebook(mse_bits)
    centroids_v, _ = get_codebook(v_bits)
    flat_q = mx.contiguous(queries.reshape(-1, dim))
    flat_k = mx.contiguous(k_packed.reshape(-1, k_pdim))
    flat_signs = mx.contiguous(qjl_signs.reshape(-1, qjl_packed_dim(dim)))
    flat_v = mx.contiguous(v_packed.reshape(-1, v_pdim))
    flat_kn = mx.contiguous(k_norms.reshape(-1).astype(mx.float32))
    flat_kg = mx.contiguous(k_gamma.reshape(-1).astype(mx.float32))
    flat_vn = mx.contiguous(v_norms.reshape(-1).astype(mx.float32))
    n_tg = B * n_q_heads * L
    kernel = _tq_sdpa_tiled_kernel(k_bits, v_bits, dim, scale)
    out = kernel(
        inputs=[
            flat_q,
            flat_k,
            flat_kn,
            flat_signs,
            flat_kg,
            flat_v,
            flat_vn,
            k_rotation,
            v_rotation,
            centroids_k,
            centroids_v,
            k_qjl,
        ],
        template=[
            ("T", queries.dtype),
            ("n_q_heads", n_q_heads),
            ("n_kv_heads", n_kv_heads),
            ("n_repeats", n_repeats),
            ("query_len", L),
            ("seq_len", S),
            ("do_causal", do_causal),
        ],
        grid=(n_tg * _TQ_TILE_THREADS, 1, 1),
        threadgroup=(_TQ_TILE_THREADS, 1, 1),
        output_shapes=[(B * n_q_heads * L, dim)],
        output_dtypes=[queries.dtype],
        stream=mx.gpu,
    )[0]
    return out.reshape(B, n_q_heads, L, dim)

TQ_2PASS_THRESHOLD = 1024
TQ_2PASS_BLOCKS = 64


@lru_cache(maxsize=None)
def _tq_sdpa_2pass1_kernel(k_bits: int, v_bits: int, dim: int, scale: float):
    """Pass 1: per-block partial softmax stats and unnormalized output."""
    mse_bits = k_bits - 1
    k_pdim = packed_dim(dim, mse_bits)
    v_pdim = packed_dim(dim, v_bits)
    qjl_pdim = qjl_packed_dim(dim)
    qjl_scale = _QJL_SCALE / dim
    unpack_k_inline = _metal_unpack_inline("MSE_BITS", "MSE_MASK", "k_packed", "k_packed_off", "i")
    unpack_v_inline = _metal_unpack_inline("V_BITS", "V_MASK", "v_packed", "v_packed_off", "i")
    source = f"""
    constexpr int DIM = {dim};
    constexpr int MSE_BITS = {mse_bits};
    constexpr int V_BITS = {v_bits};
    constexpr int K_PDIM = {k_pdim};
    constexpr int V_PDIM = {v_pdim};
    constexpr int QJL_PDIM = {qjl_pdim};
    constexpr uint MSE_MASK = (1u << MSE_BITS) - 1u;
    constexpr uint V_MASK = (1u << V_BITS) - 1u;
    constexpr int BITS_PER_WORD = 32;
    constexpr int BD = 32;
    constexpr int qk_per_thread = DIM / BD;
    constexpr int v_per_thread = DIM / BD;
    constexpr float QJL_SCALE = {qjl_scale}f;
    constexpr float ATTN_SCALE = {scale}f;

    uint tg = threadgroup_position_in_grid.x;
    uint block_idx = tg % n_blocks;
    uint q_offset = tg / n_blocks;
    uint q_batch_head_idx = q_offset / query_len;
    uint q_seq_idx = q_offset % query_len;
    uint simd_lid = thread_index_in_simdgroup;

    uint b = q_batch_head_idx / n_q_heads;
    uint h_q = q_batch_head_idx % n_q_heads;
    uint h_kv = h_q / n_repeats;
    uint kv_base = (b * n_kv_heads + h_kv) * seq_len;

    threadgroup float q_shmem[DIM];
    threadgroup float q_rot_shmem[DIM];
    threadgroup float q_s_shmem[DIM];
    threadgroup float tg_max_score;
    threadgroup float tg_sum_exp;
    threadgroup float tg_factor;
    threadgroup float tg_exp_score;

    thread float o[v_per_thread];
    for (int i = 0; i < v_per_thread; i++) {{
        o[i] = 0.0f;
    }}
    if (simd_lid == 0) {{
        tg_max_score = -1e30f;
        tg_sum_exp = 0.0f;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint q_base = q_offset * DIM;
    for (int t = 0; t < qk_per_thread; t++) {{
        int idx = simd_lid * qk_per_thread + t;
        q_shmem[idx] = static_cast<float>(queries[q_base + idx]) * ATTN_SCALE;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int idx = simd_lid; idx < DIM; idx += BD) {{
        float rot = 0.0f;
        float ps = 0.0f;
        for (int j = 0; j < DIM; j++) {{
            rot += q_shmem[j] * k_rotation[idx * DIM + j];
            ps += q_shmem[j] * k_qjl[idx * DIM + j];
        }}
        q_rot_shmem[idx] = rot;
        q_s_shmem[idx] = ps;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int ki = block_idx; ki < seq_len; ki += n_blocks) {{
        if (do_causal && ki > int(seq_len - query_len + q_seq_idx)) {{
            continue;
        }}

        uint kv_slot = kv_base + ki;
        uint k_packed_off = kv_slot * K_PDIM;
        uint k_signs_off = kv_slot * QJL_PDIM;
        float kn = k_norms[kv_slot];
        float gamma = k_gamma[kv_slot];

        float mse_part = 0.0f;
        float qjl_part = 0.0f;
        for (int t = 0; t < qk_per_thread; t++) {{
            int i = simd_lid * qk_per_thread + t;
            {unpack_k_inline}
            mse_part += centroids_k[val] * q_rot_shmem[i];
            int s_word = i / BITS_PER_WORD;
            int s_shift = i % BITS_PER_WORD;
            uint bit = (qjl_signs[k_signs_off + s_word] >> s_shift) & 1u;
            float sign = bit ? 1.0f : -1.0f;
            qjl_part += sign * q_s_shmem[i];
        }}
        float mse_dot = simd_sum(mse_part);
        float qjl_dot = simd_sum(qjl_part);
        float score = kn * (mse_dot + QJL_SCALE * gamma * qjl_dot);

        if (simd_lid == 0) {{
            float new_max = max(tg_max_score, score);
            tg_factor = metal::fast::exp(tg_max_score - new_max);
            tg_exp_score = metal::fast::exp(score - new_max);
            tg_max_score = new_max;
            tg_sum_exp = tg_sum_exp * tg_factor + tg_exp_score;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint v_packed_off = kv_slot * V_PDIM;
        float vn = v_norms[kv_slot];
        for (int t = 0; t < v_per_thread; t++) {{
            int d = simd_lid * v_per_thread + t;
            float vd = 0.0f;
            for (int i = 0; i < DIM; i++) {{
                {unpack_v_inline}
                vd += centroids_v[val] * v_rotation[i * DIM + d];
            }}
            o[t] = o[t] * tg_factor + tg_exp_score * vn * vd;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    uint out_slot = q_offset * n_blocks + block_idx;
    if (simd_lid == 0) {{
        sums[out_slot] = tg_sum_exp;
        maxs[out_slot] = tg_max_score;
    }}
    uint partial_base = out_slot * DIM + simd_lid * v_per_thread;
    for (int t = 0; t < v_per_thread; t++) {{
        partials[partial_base + t] = static_cast<T>(o[t]);
    }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_sdpa_2pass1_{k_bits}k_{v_bits}v_{dim}d",
        input_names=[
            "queries",
            "k_packed",
            "k_norms",
            "qjl_signs",
            "k_gamma",
            "v_packed",
            "v_norms",
            "k_rotation",
            "v_rotation",
            "centroids_k",
            "centroids_v",
            "k_qjl",
        ],
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


@lru_cache(maxsize=None)
def _tq_sdpa_2pass2_kernel(dim: int):
    """Pass 2: merge per-block partials with global online softmax."""
    source = f"""
    constexpr int DIM = {dim};
    constexpr int BN = 32;
    constexpr int BD = 32;
    constexpr int v_per_thread = DIM / BD;

    uint tg = threadgroup_position_in_grid.x;
    uint q_offset = tg;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    uint block_base = q_offset * n_blocks;
    partials += block_base * DIM + simd_gid * DIM + simd_lid * v_per_thread;
    sums += block_base;
    maxs += block_base;

    thread float o[v_per_thread];
    for (int i = 0; i < v_per_thread; i++) {{
        o[i] = 0.0f;
    }}

    float max_score = -1e30f;
    for (int b = 0; b < n_blocks / BN; ++b) {{
        max_score = max(max_score, maxs[simd_lid + BN * b]);
    }}
    float global_max = simd_max(max_score);

    float sum_exp_score = 0.0f;
    for (int b = 0; b < n_blocks / BN; ++b) {{
        float factor = metal::fast::exp(maxs[simd_lid + BN * b] - global_max);
        sum_exp_score += factor * sums[simd_lid + BN * b];
    }}
    sum_exp_score = simd_sum(sum_exp_score);

    for (int b = 0; b < n_blocks / BN; ++b) {{
        float factor = metal::fast::exp(maxs[simd_gid] - global_max);
        for (int t = 0; t < v_per_thread; t++) {{
            o[t] += factor * static_cast<float>(partials[t]);
        }}
        maxs += BN;
        sums += BN;
        partials += BN * DIM;
    }}

    threadgroup float outputs[BN * DIM];
    for (int t = 0; t < v_per_thread; t++) {{
        outputs[simd_lid * DIM + simd_gid] = o[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[t] = simd_sum(outputs[simd_gid * DIM + simd_lid]);
        o[t] = sum_exp_score == 0.0f ? o[t] : (o[t] / sum_exp_score);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    if (simd_lid == 0) {{
        uint out_base = q_offset * DIM + simd_gid * v_per_thread;
        for (int t = 0; t < v_per_thread; t++) {{
            out[out_base + t] = static_cast<T>(o[t]);
        }}
    }}
    """
    return mx.fast.metal_kernel(
        name=f"tq_sdpa_2pass2_{dim}d",
        input_names=["partials", "sums", "maxs"],
        output_names=["out"],
        source=source,
    )


def tq_sdpa_2pass_metal(
    queries: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    qjl_signs: mx.array,
    k_gamma: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    k_rotation: mx.array,
    v_rotation: mx.array,
    k_qjl: mx.array,
    k_bits: int,
    v_bits: int,
    dim: int,
    scale: float,
    do_causal: bool,
    n_blocks: int = TQ_2PASS_BLOCKS,
) -> mx.array:
    """Two-pass fused SDPA for long sequences (S >= 1024)."""
    if not metal_available():
        raise RuntimeError("Metal is not available")
    B, n_q_heads, L, _ = queries.shape
    S = k_packed.shape[2]
    n_kv_heads = k_packed.shape[1]
    n_repeats = n_q_heads // n_kv_heads
    mse_bits = k_bits - 1
    k_pdim = packed_dim(dim, mse_bits)
    v_pdim = packed_dim(dim, v_bits)
    centroids_k, _ = get_codebook(mse_bits)
    centroids_v, _ = get_codebook(v_bits)
    flat_q = mx.contiguous(queries.reshape(-1, dim))
    flat_k = mx.contiguous(k_packed.reshape(-1, k_pdim))
    flat_signs = mx.contiguous(qjl_signs.reshape(-1, qjl_packed_dim(dim)))
    flat_v = mx.contiguous(v_packed.reshape(-1, v_pdim))
    flat_kn = mx.contiguous(k_norms.reshape(-1).astype(mx.float32))
    flat_kg = mx.contiguous(k_gamma.reshape(-1).astype(mx.float32))
    flat_vn = mx.contiguous(v_norms.reshape(-1).astype(mx.float32))
    n_q = B * n_q_heads * L
    kernel1 = _tq_sdpa_2pass1_kernel(k_bits, v_bits, dim, scale)
    partials, sums, maxs = kernel1(
        inputs=[
            flat_q,
            flat_k,
            flat_kn,
            flat_signs,
            flat_kg,
            flat_v,
            flat_vn,
            k_rotation,
            v_rotation,
            centroids_k,
            centroids_v,
            k_qjl,
        ],
        template=[
            ("T", queries.dtype),
            ("n_q_heads", n_q_heads),
            ("n_kv_heads", n_kv_heads),
            ("n_repeats", n_repeats),
            ("query_len", L),
            ("seq_len", S),
            ("do_causal", do_causal),
            ("n_blocks", n_blocks),
        ],
        grid=(n_q * n_blocks * 32, 1, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(n_q * n_blocks, dim), (n_q * n_blocks,), (n_q * n_blocks,)],
        output_dtypes=[queries.dtype, mx.float32, mx.float32],
        stream=mx.gpu,
    )
    kernel2 = _tq_sdpa_2pass2_kernel(dim)
    out = kernel2(
        inputs=[partials, sums, maxs],
        template=[("T", queries.dtype), ("n_blocks", n_blocks)],
        grid=(n_q * _TQ_TILE_THREADS, 1, 1),
        threadgroup=(_TQ_TILE_THREADS, 1, 1),
        output_shapes=[(n_q, dim)],
        output_dtypes=[queries.dtype],
        stream=mx.gpu,
    )[0]
    return out.reshape(B, n_q_heads, L, dim)


def tq_sdpa_metal(
    queries: mx.array,
    k_packed: mx.array,
    k_norms: mx.array,
    qjl_signs: mx.array,
    k_gamma: mx.array,
    v_packed: mx.array,
    v_norms: mx.array,
    k_rotation: mx.array,
    v_rotation: mx.array,
    k_qjl: mx.array,
    k_bits: int,
    v_bits: int,
    dim: int,
    scale: float,
    do_causal: bool,
) -> mx.array:
    """Dispatch fused SDPA: 2-pass (S>=1024), vector (L<=8), else tiled."""
    S = k_packed.shape[2]
    L = queries.shape[2]
    if S >= TQ_2PASS_THRESHOLD:
        return tq_sdpa_2pass_metal(
            queries,
            k_packed,
            k_norms,
            qjl_signs,
            k_gamma,
            v_packed,
            v_norms,
            k_rotation,
            v_rotation,
            k_qjl,
            k_bits,
            v_bits,
            dim,
            scale,
            do_causal,
        )
    if L <= 8:
        return tq_sdpa_vector_metal(
            queries,
            k_packed,
            k_norms,
            qjl_signs,
            k_gamma,
            v_packed,
            v_norms,
            k_rotation,
            v_rotation,
            k_qjl,
            k_bits,
            v_bits,
            dim,
            scale,
            do_causal,
        )
    return tq_sdpa_tiled_metal(
        queries,
        k_packed,
        k_norms,
        qjl_signs,
        k_gamma,
        v_packed,
        v_norms,
        k_rotation,
        v_rotation,
        k_qjl,
        k_bits,
        v_bits,
        dim,
        scale,
        do_causal,
    )