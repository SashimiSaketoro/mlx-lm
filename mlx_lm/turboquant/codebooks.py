# Copyright © 2025 Bonsai Demo contributors.

"""Lloyd-Max scalar codebooks for TurboQuant on the unit-sphere Beta distribution.

Centroids are precomputed for head_dim=128 (Bonsai GQA) via the Beta coordinate
distribution from TurboQuant (arXiv:2504.19874), Lemma 1.
"""

from __future__ import annotations

import mlx.core as mx

HEAD_DIM = 128

# Lloyd-Max centroids for d=128, computed with MLX grid integration.
_CENTROIDS = {
    2: [
        -0.133163,
        -0.039921,
        0.040372,
        0.13355,
    ],
    3: [
        -0.189914,
        -0.119752,
        -0.067866,
        -0.021982,
        0.022232,
        0.067866,
        0.119752,
        0.189914,
    ],
    4: [
        -0.277515,
        -0.225104,
        -0.185344,
        -0.149934,
        -0.116278,
        -0.082991,
        -0.0497,
        -0.016406,
        0.016889,
        0.050183,
        0.083261,
        0.116478,
        0.150416,
        0.185657,
        0.225104,
        0.277516,
    ],
}

_BOUNDARIES = {
    2: [-0.566582, -0.086542, 0.000226, 0.086961, 0.566775],
    3: [
        -0.594957,
        -0.154833,
        -0.093809,
        -0.044924,
        0.000125,
        0.045049,
        0.093809,
        0.154833,
        0.594957,
    ],
    4: [
        -0.638758,
        -0.25131,
        -0.205224,
        -0.167639,
        -0.133106,
        -0.099634,
        -0.066346,
        -0.033053,
        0.000241,
        0.033536,
        0.066722,
        0.09987,
        0.133447,
        0.168036,
        0.20538,
        0.25131,
        0.638758,
    ],
}


def get_codebook(bits: int) -> tuple[mx.array, mx.array]:
    if bits not in _CENTROIDS:
        raise ValueError(
            f"Unsupported TurboQuant bit width {bits}. Supported: {sorted(_CENTROIDS)}"
        )
    centroids = mx.array(_CENTROIDS[bits], dtype=mx.float32)
    boundaries = mx.array(_BOUNDARIES[bits], dtype=mx.float32)
    return centroids, boundaries


def quantize_coords(values: mx.array, bits: int) -> mx.array:
    """Nearest-centroid quantization along the last axis."""
    centroids, _ = get_codebook(bits)
    # values: (..., d), centroids: (2**bits,)
    expanded = values[..., None]
    dist = mx.abs(expanded - centroids)
    return mx.argmin(dist, axis=-1).astype(mx.uint8)


def dequantize_coords(indices: mx.array, bits: int) -> mx.array:
    centroids, _ = get_codebook(bits)
    return centroids[indices.astype(mx.int32)]