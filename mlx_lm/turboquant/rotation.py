# Copyright © 2025 Bonsai Demo contributors.

"""Random orthogonal rotations for TurboQuant (QR on a Gaussian draw)."""

from __future__ import annotations

import mlx.core as mx


def make_rotation_matrix(dim: int, seed: int) -> mx.array:
    """Return a row-orthonormal rotation matrix of shape (dim, dim)."""
    if dim <= 0:
        raise ValueError("dim must be positive")
    cpu = mx.cpu
    mx.random.seed(seed)
    with mx.stream(cpu):
        gaussian = mx.random.normal(shape=(dim, dim))
        q, _ = mx.linalg.qr(gaussian)
        mx.eval(q)
    return q.astype(mx.float32)