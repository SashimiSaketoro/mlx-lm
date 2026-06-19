# Copyright © 2025 Bonsai Demo contributors.

"""TurboQuant KV cache implementations for mlx-lm."""

from __future__ import annotations

import mlx.core as mx

from mlx_lm.models.cache import _BaseCache, create_attention_mask
from mlx_lm.turboquant.codebooks import HEAD_DIM
from mlx_lm.turboquant.packing import packed_dim
from mlx_lm.turboquant.qjl import make_qjl_matrix, qjl_packed_dim
from mlx_lm.turboquant.quantize import encode_kv
from mlx_lm.turboquant.rotation import make_rotation_matrix


class AsymmetricTurboQuantCache(_BaseCache):
    """KV cache with TurboQuant_prod keys and TurboQuant_mse values.

    Keys use an inner-product-preserving quantizer (Lloyd-Max + QJL residual).
    Values use an MSE-optimal quantizer. Packed tensors are returned from
    ``update_and_fetch`` for fused attention and fused KV encode.
    """

    step = 256
    turboquant = True

    def __init__(
        self,
        head_dim: int = HEAD_DIM,
        k_bits: int = 4,
        v_bits: int = 3,
        seed: int = 42,
    ):
        if k_bits < 2:
            raise ValueError("k_bits must be >= 2 for TurboQuant_prod")
        if v_bits not in (2, 3, 4):
            raise ValueError("v_bits must be 2, 3, or 4")
        self.head_dim = head_dim
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.seed = seed
        self.offset = 0

        self._k_rotation = make_rotation_matrix(head_dim, seed)
        self._v_rotation = make_rotation_matrix(head_dim, seed + 97)
        self._k_qjl = make_qjl_matrix(head_dim, seed + 193)

        self._k_packed = None
        self._k_norms = None
        self._k_qjl_signs = None
        self._k_gamma = None

        self._v_packed = None
        self._v_norms = None

        self._dtype = mx.float16

    @property
    def k_pdim(self) -> int:
        return packed_dim(self.head_dim, self.k_bits - 1)

    @property
    def v_pdim(self) -> int:
        return packed_dim(self.head_dim, self.v_bits)

    @property
    def qjl_pdim(self) -> int:
        return qjl_packed_dim(self.head_dim)

    def _grow(self, b: int, h: int, needed: int):
        prev = self.offset
        if self._k_packed is not None and needed <= self._k_packed.shape[2]:
            return
        n = ((needed + self.step - 1) // self.step) * self.step
        k_shape = (b, h, n, self.k_pdim)
        v_shape = (b, h, n, self.v_pdim)
        if self._k_packed is None:
            self._k_packed = mx.zeros(k_shape, dtype=mx.uint32)
            self._k_norms = mx.zeros((b, h, n, 1), dtype=mx.float32)
            self._k_qjl_signs = mx.zeros((b, h, n, self.qjl_pdim), dtype=mx.uint32)
            self._k_gamma = mx.zeros((b, h, n, 1), dtype=mx.float32)
            self._v_packed = mx.zeros(v_shape, dtype=mx.uint32)
            self._v_norms = mx.zeros((b, h, n, 1), dtype=mx.float32)
            return

        def _extend(tensor, shape):
            new = mx.zeros(shape, dtype=tensor.dtype)
            new[:, :, :prev, :] = tensor[:, :, :prev, :]
            return new

        self._k_packed = _extend(self._k_packed, k_shape)
        self._k_norms = _extend(self._k_norms, (b, h, n, 1))
        self._k_qjl_signs = _extend(self._k_qjl_signs, (b, h, n, self.qjl_pdim))
        self._k_gamma = _extend(self._k_gamma, (b, h, n, 1))
        self._v_packed = _extend(self._v_packed, v_shape)
        self._v_norms = _extend(self._v_norms, (b, h, n, 1))

    def _fetch_packed(self):
        t = self.offset
        keys = (
            self._k_packed[..., :t, :],
            self._k_norms[..., :t, :],
            self._k_qjl_signs[..., :t, :],
            self._k_gamma[..., :t, :],
        )
        values = (self._v_packed[..., :t, :], self._v_norms[..., :t, :])
        return keys, values

    def update_and_fetch(self, keys, values):
        b, h, s, d = keys.shape
        if d != self.head_dim:
            raise ValueError(f"Expected head_dim={self.head_dim}, got {d}")
        self._dtype = keys.dtype
        prev = self.offset
        self._grow(b, h, prev + s)

        k_pack, k_norm, k_sign, k_gamma, v_pack, v_norm = encode_kv(
            keys,
            values,
            self._k_rotation,
            self._v_rotation,
            self._k_qjl,
            self.k_bits,
            self.v_bits,
        )

        self._k_packed[..., prev : prev + s, :] = k_pack
        self._k_norms[..., prev : prev + s, :] = k_norm
        self._k_qjl_signs[..., prev : prev + s, :] = k_sign
        self._k_gamma[..., prev : prev + s, :] = k_gamma
        self._v_packed[..., prev : prev + s, :] = v_pack
        self._v_norms[..., prev : prev + s, :] = v_norm
        self.offset += s

        return self._fetch_packed()

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self.offset, n)
        self.offset -= n
        return n

    def size(self):
        return self.offset

    def empty(self):
        return self._k_packed is None

    @property
    def nbytes(self):
        if self._k_packed is None:
            return 0
        t = self.offset
        total = 0
        total += self._k_packed[..., :t, :].nbytes
        total += self._k_norms[..., :t, :].nbytes
        total += self._k_qjl_signs[..., :t, :].nbytes
        total += self._k_gamma[..., :t, :].nbytes
        total += self._v_packed[..., :t, :].nbytes
        total += self._v_norms[..., :t, :].nbytes
        return total

    @property
    def state(self):
        if self.empty():
            return []
        t = self.offset
        return [
            self._k_packed[..., :t, :],
            self._k_norms[..., :t, :],
            self._k_qjl_signs[..., :t, :],
            self._k_gamma[..., :t, :],
            self._v_packed[..., :t, :],
            self._v_norms[..., :t, :],
        ]

    @state.setter
    def state(self, value):
        if not value:
            return
        (
            self._k_packed,
            self._k_norms,
            self._k_qjl_signs,
            self._k_gamma,
            self._v_packed,
            self._v_norms,
        ) = value
        self.offset = self._k_packed.shape[2]

    @property
    def meta_state(self):
        return f"{self.offset},{self.k_bits},{self.v_bits},{self.seed},{self.head_dim}"

    @meta_state.setter
    def meta_state(self, meta):
        parts = meta.split(",")
        self.offset = int(parts[0])
        self.k_bits = int(parts[1])
        self.v_bits = int(parts[2])
        self.seed = int(parts[3])
        self.head_dim = int(parts[4])
        self._k_rotation = make_rotation_matrix(self.head_dim, self.seed)
        self._v_rotation = make_rotation_matrix(self.head_dim, self.seed + 97)
        self._k_qjl = make_qjl_matrix(self.head_dim, self.seed + 193)

    @classmethod
    def from_state(cls, state, meta_state):
        obj = cls.__new__(cls)
        obj._k_packed = None
        obj._k_norms = None
        obj._k_qjl_signs = None
        obj._k_gamma = None
        obj._v_packed = None
        obj._v_norms = None
        obj._dtype = mx.float16
        obj.meta_state = meta_state
        obj.state = state
        return obj