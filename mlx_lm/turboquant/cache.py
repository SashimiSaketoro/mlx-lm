# Copyright © 2025 Bonsai Demo contributors.

"""TurboQuant KV cache implementations for mlx-lm."""

from __future__ import annotations

import copy
from typing import List

import mlx.core as mx

from mlx_lm.models.cache import (
    _BaseCache,
    create_attention_mask,
    create_causal_mask,
    dynamic_roll,
)
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

    @classmethod
    def merge(cls, caches: List["AsymmetricTurboQuantCache"]):
        return BatchAsymmetricTurboQuantCache.merge(caches)

    def __deepcopy__(self, memo):
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for key, value in self.__dict__.items():
            if key == "_dtype":
                setattr(obj, key, value)
            else:
                setattr(obj, key, copy.deepcopy(value, memo))
        return obj


class BatchAsymmetricTurboQuantCache(_BaseCache):
    """Batched TurboQuant KV cache for concurrent server requests."""

    step = 256
    turboquant = True

    def __init__(
        self,
        left_padding: List[int],
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
        self._k_rotation = make_rotation_matrix(head_dim, seed)
        self._v_rotation = make_rotation_matrix(head_dim, seed + 97)
        self._k_qjl = make_qjl_matrix(head_dim, seed + 193)

        self._k_packed = None
        self._k_norms = None
        self._k_qjl_signs = None
        self._k_gamma = None
        self._v_packed = None
        self._v_norms = None

        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._right_padding = None
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

    def _tensor_fields(self):
        return (
            self._k_packed,
            self._k_norms,
            self._k_qjl_signs,
            self._k_gamma,
            self._v_packed,
            self._v_norms,
        )

    def _set_tensor_fields(self, fields):
        (
            self._k_packed,
            self._k_norms,
            self._k_qjl_signs,
            self._k_gamma,
            self._v_packed,
            self._v_norms,
        ) = fields

    def _grow(self, b: int, h: int, needed: int):
        prev = self._idx
        if self._k_packed is not None and needed <= self._k_packed.shape[2]:
            return
        n = ((needed + self.step - 1) // self.step) * self.step
        shapes = (
            (b, h, n, self.k_pdim),
            (b, h, n, 1),
            (b, h, n, self.qjl_pdim),
            (b, h, n, 1),
            (b, h, n, self.v_pdim),
            (b, h, n, 1),
        )
        dtypes = (mx.uint32, mx.float32, mx.uint32, mx.float32, mx.uint32, mx.float32)
        if self._k_packed is None:
            self._set_tensor_fields(
                tuple(mx.zeros(shape, dtype=dtype) for shape, dtype in zip(shapes, dtypes))
            )
            return

        def _extend(tensor, shape, dtype):
            new = mx.zeros(shape, dtype=dtype)
            new[:, :, :prev, :] = tensor[:, :, :prev, :]
            return new

        self._set_tensor_fields(
            tuple(
                _extend(tensor, shape, dtype)
                for tensor, shape, dtype in zip(self._tensor_fields(), shapes, dtypes)
            )
        )

    def _fetch_packed(self):
        t = self._idx
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
        prev = self._idx
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
        self._idx += s
        return self._fetch_packed()

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self._k_packed is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchTurboQuantCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self._set_tensor_fields(
                tuple(
                    dynamic_roll(tensor, padding[:, None], axis=2)
                    for tensor in self._tensor_fields()
                )
            )
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def filter(self, batch_indices):
        if self._k_packed is not None:
            self._set_tensor_fields(
                tuple(tensor[batch_indices] for tensor in self._tensor_fields())
            )
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self._k_packed is not None:
                self._set_tensor_fields(
                    tuple(
                        tensor[..., min_left_pad:, :]
                        for tensor in self._tensor_fields()
                    )
                )
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other: "BatchAsymmetricTurboQuantCache"):
        if self._k_packed is None and other._k_packed is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        h = None
        if self._k_packed is not None:
            _, h, _, _ = self._k_packed.shape
        elif other._k_packed is not None:
            _, h, _, _ = other._k_packed.shape

        max_size = 0
        if self._k_packed is not None:
            max_size = max(max_size, self._k_packed.shape[2])
        if other._k_packed is not None:
            max_size = max(max_size, other._k_packed.shape[2])

        def pad(cache):
            tensors = cache._tensor_fields()
            if cache._k_packed is None:
                bc = cache.offset.shape[0]
                tensors = (
                    mx.array([]).reshape(bc, h, 0, cache.k_pdim),
                    mx.array([]).reshape(bc, h, 0, 1),
                    mx.array([]).reshape(bc, h, 0, cache.qjl_pdim),
                    mx.array([]).reshape(bc, h, 0, 1),
                    mx.array([]).reshape(bc, h, 0, cache.v_pdim),
                    mx.array([]).reshape(bc, h, 0, 1),
                )
            left = max_idx - cache._idx
            right = max_size - tensors[0].shape[2] - left
            if right < 0:
                tensors = tuple(t[..., :right, :] for t in tensors)
                right = 0
            if left != 0 or right != 0:
                pad_spec = [(0, 0), (0, 0), (left, right), (0, 0)]
                tensors = tuple(mx.pad(t, pad_spec) for t in tensors)
            left_padding = cache.left_padding + left
            return tensors, cache.offset, left_padding

        self_fields = pad(self)
        other_fields = pad(other)

        self._set_tensor_fields(
            tuple(
                mx.concatenate([a, b])
                for a, b in zip(self_fields[0], other_fields[0])
            )
        )
        self.offset, self.left_padding = map(
            mx.concatenate, zip(self_fields[1:], other_fields[1:])
        )
        self._idx = max_idx

    def extract(self, idx: int) -> AsymmetricTurboQuantCache:
        cache = AsymmetricTurboQuantCache(
            head_dim=self.head_dim,
            k_bits=self.k_bits,
            v_bits=self.v_bits,
            seed=self.seed,
        )
        if self._k_packed is None:
            return cache
        padding = self.left_padding[idx].item()
        end = self._idx
        cache._k_packed = mx.contiguous(self._k_packed[idx : idx + 1, :, padding:end, :])
        cache._k_norms = mx.contiguous(self._k_norms[idx : idx + 1, :, padding:end, :])
        cache._k_qjl_signs = mx.contiguous(
            self._k_qjl_signs[idx : idx + 1, :, padding:end, :]
        )
        cache._k_gamma = mx.contiguous(self._k_gamma[idx : idx + 1, :, padding:end, :])
        cache._v_packed = mx.contiguous(self._v_packed[idx : idx + 1, :, padding:end, :])
        cache._v_norms = mx.contiguous(self._v_norms[idx : idx + 1, :, padding:end, :])
        cache.offset = cache._k_packed.shape[2]
        cache._dtype = self._dtype
        return cache

    @classmethod
    def merge(cls, caches: List[AsymmetricTurboQuantCache]):
        if not caches:
            raise ValueError("Cannot merge an empty cache list")

        first = caches[0]
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        if max_length == 0:
            return cls(
                [0] * len(caches),
                head_dim=first.head_dim,
                k_bits=first.k_bits,
                v_bits=first.v_bits,
                seed=first.seed,
            )

        padding = [max_length - length for length in lengths]
        b = len(caches)
        h = next(c._k_packed.shape[1] for c in caches if not c.empty())
        k_pdim = first.k_pdim
        v_pdim = first.v_pdim
        qjl_pdim = first.qjl_pdim

        k_packed = mx.zeros((b, h, max_length, k_pdim), dtype=mx.uint32)
        k_norms = mx.zeros((b, h, max_length, 1), dtype=mx.float32)
        k_signs = mx.zeros((b, h, max_length, qjl_pdim), dtype=mx.uint32)
        k_gamma = mx.zeros((b, h, max_length, 1), dtype=mx.float32)
        v_packed = mx.zeros((b, h, max_length, v_pdim), dtype=mx.uint32)
        v_norms = mx.zeros((b, h, max_length, 1), dtype=mx.float32)

        for i, (pad, cache) in enumerate(zip(padding, caches)):
            if cache.empty():
                continue
            t = cache.offset
            sl = slice(pad, pad + t)
            k_packed[i : i + 1, :, sl, :] = cache._k_packed[..., :t, :]
            k_norms[i : i + 1, :, sl, :] = cache._k_norms[..., :t, :]
            k_signs[i : i + 1, :, sl, :] = cache._k_qjl_signs[..., :t, :]
            k_gamma[i : i + 1, :, sl, :] = cache._k_gamma[..., :t, :]
            v_packed[i : i + 1, :, sl, :] = cache._v_packed[..., :t, :]
            v_norms[i : i + 1, :, sl, :] = cache._v_norms[..., :t, :]

        batch = cls(
            padding,
            head_dim=first.head_dim,
            k_bits=first.k_bits,
            v_bits=first.v_bits,
            seed=first.seed,
        )
        batch._k_packed = k_packed
        batch._k_norms = k_norms
        batch._k_qjl_signs = k_signs
        batch._k_gamma = k_gamma
        batch._v_packed = v_packed
        batch._v_norms = v_norms
        batch._idx = max_length
        batch.offset += max_length
        return batch

    def size(self):
        return self._idx

    def empty(self):
        return self._k_packed is None

    @property
    def nbytes(self):
        if self._k_packed is None:
            return 0
        t = self._idx
        total = 0
        for tensor in self._tensor_fields():
            total += tensor[..., :t, :].nbytes
        return total

    def __deepcopy__(self, memo):
        cls = self.__class__
        obj = cls.__new__(cls)
        memo[id(self)] = obj
        for key, value in self.__dict__.items():
            if key == "_dtype":
                setattr(obj, key, value)
            elif key in ("_k_rotation", "_v_rotation", "_k_qjl"):
                setattr(obj, key, value)
            else:
                setattr(obj, key, copy.deepcopy(value, memo))
        return obj