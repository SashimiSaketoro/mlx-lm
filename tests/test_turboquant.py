# Copyright © 2025 Bonsai Demo contributors.

import unittest

import mlx.core as mx

from mlx_lm.models.cache import KVCache, save_prompt_cache, load_prompt_cache
from mlx_lm.turboquant.cache import AsymmetricTurboQuantCache
from mlx_lm.turboquant.factory import make_turboquant_cache
from mlx_lm.turboquant.packing import pack_indices, unpack_indices
from mlx_lm.turboquant.quantize import (
    decode_mse,
    decode_prod,
    encode_kv,
    encode_mse,
    encode_prod,
)
from mlx_lm.turboquant.rotation import make_rotation_matrix
from mlx_lm.turboquant.qjl import make_qjl_matrix, qjl_packed_dim
from mlx_lm.turboquant.attention import (
    av_weighted_sum_vectorized,
    qk_scores_vectorized,
    turboquant_scaled_dot_product_attention,
    _qk_scores_reference,
    _sdpa_decode_fallback,
)
from mlx_lm.turboquant.kernels import (
    av_weighted_sum_metal,
    decode_mse_metal,
    decode_prod_metal,
    encode_mse_metal,
    encode_prod_metal,
    metal_available,
    qk_scores_metal,
)
from mlx_lm.turboquant.quantize import _encode_mse_ref, _encode_prod_ref


class TurboQuantTests(unittest.TestCase):
    def test_pack_roundtrip(self):
        for bits in (2, 3, 4):
            dim = 128
            indices = mx.random.randint(0, 2**bits, (4, 8, dim)).astype(mx.uint8)
            packed = pack_indices(indices, bits)
            restored = unpack_indices(packed, bits, dim)
            self.assertTrue(mx.array_equal(indices, restored))

    def test_mse_roundtrip_error(self):
        dim = 128
        rotation = make_rotation_matrix(dim, 7)
        vec = mx.random.normal(shape=(2, 3, dim)).astype(mx.float32)
        packed, norms = encode_mse(vec, rotation, bits=3)
        restored = decode_mse(packed, norms, rotation, bits=3, dim=dim)
        err = mx.mean((vec - restored) ** 2).item()
        self.assertLess(err, 0.05)

    def test_prod_reconstruction(self):
        dim = 128
        rotation = make_rotation_matrix(dim, 11)
        s_matrix = make_qjl_matrix(dim, 11)
        vec = mx.random.normal(shape=(1, 2, dim)).astype(mx.float32)
        packed, norms, signs, gamma = encode_prod(vec, rotation, s_matrix, bits=4)
        restored = decode_prod(
            packed, norms, signs, gamma, rotation, s_matrix, bits=4, dim=dim
        )
        num = mx.linalg.norm(vec - restored)
        den = mx.linalg.norm(vec)
        self.assertLess((num / den).item(), 0.45)

    def test_qjl_signs_packed(self):
        dim = 128
        s_matrix = make_qjl_matrix(dim, 3)
        vec = mx.random.normal(shape=(2, dim)).astype(mx.float32)
        rotation = make_rotation_matrix(dim, 5)
        packed, _, signs, gamma = encode_prod(vec, rotation, s_matrix, bits=4)
        self.assertEqual(signs.shape[-1], qjl_packed_dim(dim))
        self.assertEqual(signs.dtype, mx.uint32)
        restored = decode_prod(
            packed, mx.ones((2, 1)), signs, gamma, rotation, s_matrix, bits=4, dim=dim
        )
        self.assertEqual(restored.shape, vec.shape)

    def test_cache_update_shapes(self):
        cache = AsymmetricTurboQuantCache(head_dim=128, k_bits=4, v_bits=3, seed=1)
        keys = mx.random.normal(shape=(1, 8, 16, 128)).astype(mx.float16)
        values = mx.random.normal(shape=(1, 8, 16, 128)).astype(mx.float16)
        k_out, v_out = cache.update_and_fetch(keys, values)
        self.assertEqual(len(k_out), 4)
        self.assertEqual(len(v_out), 2)
        self.assertEqual(k_out[0].shape[:3], (1, 8, 16))
        self.assertEqual(v_out[0].shape[:3], (1, 8, 16))
        self.assertEqual(cache.offset, 16)
        self.assertGreater(cache.nbytes, 0)
        self.assertEqual(cache._k_qjl_signs.shape[-1], qjl_packed_dim(128))
        self.assertTrue(cache.turboquant)

        k2, v2 = cache.update_and_fetch(
            mx.random.normal(shape=(1, 8, 1, 128)).astype(mx.float16),
            mx.random.normal(shape=(1, 8, 1, 128)).astype(mx.float16),
        )
        self.assertEqual(k2[0].shape[2], 17)
        self.assertEqual(v2[0].shape[2], 17)

    def test_factory_layer_adaptive(self):
        class Dummy:
            layers = [object() for _ in range(6)]

        caches = make_turboquant_cache(
            Dummy(), k_bits=4, v_bits=3, fp16_layers=1, head_dim=128, seed=0
        )
        self.assertEqual(len(caches), 6)
        self.assertIsInstance(caches[0], type(caches[5]))
        from mlx_lm.models.cache import KVCache

        self.assertIsInstance(caches[0], KVCache)
        self.assertIsInstance(caches[1], AsymmetricTurboQuantCache)
        self.assertIsInstance(caches[4], AsymmetricTurboQuantCache)
        self.assertIsInstance(caches[5], KVCache)

    def test_cache_state_roundtrip(self):
        import tempfile
        import os

        cache = AsymmetricTurboQuantCache(head_dim=128, k_bits=4, v_bits=3, seed=9)
        keys = mx.random.normal(shape=(1, 8, 8, 128)).astype(mx.float16)
        values = mx.random.normal(shape=(1, 8, 8, 128)).astype(mx.float16)
        cache.update_and_fetch(keys, values)
        nbytes_before = cache.nbytes

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tq_cache.safetensors")
            save_prompt_cache(path, [cache])
            loaded = load_prompt_cache(path)[0]

        self.assertIsInstance(loaded, AsymmetricTurboQuantCache)
        self.assertEqual(loaded.offset, 8)
        self.assertEqual(loaded.nbytes, nbytes_before)
        _, _ = loaded.update_and_fetch(
            mx.random.normal(shape=(1, 8, 1, 128)).astype(mx.float16),
            mx.random.normal(shape=(1, 8, 1, 128)).astype(mx.float16),
        )
        self.assertEqual(loaded.offset, 9)

    @unittest.skipUnless(metal_available(), "Metal not available")
    def test_metal_encode_kv_parity(self):
        dim = 128
        rotation_k = make_rotation_matrix(dim, 19)
        rotation_v = make_rotation_matrix(dim, 23)
        s_matrix = make_qjl_matrix(dim, 23)
        vec = mx.random.normal(shape=(1, 8, 64, dim)).astype(mx.float16)

        ref_k = _encode_prod_ref(vec, rotation_k, s_matrix, bits=4)
        ref_v = _encode_mse_ref(vec, rotation_v, bits=3)
        met = encode_kv(vec, vec, rotation_k, rotation_v, s_matrix, 4, 3)

        self.assertTrue(mx.array_equal(ref_k[0], met[0]))
        self.assertTrue(mx.array_equal(ref_k[2], met[2]))
        self.assertTrue(mx.array_equal(ref_v[0], met[4]))
        self.assertLess(mx.max(mx.abs(ref_k[1] - met[1])).item(), 1e-5)
        self.assertLess(mx.max(mx.abs(ref_k[3] - met[3])).item(), 1e-5)
        self.assertLess(mx.max(mx.abs(ref_v[1] - met[5])).item(), 1e-5)

    @unittest.skipUnless(metal_available(), "Metal not available")
    def test_metal_encode_parity(self):
        dim = 128
        rotation = make_rotation_matrix(dim, 19)
        s_matrix = make_qjl_matrix(dim, 23)
        vec = mx.random.normal(shape=(1, 8, 64, dim)).astype(mx.float16)

        ref_vp, ref_vn = _encode_mse_ref(vec, rotation, bits=3)
        met_vp, met_vn = encode_mse_metal(vec, rotation, bits=3, dim=dim)
        self.assertTrue(mx.array_equal(ref_vp, met_vp))
        self.assertLess(mx.max(mx.abs(ref_vn - met_vn)).item(), 1e-5)

        ref_k = _encode_prod_ref(vec, rotation, s_matrix, bits=4)
        met_k = encode_prod_metal(vec, rotation, s_matrix, bits=4, dim=dim)
        self.assertTrue(mx.array_equal(ref_k[0], met_k[0]))
        self.assertTrue(mx.array_equal(ref_k[2], met_k[2]))
        self.assertLess(mx.max(mx.abs(ref_k[1] - met_k[1])).item(), 1e-5)
        self.assertLess(mx.max(mx.abs(ref_k[3] - met_k[3])).item(), 1e-5)

    @unittest.skipUnless(metal_available(), "Metal not available")
    def test_metal_decode_parity(self):
        dim = 128
        rotation = make_rotation_matrix(dim, 13)
        s_matrix = make_qjl_matrix(dim, 17)
        vec = mx.random.normal(shape=(1, 4, 32, dim)).astype(mx.float16)
        v_pack, v_norm = encode_mse(vec, rotation, bits=3)
        k_pack, k_norm, k_sign, k_gamma = encode_prod(vec, rotation, s_matrix, bits=4)

        ref_v = decode_mse(v_pack, v_norm, rotation, bits=3, dim=dim)
        met_v = decode_mse_metal(v_pack, v_norm, rotation, bits=3, dim=dim, dtype=mx.float16)
        self.assertLess(mx.max(mx.abs(ref_v.astype(mx.float16) - met_v)).item(), 1e-3)

        ref_k = decode_prod(
            k_pack, k_norm, k_sign, k_gamma, rotation, s_matrix, bits=4, dim=dim
        )
        met_k = decode_prod_metal(
            k_pack, k_norm, k_sign, k_gamma, rotation, s_matrix, bits=4, dim=dim, dtype=mx.float16
        )
        self.assertLess(mx.max(mx.abs(ref_k.astype(mx.float16) - met_k)).item(), 1e-3)

    @unittest.skipUnless(metal_available(), "Metal not available")
    def test_metal_av_parity(self):
        dim = 128
        cache = AsymmetricTurboQuantCache(head_dim=dim, k_bits=4, v_bits=3, seed=11)
        keys = mx.random.normal(shape=(1, 8, 32, dim)).astype(mx.float16)
        values = mx.random.normal(shape=(1, 8, 32, dim)).astype(mx.float16)
        queries = mx.random.normal(shape=(1, 32, 4, dim)).astype(mx.float16)
        _, vpack = cache.update_and_fetch(keys, values)
        attn = mx.softmax(
            mx.random.normal(shape=(1, 32, 4, 32)).astype(mx.float32),
            axis=-1,
            precise=True,
        ).astype(mx.float16)
        met = av_weighted_sum_metal(
            attn, *vpack, cache._v_rotation, cache.v_bits, dim, dtype=mx.float16
        )
        ref = av_weighted_sum_vectorized(
            attn, *vpack, cache._v_rotation, cache.v_bits, dim
        )
        self.assertLess(
            mx.max(mx.abs(met.astype(mx.float32) - ref.astype(mx.float32))).item(), 1e-3
        )

    @unittest.skipUnless(metal_available(), "Metal not available")
    def test_metal_qk_scores_parity(self):
        dim = 128
        cache = AsymmetricTurboQuantCache(head_dim=dim, k_bits=4, v_bits=3, seed=7)
        keys = mx.random.normal(shape=(1, 8, 32, dim)).astype(mx.float16)
        values = mx.random.normal(shape=(1, 8, 32, dim)).astype(mx.float16)
        queries = mx.random.normal(shape=(1, 32, 4, dim)).astype(mx.float16)
        kpack, _ = cache.update_and_fetch(keys, values)
        scale = dim**-0.5
        met = qk_scores_metal(
            queries, *kpack, cache._k_rotation, cache._k_qjl, 4, dim, scale
        )
        ref = qk_scores_vectorized(
            queries, *kpack, cache._k_rotation, cache._k_qjl, 4, dim, scale
        )
        self.assertLess(
            mx.max(mx.abs(met.astype(mx.float32) - ref.astype(mx.float32))).item(), 1e-3
        )

    def test_fused_qk_scores(self):
        dim = 128
        cache = AsymmetricTurboQuantCache(head_dim=dim, k_bits=4, v_bits=3, seed=3)
        keys = mx.random.normal(shape=(1, 8, 24, dim)).astype(mx.float16)
        values = mx.random.normal(shape=(1, 8, 24, dim)).astype(mx.float16)
        queries = mx.random.normal(shape=(1, 32, 4, dim)).astype(mx.float16)
        kpack, _ = cache.update_and_fetch(keys, values)
        scale = dim**-0.5
        fused = qk_scores_vectorized(
            queries, *kpack, cache._k_rotation, cache._k_qjl, 4, dim, scale
        )
        ref = _qk_scores_reference(
            queries, *kpack, cache._k_rotation, cache._k_qjl, 4, dim, scale
        )
        self.assertLess(
            mx.max(mx.abs(fused.astype(mx.float32) - ref.astype(mx.float32))).item(), 1e-4
        )

    def test_fused_sdpa(self):
        dim = 128
        cache = AsymmetricTurboQuantCache(head_dim=dim, k_bits=4, v_bits=3, seed=5)
        keys = mx.random.normal(shape=(1, 8, 16, dim)).astype(mx.float16)
        values = mx.random.normal(shape=(1, 8, 16, dim)).astype(mx.float16)
        queries = mx.random.normal(shape=(1, 32, 4, dim)).astype(mx.float16)
        kpack, vpack = cache.update_and_fetch(keys, values)
        scale = dim**-0.5
        out = turboquant_scaled_dot_product_attention(
            queries, kpack, vpack, cache, scale, "causal"
        )
        self.assertEqual(out.shape, queries.shape)

    @unittest.skipUnless(metal_available(), "Metal not available")
    def test_tq_sdpa_2pass_parity(self):
        dim = 128
        for L, S in ((1, 2048), (1, 4096), (32, 2048)):
            cache = AsymmetricTurboQuantCache(head_dim=dim, k_bits=4, v_bits=3, seed=23)
            keys = mx.random.normal(shape=(1, 8, S, dim)).astype(mx.float16)
            values = mx.random.normal(shape=(1, 8, S, dim)).astype(mx.float16)
            queries = mx.random.normal(shape=(1, 32, L, dim)).astype(mx.float16)
            kpack, vpack = cache.update_and_fetch(keys, values)
            scale = dim**-0.5
            met = turboquant_scaled_dot_product_attention(
                queries, kpack, vpack, cache, scale, "causal"
            )
            ref = _sdpa_decode_fallback(
                queries, kpack, vpack, cache, scale, "causal"
            )
            self.assertLess(
                mx.max(mx.abs(met.astype(mx.float32) - ref.astype(mx.float32))).item(),
                1e-2,
            )

    @unittest.skipUnless(metal_available(), "Metal not available")
    def test_tq_sdpa_tiled_parity(self):
        dim = 128
        for L, S in ((32, 128), (64, 256), (128, 512)):
            cache = AsymmetricTurboQuantCache(head_dim=dim, k_bits=4, v_bits=3, seed=19)
            keys = mx.random.normal(shape=(1, 8, S, dim)).astype(mx.float16)
            values = mx.random.normal(shape=(1, 8, S, dim)).astype(mx.float16)
            queries = mx.random.normal(shape=(1, 32, L, dim)).astype(mx.float16)
            kpack, vpack = cache.update_and_fetch(keys, values)
            scale = dim**-0.5
            met = turboquant_scaled_dot_product_attention(
                queries, kpack, vpack, cache, scale, "causal"
            )
            ref = _sdpa_decode_fallback(
                queries, kpack, vpack, cache, scale, "causal"
            )
            self.assertLess(
                mx.max(mx.abs(met.astype(mx.float32) - ref.astype(mx.float32))).item(),
                1e-2,
            )

    @unittest.skipUnless(metal_available(), "Metal not available")
    def test_tq_sdpa_vector_parity(self):
        dim = 128
        for L, S in ((1, 64), (1, 512), (4, 128), (8, 256)):
            cache = AsymmetricTurboQuantCache(head_dim=dim, k_bits=4, v_bits=3, seed=17)
            keys = mx.random.normal(shape=(1, 8, S, dim)).astype(mx.float16)
            values = mx.random.normal(shape=(1, 8, S, dim)).astype(mx.float16)
            queries = mx.random.normal(shape=(1, 32, L, dim)).astype(mx.float16)
            kpack, vpack = cache.update_and_fetch(keys, values)
            scale = dim**-0.5
            met = turboquant_scaled_dot_product_attention(
                queries, kpack, vpack, cache, scale, "causal"
            )
            ref = _sdpa_decode_fallback(
                queries, kpack, vpack, cache, scale, "causal"
            )
            self.assertLess(
                mx.max(mx.abs(met.astype(mx.float32) - ref.astype(mx.float32))).item(),
                1e-2,
            )

    @unittest.skipUnless(metal_available(), "Metal not available")
    def test_metal_sdpa_parity(self):
        dim = 128
        cache = AsymmetricTurboQuantCache(head_dim=dim, k_bits=4, v_bits=3, seed=13)
        keys = mx.random.normal(shape=(1, 8, 24, dim)).astype(mx.float16)
        values = mx.random.normal(shape=(1, 8, 24, dim)).astype(mx.float16)
        queries = mx.random.normal(shape=(1, 32, 4, dim)).astype(mx.float16)
        kpack, vpack = cache.update_and_fetch(keys, values)
        scale = dim**-0.5
        met = turboquant_scaled_dot_product_attention(
            queries, kpack, vpack, cache, scale, "causal"
        )
        ref = _sdpa_decode_fallback(
            queries, kpack, vpack, cache, scale, "causal"
        )
        self.assertLess(
            mx.max(mx.abs(met.astype(mx.float32) - ref.astype(mx.float32))).item(), 1e-2
        )

    def test_tq_smaller_than_fp16(self):
        dim = 128
        seq = 256
        tq = AsymmetricTurboQuantCache(head_dim=dim, k_bits=4, v_bits=3, seed=0)
        fp16 = KVCache()
        keys = mx.zeros((1, 8, seq, dim), dtype=mx.float16)
        values = mx.zeros((1, 8, seq, dim), dtype=mx.float16)
        tq.update_and_fetch(keys, values)
        fp16.update_and_fetch(keys, values)
        mx.eval(tq.nbytes, fp16.nbytes)
        self.assertLess(tq.nbytes, fp16.nbytes)

    def test_merge_supports_batching(self):
        from mlx_lm.turboquant.cache import BatchAsymmetricTurboQuantCache
        from mlx_lm.generate import _merge_caches

        caches = []
        for length in (5, 6, 7):
            cache = AsymmetricTurboQuantCache(head_dim=128, k_bits=4, v_bits=3, seed=7)
            keys = mx.random.normal(shape=(1, 8, length, 128)).astype(mx.float16)
            values = mx.random.normal(shape=(1, 8, length, 128)).astype(mx.float16)
            cache.update_and_fetch(keys, values)
            caches.append([cache])

        merged = _merge_caches(caches)
        self.assertEqual(len(merged), 1)
        self.assertIsInstance(merged[0], BatchAsymmetricTurboQuantCache)
        self.assertEqual(merged[0]._k_packed.shape[0], 3)
        self.assertEqual(merged[0].size(), 7)

    def test_batch_prefill_matches_single(self):
        from mlx_lm.turboquant.cache import BatchAsymmetricTurboQuantCache

        dim = 128
        keys_a = mx.random.normal(shape=(1, 8, 5, dim)).astype(mx.float16)
        vals_a = mx.random.normal(shape=(1, 8, 5, dim)).astype(mx.float16)
        keys_b = mx.random.normal(shape=(1, 8, 3, dim)).astype(mx.float16)
        vals_b = mx.random.normal(shape=(1, 8, 3, dim)).astype(mx.float16)

        single_a = AsymmetricTurboQuantCache(head_dim=dim, k_bits=4, v_bits=3, seed=7)
        single_b = AsymmetricTurboQuantCache(head_dim=dim, k_bits=4, v_bits=3, seed=7)
        single_a.update_and_fetch(keys_a, vals_a)
        single_b.update_and_fetch(keys_b, vals_b)

        batch = BatchAsymmetricTurboQuantCache.merge([single_a, single_b])
        extracted_a = batch.extract(0)
        extracted_b = batch.extract(1)
        mx.eval(
            single_a._k_packed,
            extracted_a._k_packed,
            single_b._k_packed,
            extracted_b._k_packed,
        )
        self.assertEqual(extracted_a.offset, single_a.offset)
        self.assertEqual(extracted_b.offset, single_b.offset)
        self.assertTrue(
            mx.allclose(
                extracted_a._k_norms[..., : extracted_a.offset, :].astype(mx.float32),
                single_a._k_norms[..., : single_a.offset, :].astype(mx.float32),
                rtol=1e-5,
                atol=1e-5,
            )
        )
        self.assertTrue(
            mx.allclose(
                extracted_b._k_norms[..., : extracted_b.offset, :].astype(mx.float32),
                single_b._k_norms[..., : single_b.offset, :].astype(mx.float32),
                rtol=1e-5,
                atol=1e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()