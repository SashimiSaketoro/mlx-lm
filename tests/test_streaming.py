# Copyright © 2024 Apple Inc.

import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_lm.streaming.config import StreamingConfig
from mlx_lm.streaming.layer_loader import RollingWindowLoader
from mlx_lm.streaming.split_model import split_model_by_layers


def test_streaming_config_window_size():
    cfg = StreamingConfig(max_memory_gb=16.0, safety_margin=0.8)
    layer_size = 200_000_000
    window = cfg.compute_optimal_window(layer_size, kv_cache_bytes=1_000_000_000)
    assert window >= 1


def test_split_and_load_layers():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        weights = {
            "model.layers.0.self_attn.q_proj.weight": mx.random.normal((8, 8)),
            "model.layers.1.self_attn.q_proj.weight": mx.random.normal((8, 8)),
            "model.embed_tokens.weight": mx.random.normal((100, 8)),
            "model.norm.weight": mx.ones((8,)),
        }
        model_path = tmp_path / "model.safetensors"
        mx.save_safetensors(str(model_path), weights)

        out_dir = tmp_path / "split"
        split_model_by_layers(model_path, out_dir)

        assert (out_dir / "layer_0.safetensors").exists()
        assert (out_dir / "fixed_weights.safetensors").exists()

        cfg = StreamingConfig(window_size=2, verbose=False)
        loader = RollingWindowLoader(
            model_path=out_dir,
            num_layers=2,
            layer_size_bytes=1024,
            streaming_config=cfg,
        )
        lw = loader.get_layer(0)
        assert isinstance(lw.weights, dict)
        assert all(isinstance(v, mx.array) for v in lw.weights.values())
        assert "self_attn.q_proj.weight" in lw.weights
        assert isinstance(lw.weights["self_attn.q_proj.weight"], mx.array)