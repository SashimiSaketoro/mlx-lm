# Copyright © 2024 Apple Inc.

"""End-to-end tests for layer-streaming through mlx_lm.generate."""

import json
import tempfile
from pathlib import Path

import mlx.core as mx
import pytest

from mlx_lm.generate import generate_step
from mlx_lm.models.llama import Model, ModelArgs
from mlx_lm.streaming import StreamingConfig, load_streaming
from mlx_lm.streaming.split_model import ensure_streaming_layout
from mlx_lm.utils import save_model


def _tiny_llama_config():
    return {
        "model_type": "llama",
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5,
        "tie_word_embeddings": True,
    }


def _build_tiny_model_dir(tmp_path: Path) -> Path:
    config = _tiny_llama_config()
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)

    args = ModelArgs.from_dict(config)
    model = Model(args)
    mx.eval(model.parameters())
    save_model(tmp_path, model)
    ensure_streaming_layout(tmp_path, verbose=False)
    return tmp_path


def test_load_streaming_auto_split():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = _build_tiny_model_dir(Path(tmp))
        assert (model_dir / "fixed_weights.safetensors").exists()
        assert (model_dir / "layer_0.safetensors").exists()

        model, tokenizer, config = load_streaming(
            str(model_dir),
            StreamingConfig(window_size=1, verbose=False),
            load_tokenizer=False,
        )
        assert tokenizer is None
        stats = model.get_stats()
        assert stats["streaming"]["window_size"] >= 1
        assert config["num_hidden_layers"] == 2


def test_streaming_forward_returns_mx_array():
    """StreamingModelWrapper forward must return mx.array logits."""
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = _build_tiny_model_dir(Path(tmp))
        model, _, _ = load_streaming(
            str(model_dir),
            StreamingConfig(window_size=1, verbose=False),
            load_tokenizer=False,
        )

        cache = model.make_cache()
        inputs = mx.array([[1, 2, 3]])
        logits = model(inputs, cache=cache)
        mx.eval(logits)

        assert isinstance(logits, mx.array)
        assert logits.ndim == 3
        assert logits.shape[-1] == _tiny_llama_config()["vocab_size"]


def test_generate_step_on_streaming_model():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = _build_tiny_model_dir(Path(tmp))
        model, _, _ = load_streaming(
            str(model_dir),
            StreamingConfig(window_size=1, verbose=False),
            load_tokenizer=False,
        )

        prompt = mx.array([1, 2, 3])
        tokens = []
        for token, _logprobs in generate_step(prompt, model, max_tokens=3):
            if isinstance(token, mx.array):
                tokens.append(int(token.item()))
            else:
                tokens.append(int(token))

        assert len(tokens) == 3


def test_generate_step_sampling_returns_mx_array_logprobs():
    """generate_step logprobs must be mx.array through the streaming path."""
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = _build_tiny_model_dir(Path(tmp))
        model, _, _ = load_streaming(
            str(model_dir),
            StreamingConfig(window_size=1, verbose=False),
            load_tokenizer=False,
        )

        prompt = mx.array([1, 2, 3])
        for _token, logprobs in generate_step(prompt, model, max_tokens=2):
            assert isinstance(logprobs, mx.array)
            assert logprobs.ndim == 1
            assert logprobs.shape[0] == _tiny_llama_config()["vocab_size"]