#!/usr/bin/env python3
# Copyright © 2024 Apple Inc.

"""Consumer verification for mlx-lm layer-streaming (MLX-first tensor paths)."""

import json
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mlx.core as mx

from mlx_lm.generate import generate_step
from mlx_lm.models.llama import Model, ModelArgs
from mlx_lm.streaming import StreamingConfig, load_streaming
from mlx_lm.streaming.split_model import ensure_streaming_layout, split_model_by_layers
from mlx_lm.utils import save_model


def _tiny_config():
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


def _build_model_dir(tmp_path: Path) -> Path:
    config = _tiny_config()
    with open(tmp_path / "config.json", "w") as f:
        json.dump(config, f)
    args = ModelArgs.from_dict(config)
    model = Model(args)
    mx.eval(model.parameters())
    save_model(tmp_path, model)
    ensure_streaming_layout(tmp_path, verbose=False)
    return tmp_path


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = _build_model_dir(Path(tmp))

        # Split I/O uses mx.load / mx.save_safetensors
        weights = mx.load(str(model_dir / "layer_0.safetensors"))
        assert all(isinstance(v, mx.array) for v in weights.values())
        print("SPLIT OK", isinstance(weights["self_attn.q_proj.weight"], mx.array))

        model, _, _ = load_streaming(
            str(model_dir),
            StreamingConfig(window_size=1, verbose=False),
            load_tokenizer=False,
        )
        print("LOAD OK", type(model).__name__)

        cache = model.make_cache()
        logits = model(mx.array([[1, 2, 3]]), cache=cache)
        mx.eval(logits)
        assert isinstance(logits, mx.array)
        print("FORWARD OK", isinstance(logits, mx.array), logits.shape)

        prompt = mx.array([1, 2, 3])
        for token, logprobs in generate_step(prompt, model, max_tokens=2):
            assert isinstance(logprobs, mx.array)
            assert isinstance(token, (int, mx.integer_types if hasattr(mx, "integer_types") else int))
        print("SAMPLE OK", isinstance(logprobs, mx.array))

        # Standalone split entry path
        mono = tmp + "/mono.safetensors"
        mx.save_safetensors(
            mono,
            {
                "model.layers.0.self_attn.q_proj.weight": mx.random.normal((8, 8)),
                "model.embed_tokens.weight": mx.random.normal((100, 8)),
            },
        )
        out = Path(tmp) / "split2"
        split_model_by_layers(Path(mono), out)
        assert (out / "layer_0.safetensors").exists()
        print("SPLIT_CLI OK", True)

    print("ALL CONSUMER TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())