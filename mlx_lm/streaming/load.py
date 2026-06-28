# Copyright © 2024 Apple Inc.

"""Load mlx-lm models for layer-streaming inference."""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import _download, _get_classes, load_config, load_tokenizer

from .config import StreamingConfig
from .layer_loader import RollingWindowLoader
from .wrapper import StreamingModelWrapper


def _dtype_bytes(config: dict) -> int:
    q = config.get("quantization") or config.get("quantization_config")
    if q:
        bits = q.get("bits", 4) if isinstance(q, dict) else 4
        return max(1, bits // 8)
    torch_dtype = config.get("torch_dtype", "bfloat16")
    if "float32" in str(torch_dtype):
        return 4
    return 2


def _estimate_kv_cache_bytes(config: dict, streaming_config: StreamingConfig) -> int:
    n_layers = config.get("num_hidden_layers", 32)
    n_kv = config.get("num_key_value_heads") or config.get("num_attention_heads", 32)
    hidden = config.get("hidden_size", 4096)
    n_heads = config.get("num_attention_heads", 32)
    head_dim = hidden // n_heads
    bpe = 2
    return int(
        2 * n_layers * streaming_config.max_seq_len * n_kv * head_dim * bpe
    )


def load_streaming(
    path_or_hf_repo: str,
    streaming_config: Optional[StreamingConfig] = None,
    tokenizer_config: Optional[dict] = None,
    revision: Optional[str] = None,
) -> Tuple[StreamingModelWrapper, TokenizerWrapper, Dict]:
    """
    Load a model for layer-streaming inference.

    The model directory should contain per-layer safetensors (from
    ``mlx_lm.streaming.split_model``) or a monolithic ``model*.safetensors``
    fallback.

    Args:
        path_or_hf_repo: Local path or HuggingFace repo id.
        streaming_config: Memory/window settings.
        tokenizer_config: Passed to ``load_tokenizer``.
        revision: HF revision id.

    Returns:
        (StreamingModelWrapper, tokenizer, config dict)
    """
    streaming_config = streaming_config or StreamingConfig()
    tokenizer_config = tokenizer_config or {}

    model_path = _download(path_or_hf_repo, revision=revision)
    config = load_config(model_path)

    model_class, model_args_class = _get_classes(config)
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    fixed_file = model_path / "fixed_weights.safetensors"
    if fixed_file.exists():
        fixed_weights = mx.load(str(fixed_file))
        if hasattr(model, "sanitize"):
            fixed_weights = model.sanitize(fixed_weights)
        model.load_weights(list(fixed_weights.items()), strict=False)
        mx.eval(model.parameters())
    else:
        raise FileNotFoundError(
            f"{fixed_file} not found. Run: python -m mlx_lm.streaming.split_model "
            f"<model.safetensors> {model_path}"
        )

    layer_size = streaming_config.estimate_layer_memory(
        hidden_size=config["hidden_size"],
        intermediate_size=config["intermediate_size"],
        num_attention_heads=config["num_attention_heads"],
        num_key_value_heads=config.get("num_key_value_heads")
        or config["num_attention_heads"],
        dtype_bytes=_dtype_bytes(config),
    )

    kv_bytes = _estimate_kv_cache_bytes(config, streaming_config)
    loader = RollingWindowLoader(
        model_path=model_path,
        num_layers=config["num_hidden_layers"],
        layer_size_bytes=layer_size,
        streaming_config=streaming_config,
        kv_cache_bytes=kv_bytes,
    )
    loader.preload_window()

    wrapped = StreamingModelWrapper(model, loader)
    wrapped.eval()

    tokenizer = load_tokenizer(model_path, tokenizer_config)
    return wrapped, tokenizer, config