# Copyright © 2024 Apple Inc.

"""Benchmark layer-streaming generation throughput."""

import argparse
import time

from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from .config import StreamingConfig
from .load import load_streaming


def benchmark_streaming(
    model_path: str,
    prompt_tokens: list,
    max_tokens: int = 32,
    max_memory_gb: float = 20.0,
    verbose: bool = True,
) -> dict:
    """Run a streaming benchmark and return timing stats."""
    import mlx.core as mx

    config = StreamingConfig(max_memory_gb=max_memory_gb, verbose=verbose)
    model, _tokenizer, _cfg = load_streaming(model_path, streaming_config=config)

    prompt = mx.array([prompt_tokens])
    sampler = make_sampler(temp=0.0)

    start = time.perf_counter()
    count = 0
    for _ in generate_step(prompt, model, max_tokens=max_tokens, sampler=sampler):
        count += 1
    elapsed = time.perf_counter() - start

    stats = {
        "tokens": count,
        "elapsed_s": elapsed,
        "tok_per_s": count / elapsed if elapsed > 0 else 0.0,
        "ms_per_token": (elapsed / count * 1000) if count > 0 else 0.0,
        "streaming": model.get_stats(),
    }
    if verbose:
        print(f"Tokens: {stats['tokens']}")
        print(f"Throughput: {stats['tok_per_s']:.2f} tok/s")
        print(f"Window: {stats['streaming']['streaming']['window_size']} layers")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark layer-streaming inference")
    parser.add_argument("--model", required=True, help="Model path or HF repo")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-memory-gb", type=float, default=20.0)
    parser.add_argument("--prompt-ids", default="1,2,3,4", help="Comma-separated token ids")
    args = parser.parse_args()
    prompt_tokens = [int(x) for x in args.prompt_ids.split(",")]
    benchmark_streaming(
        args.model,
        prompt_tokens,
        max_tokens=args.max_tokens,
        max_memory_gb=args.max_memory_gb,
    )


if __name__ == "__main__":
    main()