# Copyright © 2024 Apple Inc.

"""Streaming inference configuration and memory estimation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StreamingConfig:
    """Runtime settings for layer-streaming inference."""

    max_memory_gb: float = 20.0
    window_size: Optional[int] = None
    safety_margin: float = 0.8
    prefetch_layers: int = 1
    clear_cache_every_n_layers: int = 5
    max_seq_len: int = 4096
    verbose: bool = False

    def estimate_layer_memory(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        dtype_bytes: int = 2,
    ) -> int:
        """Estimate bytes for one transformer layer."""
        head_dim = hidden_size // num_attention_heads
        kv_dim = head_dim * num_key_value_heads

        attn_params = (
            hidden_size * hidden_size
            + hidden_size * kv_dim * 2
            + hidden_size * hidden_size
        )
        ffn_params = (
            hidden_size * intermediate_size * 2
            + intermediate_size * hidden_size
        )
        norm_params = hidden_size * 2
        total_params = attn_params + ffn_params + norm_params
        return int(total_params * dtype_bytes)

    def compute_optimal_window(
        self,
        layer_size_bytes: int,
        kv_cache_bytes: int,
        fixed_memory_bytes: int = 2_000_000_000,
    ) -> int:
        """Compute how many layers fit in the rolling window."""
        available = self.max_memory_gb * 1_000_000_000 * self.safety_margin
        activation_overhead = layer_size_bytes * 2
        usable = available - kv_cache_bytes - activation_overhead - fixed_memory_bytes
        window = max(1, int(usable / layer_size_bytes))

        if self.verbose:
            print("Streaming memory breakdown:")
            print(f"  Available: {available / 1e9:.2f} GB")
            print(f"  KV cache: {kv_cache_bytes / 1e9:.2f} GB")
            print(f"  Activations: {activation_overhead / 1e9:.2f} GB")
            print(f"  Fixed overhead: {fixed_memory_bytes / 1e9:.2f} GB")
            print(f"  Layer size: {layer_size_bytes / 1e6:.2f} MB")
            print(f"  Window size: {window} layers")

        return window