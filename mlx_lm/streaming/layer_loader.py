# Copyright © 2024 Apple Inc.

"""Rolling-window layer weight loader with async prefetch."""

import mlx.core as mx
from collections import deque
from pathlib import Path
from typing import Dict, Optional

from .config import StreamingConfig


class LayerWeights:
    """Weights for a single transformer layer."""

    def __init__(self, weights: Dict[str, mx.array], layer_idx: int):
        self.weights = weights
        self.layer_idx = layer_idx
        self._memory_bytes: Optional[int] = None

    @property
    def memory_bytes(self) -> int:
        if self._memory_bytes is None:
            self._memory_bytes = sum(w.nbytes for w in self.weights.values())
        return self._memory_bytes


class RollingWindowLoader:
    """
    Sliding window of layer safetensors with eviction and prefetch.

    Expects per-layer files produced by ``split_model_by_layers``:
    ``layer_{i}.safetensors`` plus optional ``fixed_weights.safetensors``.
    """

    def __init__(
        self,
        model_path: Path,
        num_layers: int,
        layer_size_bytes: int,
        streaming_config: StreamingConfig,
        kv_cache_bytes: int = 0,
        layer_key_prefix: str = "model.layers",
    ):
        self.model_path = Path(model_path)
        self.num_layers = num_layers
        self.streaming_config = streaming_config
        self.layer_key_prefix = layer_key_prefix

        if streaming_config.window_size is None:
            self.window_size = streaming_config.compute_optimal_window(
                layer_size_bytes, kv_cache_bytes
            )
        else:
            self.window_size = streaming_config.window_size

        if streaming_config.verbose:
            print(f"RollingWindowLoader: path={model_path}, window={self.window_size}")

        self.layer_queue: deque = deque(maxlen=self.window_size)
        self.loaded_layers: set = set()
        self._layer_files = self._discover_layer_files()

    def _discover_layer_files(self) -> Dict[int, Optional[Path]]:
        patterns = [
            "layer_{}.safetensors",
            "model.layers.{}.safetensors",
            "layers.{}.safetensors",
        ]
        layer_files = {}
        for layer_idx in range(self.num_layers):
            found = None
            for pattern in patterns:
                path = self.model_path / pattern.format(layer_idx)
                if path.exists():
                    found = path
                    break
            layer_files[layer_idx] = found
        return layer_files

    def _load_from_monolithic(self, layer_idx: int) -> Dict[str, mx.array]:
        model_file = self.model_path / "model.safetensors"
        if not model_file.exists():
            weight_files = list(self.model_path.glob("model*.safetensors"))
            if not weight_files:
                raise FileNotFoundError(
                    f"No weights for layer {layer_idx} in {self.model_path}"
                )
            model_file = weight_files[0]

        all_weights = mx.load(str(model_file))
        patterns = [
            f"{self.layer_key_prefix}.{layer_idx}.",
            f"layers.{layer_idx}.",
            f"transformer.h.{layer_idx}.",
        ]
        weights = {}
        for key, tensor in all_weights.items():
            for pattern in patterns:
                if key.startswith(pattern):
                    weights[key[len(pattern):]] = tensor
                    break
        if not weights:
            raise FileNotFoundError(
                f"Layer {layer_idx} not found in {model_file}"
            )
        return weights

    def load_layer(self, layer_idx: int, prefetch: bool = False) -> LayerWeights:
        layer_file = self._layer_files.get(layer_idx)
        if layer_file is None:
            weights = self._load_from_monolithic(layer_idx)
        else:
            weights = mx.load(str(layer_file))

        layer_weights = LayerWeights(weights, layer_idx)
        if prefetch and weights:
            mx.async_eval(list(weights.values()))
        return layer_weights

    def preload_window(self, start_idx: int = 0):
        end_idx = min(start_idx + self.window_size, self.num_layers)
        for i in range(start_idx, end_idx):
            lw = self.load_layer(i, prefetch=True)
            self.layer_queue.append(lw)
            self.loaded_layers.add(i)
        all_arrays = []
        for layer in self.layer_queue:
            all_arrays.extend(layer.weights.values())
        if all_arrays:
            mx.eval(all_arrays)

    def get_layer(self, layer_idx: int) -> LayerWeights:
        for layer in self.layer_queue:
            if layer.layer_idx == layer_idx:
                return layer

        if len(self.layer_queue) >= self.window_size:
            evicted = self.layer_queue.popleft()
            self.loaded_layers.discard(evicted.layer_idx)
            del evicted
            if (
                layer_idx % self.streaming_config.clear_cache_every_n_layers == 0
            ):
                mx.clear_cache()

        layer_weights = self.load_layer(layer_idx, prefetch=False)
        mx.eval(list(layer_weights.weights.values()))
        self.layer_queue.append(layer_weights)
        self.loaded_layers.add(layer_idx)

        for offset in range(1, self.streaming_config.prefetch_layers + 1):
            next_idx = layer_idx + offset
            if (
                next_idx < self.num_layers
                and next_idx not in self.loaded_layers
            ):
                self.load_layer(next_idx, prefetch=True)

        return layer_weights

    def get_memory_usage(self) -> dict:
        total_bytes = sum(layer.memory_bytes for layer in self.layer_queue)
        return {
            "loaded_layers": len(self.loaded_layers),
            "layer_indices": sorted(list(self.loaded_layers)),
            "total_mb": total_bytes / 1e6,
            "total_gb": total_bytes / 1e9,
            "window_size": self.window_size,
        }

    def clear(self):
        self.layer_queue.clear()
        self.loaded_layers.clear()
        mx.clear_cache()