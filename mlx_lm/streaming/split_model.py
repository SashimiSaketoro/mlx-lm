# Copyright © 2024 Apple Inc.

"""Split monolithic safetensors into per-layer files for streaming inference."""

import argparse
import mlx.core as mx
from pathlib import Path
from typing import Dict, List


def split_model_by_layers(
    model_path: Path,
    output_dir: Path,
    layer_prefix: str = "model.layers",
) -> None:
    """
    Split a monolithic safetensors model into per-layer files.

    Writes ``layer_{i}.safetensors`` with keys stripped of the layer prefix,
    and ``fixed_weights.safetensors`` for embeddings, norm, and lm_head.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Splitting {model_path} -> {output_dir}")

    all_weights = mx.load(str(model_path))
    layer_keys: Dict[int, List[str]] = {}
    fixed_keys: List[str] = []

    for key in all_weights.keys():
        if layer_prefix in key:
            parts = key.split(".")
            try:
                layer_idx_pos = parts.index("layers") + 1
                layer_idx = int(parts[layer_idx_pos])
                layer_keys.setdefault(layer_idx, []).append(key)
            except (ValueError, IndexError):
                fixed_keys.append(key)
        else:
            fixed_keys.append(key)

    print(f"Layers: {len(layer_keys)}, fixed tensors: {len(fixed_keys)}")

    for layer_idx in sorted(layer_keys.keys()):
        layer_dict = {}
        for key in layer_keys[layer_idx]:
            local_key = key.replace(f"{layer_prefix}.{layer_idx}.", "")
            layer_dict[local_key] = all_weights[key]
        out = output_dir / f"layer_{layer_idx}.safetensors"
        mx.save_safetensors(str(out), layer_dict)

    if fixed_keys:
        fixed_dict = {key: all_weights[key] for key in fixed_keys}
        mx.save_safetensors(str(output_dir / "fixed_weights.safetensors"), fixed_dict)

    print("Split complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Split a model into per-layer safetensors for streaming inference"
    )
    parser.add_argument("model_path", type=Path, help="Path to model.safetensors")
    parser.add_argument("output_dir", type=Path, help="Output directory")
    parser.add_argument(
        "--layer-prefix",
        default="model.layers",
        help="Layer key prefix in the checkpoint",
    )
    args = parser.parse_args()
    split_model_by_layers(args.model_path, args.output_dir, args.layer_prefix)


if __name__ == "__main__":
    main()