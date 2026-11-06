#!/usr/bin/env python3
"""Export a NumPy training checkpoint to Bermuda's compact float network file."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


MAGIC = b"BMNNUE1\0"
INPUTS = 768
SUPPORTED_HIDDEN_SIZE = 64
HEADER = struct.Struct("<8sIIf")


def export_checkpoint(checkpoint_path: Path, output_path: Path) -> None:
    with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
        feature_weights = np.asarray(checkpoint["feature_weights"], dtype="<f4")
        feature_bias = np.asarray(checkpoint["feature_bias"], dtype="<f4")
        output_weights = np.asarray(checkpoint["output_weights"], dtype="<f4")
        output_bias = np.asarray(checkpoint["output_bias"], dtype="<f4")
        metadata = json.loads(str(checkpoint["metadata"].item()))

    if feature_weights.ndim != 2 or feature_weights.shape[0] != INPUTS + 1:
        raise ValueError("checkpoint feature weights must have shape (769, hidden)")
    hidden_size = feature_weights.shape[1]
    if hidden_size < 1:
        raise ValueError("checkpoint hidden size must be positive")
    if hidden_size != SUPPORTED_HIDDEN_SIZE:
        raise ValueError(
            f"this Bermuda build supports hidden size {SUPPORTED_HIDDEN_SIZE}, got {hidden_size}"
        )
    if feature_bias.shape != (hidden_size,):
        raise ValueError("checkpoint feature bias has the wrong shape")
    if output_weights.shape != (2 * hidden_size,):
        raise ValueError("checkpoint output weights have the wrong shape")
    if output_bias.shape != (1,):
        raise ValueError("checkpoint output bias must contain one value")

    arrays = (feature_weights[:INPUTS], feature_bias, output_weights, output_bias)
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("checkpoint contains non-finite weights")
    if np.any(feature_weights[INPUTS] != 0):
        raise ValueError("checkpoint padding feature must be zero")

    eval_scale = float(metadata.get("eval_scale", 400.0))
    if not np.isfinite(eval_scale) or eval_scale <= 0:
        raise ValueError("checkpoint evaluation scale must be positive and finite")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        output.write(HEADER.pack(MAGIC, INPUTS, hidden_size, eval_scale))
        for array in arrays:
            output.write(np.ascontiguousarray(array, dtype="<f4").tobytes())

    print(f"Exported {hidden_size}-wide NNUE to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("bermuda.nnue"))
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    export_checkpoint(args.checkpoint, args.output)


if __name__ == "__main__":
    main()