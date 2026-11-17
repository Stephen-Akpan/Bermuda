#!/usr/bin/env python3
"""Validate the epoch-7 residual checkpoint and prepare an isolated match export."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "nnue-training"))

from neural.export.graph_export import export_checkpoint
from export_nnue import export_checkpoint as export_nnue_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(
    graph_checkpoint: Path,
    baseline_checkpoint: Path,
    baseline_nnue: Path,
    graph_export: Path,
) -> dict[str, object]:
    graph_checkpoint = graph_checkpoint.resolve()
    baseline_checkpoint = baseline_checkpoint.resolve()
    baseline_nnue = baseline_nnue.resolve()
    graph_export = graph_export.resolve()

    checkpoint = torch.load(graph_checkpoint, map_location="cpu", weights_only=False)
    if int(checkpoint.get("epoch", -1)) != 7:
        raise ValueError(f"expected best graph checkpoint from epoch 7, got {checkpoint.get('epoch')}")
    if float(checkpoint.get("residual_multiplier", float("nan"))) != 1.0:
        raise ValueError("epoch-7 checkpoint residual multiplier must be exactly 1.0")

    baseline_id = sha256_file(baseline_checkpoint)
    if checkpoint.get("baseline_checkpoint_id") != baseline_id:
        raise ValueError("graph checkpoint baseline identifier does not match the selected MLP checkpoint")

    with tempfile.TemporaryDirectory(prefix="bermuda-gnn-verify-") as temporary_directory:
        expected_nnue = Path(temporary_directory) / "baseline.nnue"
        with contextlib.redirect_stdout(io.StringIO()):
            export_nnue_checkpoint(baseline_checkpoint, expected_nnue)
        if expected_nnue.read_bytes() != baseline_nnue.read_bytes():
            raise ValueError("selected NNUE file is not a byte-identical export of the MLP checkpoint")

        expected_graph = Path(temporary_directory) / "candidate.bmgnn"
        export_checkpoint(graph_checkpoint, expected_graph)
        expected_graph_hash = sha256_file(expected_graph)

        graph_export.parent.mkdir(parents=True, exist_ok=True)
        if graph_export.exists():
            if sha256_file(graph_export) != expected_graph_hash:
                raise FileExistsError(
                    f"refusing to overwrite a different graph export: {graph_export}"
                )
        else:
            export_checkpoint(graph_checkpoint, graph_export)

    return {
        "graph_checkpoint_path": str(graph_checkpoint),
        "graph_checkpoint_sha256": sha256_file(graph_checkpoint),
        "graph_checkpoint_epoch": 7,
        "graph_export_path": str(graph_export),
        "graph_export_sha256": sha256_file(graph_export),
        "baseline_mlp_checkpoint_path": str(baseline_checkpoint),
        "baseline_mlp_checkpoint_sha256": baseline_id,
        "baseline_nnue_path": str(baseline_nnue),
        "baseline_nnue_sha256": sha256_file(baseline_nnue),
        "residual_multiplier": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-nnue", type=Path, required=True)
    parser.add_argument("--graph-export", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(prepare(
        arguments.graph_checkpoint,
        arguments.baseline_checkpoint,
        arguments.baseline_nnue,
        arguments.graph_export,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
