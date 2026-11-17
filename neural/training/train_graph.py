"""Train Bermuda's graph evaluator against cached frozen-baseline residuals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from neural.models.graph_evaluator import GraphEvaluator
from .baseline import FrozenBaseline, sha256_file
from .residual_data import ResidualGraphDataset, collate_residual_batch, prepare_residual_cache


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = REPOSITORY_ROOT / "nnue-training" / "checkpoints-cpu" / "best.npz"
DEFAULT_TRAIN_DATA = REPOSITORY_ROOT / "nnue-training" / "data" / "train.txt"
DEFAULT_VALIDATION_DATA = REPOSITORY_ROOT / "nnue-training" / "data" / "validation.txt"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "neural" / "checkpoints"
DEFAULT_CACHE = REPOSITORY_ROOT / "neural" / "cache" / "residual"

MODEL_CONFIG = {
    "square_feature_count": 16,
    "global_feature_count": 5,
    "square_count": 64,
    "hidden_size": 32,
    "message_passing_layers": 1,
    "pooled_size": 133,
    "value_hidden_size": 64,
}

HISTORY_FIELDS = (
    "epoch",
    "train_residual_loss",
    "baseline_mae",
    "baseline_rmse",
    "combined_mae",
    "combined_rmse",
    "combined_pearson",
    "residual_loss",
    "residual_pearson",
    "predicted_residual_mean",
    "predicted_residual_std",
    "actual_residual_mean",
    "actual_residual_std",
    "best_combined_mae",
)


def _pearson_correlation(prediction: torch.Tensor, target: torch.Tensor) -> float:
    prediction = prediction.reshape(-1).double()
    target = target.reshape(-1).double()
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    denominator = torch.linalg.vector_norm(prediction) * torch.linalg.vector_norm(target)
    if denominator.item() == 0.0:
        return 0.0
    return float(torch.dot(prediction, target).div(denominator).item())


def _move_graph_inputs(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: batch[name].to(device, non_blocking=True)
        for name in (
            "square_features",
            "global_features",
            "attack_adj",
            "defence_adj",
            "white_king_square",
            "black_king_square",
        )
    }


def _evaluate(
    model: GraphEvaluator,
    validation_loader: DataLoader,
    device: torch.device,
    residual_multiplier: float,
) -> dict[str, float]:
    model.eval()
    residual_predictions = []
    residual_targets = []
    baseline_predictions = []
    original_targets = []
    smooth_l1_sum = 0.0
    count = 0

    with torch.inference_mode():
        for batch in validation_loader:
            inputs = _move_graph_inputs(batch, device)
            actual_residual = batch["residual_targets"].to(device)
            residual_prediction = model(**inputs) * residual_multiplier
            smooth_l1_sum += float(
                nn.functional.smooth_l1_loss(
                    residual_prediction, actual_residual, reduction="sum"
                ).item()
            )
            count += actual_residual.shape[0]
            residual_predictions.append(residual_prediction.cpu())
            residual_targets.append(actual_residual.cpu())
            baseline_predictions.append(batch["baseline_predictions"])
            original_targets.append(batch["original_targets"])

    if count == 0:
        raise ValueError("validation dataset is empty")

    residual_prediction = torch.cat(residual_predictions)
    residual_target = torch.cat(residual_targets)
    baseline_prediction = torch.cat(baseline_predictions)
    original_target = torch.cat(original_targets)
    combined_prediction = baseline_prediction + residual_prediction
    combined_error = combined_prediction - original_target
    baseline_error = baseline_prediction - original_target

    return {
        "baseline_mae": float(baseline_error.abs().mean().item()),
        "baseline_rmse": float(baseline_error.square().mean().sqrt().item()),
        "combined_mae": float(combined_error.abs().mean().item()),
        "combined_rmse": float(combined_error.square().mean().sqrt().item()),
        "combined_pearson": _pearson_correlation(combined_prediction, original_target),
        "residual_loss": smooth_l1_sum / count,
        "residual_pearson": _pearson_correlation(residual_prediction, residual_target),
        "predicted_residual_mean": float(residual_prediction.mean().item()),
        "predicted_residual_std": float(residual_prediction.std(unbiased=False).item()),
        "actual_residual_mean": float(residual_target.mean().item()),
        "actual_residual_std": float(residual_target.std(unbiased=False).item()),
    }


def _atomic_save_checkpoint(checkpoint: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary_path)
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _checkpoint_payload(
    model: GraphEvaluator,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_combined_mae: float,
    baseline: FrozenBaseline,
    residual_multiplier: float,
    train_dataset: ResidualGraphDataset,
    validation_dataset: ResidualGraphDataset,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_config": MODEL_CONFIG,
        "graph_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_validation_metric": {"combined_mae": best_combined_mae},
        "baseline_checkpoint_path": str(baseline.checkpoint_path),
        "baseline_checkpoint_id": baseline.checkpoint_id,
        "residual_multiplier": residual_multiplier,
        "train_cache_signature": train_dataset.metadata["cache_signature"],
        "validation_cache_signature": validation_dataset.metadata["cache_signature"],
        "training_config": config,
    }


def _load_resume(
    path: Path,
    model: GraphEvaluator,
    optimizer: torch.optim.Optimizer,
    baseline: FrozenBaseline,
    residual_multiplier: float,
    train_dataset: ResidualGraphDataset,
    validation_dataset: ResidualGraphDataset,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_config") != MODEL_CONFIG:
        raise ValueError("resume checkpoint model architecture is incompatible")
    if checkpoint.get("baseline_checkpoint_id") != baseline.checkpoint_id:
        raise ValueError("resume checkpoint was trained against a different baseline")
    if checkpoint.get("residual_multiplier") != residual_multiplier:
        raise ValueError("resume checkpoint residual multiplier does not match")
    if checkpoint.get("train_cache_signature") != train_dataset.metadata["cache_signature"]:
        raise ValueError("resume checkpoint training dataset does not match")
    if checkpoint.get("validation_cache_signature") != validation_dataset.metadata["cache_signature"]:
        raise ValueError("resume checkpoint validation dataset does not match")
    model.load_state_dict(checkpoint["graph_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_validation_metric"]["combined_mae"]),
    )


def train(args: argparse.Namespace) -> Path:
    if args.train_data.resolve() == args.validation_data.resolve():
        raise ValueError("training and validation data must be different files")
    if sha256_file(args.train_data) == sha256_file(args.validation_data):
        raise ValueError("training and validation data files have identical contents")
    if args.epochs < 1 or args.batch_size < 1 or args.preparation_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning rate must be positive and weight decay non-negative")
    if args.residual_multiplier <= 0.0:
        raise ValueError("residual multiplier must be positive")
    if args.gradient_clip_norm <= 0.0:
        raise ValueError("gradient clip norm must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    baseline = FrozenBaseline(args.baseline_checkpoint, device=device)
    for parameter in baseline.parameters():
        parameter.requires_grad_(False)
    baseline.eval()

    train_dataset = prepare_residual_cache(
        args.train_data,
        args.cache_dir,
        "train",
        baseline,
        max_positions=args.max_positions,
        preparation_batch_size=args.preparation_batch_size,
    )
    validation_dataset = prepare_residual_cache(
        args.validation_data,
        args.cache_dir,
        "validation",
        baseline,
        max_positions=args.max_positions,
        preparation_batch_size=args.preparation_batch_size,
    )

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_residual_batch,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_residual_batch,
        num_workers=0,
    )

    model = GraphEvaluator().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    config = {
        "optimizer": "AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "loss": "SmoothL1Loss",
        "gradient_clip_norm": args.gradient_clip_norm,
        "residual_multiplier": args.residual_multiplier,
        "seed": args.seed,
        "device": str(device),
        "max_positions": args.max_positions,
    }

    start_epoch = 1
    best_combined_mae = math.inf
    if args.resume is not None:
        start_epoch, best_combined_mae = _load_resume(
            args.resume,
            model,
            optimizer,
            baseline,
            args.residual_multiplier,
            train_dataset,
            validation_dataset,
            device,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "training_history.csv"
    history_mode = "a" if args.resume is not None and history_path.is_file() else "w"
    with history_path.open(history_mode, newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
        if history_mode == "w":
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            weighted_loss = 0.0
            sample_count = 0
            for batch in train_loader:
                inputs = _move_graph_inputs(batch, device)
                residual_targets = batch["residual_targets"].to(device)
                optimizer.zero_grad(set_to_none=True)
                graph_prediction = model(**inputs)
                residual_prediction = graph_prediction * args.residual_multiplier
                loss = nn.functional.smooth_l1_loss(residual_prediction, residual_targets)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                optimizer.step()
                current_batch_size = residual_targets.shape[0]
                weighted_loss += float(loss.detach().item()) * current_batch_size
                sample_count += current_batch_size

            if sample_count == 0:
                raise ValueError("training dataset is empty")
            train_residual_loss = weighted_loss / sample_count
            validation_metrics = _evaluate(
                model, validation_loader, device, args.residual_multiplier
            )
            is_best = validation_metrics["combined_mae"] < best_combined_mae
            best_combined_mae = min(best_combined_mae, validation_metrics["combined_mae"])
            row = {
                "epoch": epoch,
                "train_residual_loss": train_residual_loss,
                **validation_metrics,
                "best_combined_mae": best_combined_mae,
            }
            writer.writerow(row)
            history_file.flush()
            print(json.dumps(row))

            payload = _checkpoint_payload(
                model,
                optimizer,
                epoch,
                best_combined_mae,
                baseline,
                args.residual_multiplier,
                train_dataset,
                validation_dataset,
                config,
            )
            _atomic_save_checkpoint(payload, args.output_dir / "last.pt")
            if is_best:
                _atomic_save_checkpoint(payload, args.output_dir / "best.pt")

    return history_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--validation-data", type=Path, default=DEFAULT_VALIDATION_DATA)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--preparation-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--residual-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--resume", type=Path)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    for name in ("train_data", "validation_data", "baseline_checkpoint"):
        path = getattr(args, name)
        if not path.is_file():
            raise SystemExit(f"{name.replace('_', '-')} does not exist: {path}")
    if args.resume is not None and not args.resume.is_file():
        raise SystemExit(f"resume checkpoint does not exist: {args.resume}")
    train(args)


if __name__ == "__main__":
    main()