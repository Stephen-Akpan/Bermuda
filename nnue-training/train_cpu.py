#!/usr/bin/env python3
"""Train a small dual-perspective Chess768 evaluator on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import chess
import numpy as np
import psutil
from threadpoolctl import threadpool_limits


INPUTS = 768
PADDING_INDEX = INPUTS
MAX_PIECES = 32


def initialise_model(hidden_size: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feature_weights = rng.normal(0.0, 0.02, (INPUTS + 1, hidden_size)).astype(np.float32)
    feature_weights[PADDING_INDEX] = 0.0
    return {
        "feature_weights": feature_weights,
        "feature_bias": np.zeros(hidden_size, dtype=np.float32),
        "output_weights": rng.normal(0.0, 0.02, (2 * hidden_size,)).astype(np.float32),
        "output_bias": np.zeros(1, dtype=np.float32),
    }


def features_for(board: chess.Board, perspective: chess.Color) -> list[int]:
    features = []
    for square, piece in board.piece_map().items():
        relative_piece = piece.piece_type - 1
        if piece.color != perspective:
            relative_piece += 6
        relative_square = square if perspective == chess.WHITE else square ^ 56
        features.append(relative_piece * 64 + relative_square)
    return features


def load_data(path: Path, max_positions: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_rows = []
    score_rows = []
    result_rows = []
    errors = 0

    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if max_positions is not None and len(feature_rows) >= max_positions:
                break
            fields = line.rstrip("\r\n").split("|")
            if len(fields) != 3:
                errors += 1
                continue

            try:
                board = chess.Board(fields[0].strip())
                score_white = float(fields[1].strip())
                result_white = float(fields[2].strip())
                if not board.is_valid() or not 0.0 <= result_white <= 1.0:
                    raise ValueError("invalid position or result")
                if not abs(score_white) < 10_000:
                    raise ValueError("score outside accepted range")

                stm_features = features_for(board, board.turn)
                ntm_features = features_for(board, not board.turn)
                if len(stm_features) > MAX_PIECES or len(ntm_features) != len(stm_features):
                    raise ValueError("unsupported piece count")

                row = np.full((2, MAX_PIECES), PADDING_INDEX, dtype=np.uint16)
                row[0, :len(stm_features)] = stm_features
                row[1, :len(ntm_features)] = ntm_features

                if board.turn == chess.WHITE:
                    score_stm, result_stm = score_white, result_white
                else:
                    score_stm, result_stm = -score_white, 1.0 - result_white

                feature_rows.append(row)
                score_rows.append(score_stm)
                result_rows.append(result_stm)
            except (ValueError, IndexError):
                errors += 1

    if errors:
        print(f"Skipped malformed rows: {errors}")
    if not feature_rows:
        raise ValueError(f"No valid positions found in {path}")

    return (
        np.stack(feature_rows),
        np.asarray(score_rows, dtype=np.float32),
        np.asarray(result_rows, dtype=np.float32),
    )


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def make_target(scores: np.ndarray, results: np.ndarray, eval_scale: float, wdl: float) -> np.ndarray:
    score_target = sigmoid(scores / eval_scale)
    return wdl * results + (1.0 - wdl) * score_target


def forward(
    model: dict[str, np.ndarray], features: np.ndarray
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    feature_weights = model["feature_weights"]
    indices = features.astype(np.intp, copy=False)
    stm_acc = feature_weights[indices[:, 0]].sum(axis=1) + model["feature_bias"]
    ntm_acc = feature_weights[indices[:, 1]].sum(axis=1) + model["feature_bias"]
    stm_clip = np.clip(stm_acc, 0.0, 1.0)
    ntm_clip = np.clip(ntm_acc, 0.0, 1.0)
    hidden = np.concatenate((stm_clip * stm_clip, ntm_clip * ntm_clip), axis=1)
    logits = hidden @ model["output_weights"] + model["output_bias"][0]
    return logits, (indices, stm_acc, ntm_acc, stm_clip, ntm_clip, hidden)


def train_batch(
    model: dict[str, np.ndarray],
    moments: dict[str, dict[str, np.ndarray]],
    features: np.ndarray,
    scores: np.ndarray,
    results: np.ndarray,
    args: argparse.Namespace,
    step: int,
) -> float:
    logits, cache = forward(model, features)
    indices, stm_acc, ntm_acc, stm_clip, ntm_clip, hidden = cache
    target = make_target(scores, results, args.eval_scale, args.wdl)
    prediction = sigmoid(logits)
    error = prediction - target
    loss = float(np.mean(error * error))

    batch_size = len(features)
    dlogit = (2.0 / batch_size) * error * prediction * (1.0 - prediction)
    grad_output_weights = hidden.T @ dlogit
    grad_output_bias = np.asarray([dlogit.sum()], dtype=np.float32)
    grad_hidden = dlogit[:, None] * model["output_weights"]
    hidden_size = model["feature_bias"].shape[0]
    grad_stm = grad_hidden[:, :hidden_size] * 2.0 * stm_clip
    grad_ntm = grad_hidden[:, hidden_size:] * 2.0 * ntm_clip
    grad_stm *= (stm_acc > 0.0) & (stm_acc < 1.0)
    grad_ntm *= (ntm_acc > 0.0) & (ntm_acc < 1.0)

    grad_feature_weights = np.zeros_like(model["feature_weights"])
    np.add.at(
        grad_feature_weights,
        indices[:, 0].reshape(-1),
        np.broadcast_to(grad_stm[:, None, :], (batch_size, MAX_PIECES, hidden_size)).reshape(-1, hidden_size),
    )
    np.add.at(
        grad_feature_weights,
        indices[:, 1].reshape(-1),
        np.broadcast_to(grad_ntm[:, None, :], (batch_size, MAX_PIECES, hidden_size)).reshape(-1, hidden_size),
    )
    grad_feature_weights[PADDING_INDEX] = 0.0

    gradients = {
        "feature_weights": grad_feature_weights,
        "feature_bias": grad_stm.sum(axis=0) + grad_ntm.sum(axis=0),
        "output_weights": grad_output_weights,
        "output_bias": grad_output_bias,
    }
    beta1, beta2 = 0.9, 0.999
    for name, parameter in model.items():
        gradient = gradients[name]
        moments["first"][name] *= beta1
        moments["first"][name] += (1.0 - beta1) * gradient
        moments["second"][name] *= beta2
        moments["second"][name] += (1.0 - beta2) * gradient * gradient
        first = moments["first"][name] / (1.0 - beta1**step)
        second = moments["second"][name] / (1.0 - beta2**step)
        parameter -= args.learning_rate * (
            first / (np.sqrt(second) + 1e-8) + args.weight_decay * parameter
        )
    model["feature_weights"][PADDING_INDEX] = 0.0
    return loss


def evaluate_metrics(
    model: dict[str, np.ndarray],
    data: tuple[np.ndarray, np.ndarray, np.ndarray],
    batch_size: int,
    eval_scale: float,
    wdl: float,
) -> dict[str, float]:
    features, scores, results = data
    squared_errors = {"loss": 0.0, "search_loss": 0.0, "wdl_loss": 0.0}
    count = 0
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        logits, _ = forward(model, features[start:end])
        prediction = sigmoid(logits)
        search_target = sigmoid(scores[start:end] / eval_scale)
        wdl_target = results[start:end]
        target = wdl * wdl_target + (1.0 - wdl) * search_target
        squared_errors["loss"] += float(np.sum((prediction - target) ** 2))
        squared_errors["search_loss"] += float(np.sum((prediction - search_target) ** 2))
        squared_errors["wdl_loss"] += float(np.sum((prediction - wdl_target) ** 2))
        count += end - start

    metrics = {name: value / count for name, value in squared_errors.items()}
    metrics["rmse"] = metrics["loss"] ** 0.5
    metrics["search_rmse"] = metrics["search_loss"] ** 0.5
    metrics["wdl_rmse"] = metrics["wdl_loss"] ** 0.5
    return metrics


def train(args: argparse.Namespace) -> None:
    process = psutil.Process()
    peak_rss_bytes = process.memory_info().rss
    training = load_data(args.data, args.max_positions)
    validation = load_data(args.validation, args.max_positions) if args.validation else None
    peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)
    print(f"Training positions: {len(training[0]):,}")
    if validation:
        print(f"Validation positions: {len(validation[0]):,}")

    rng = np.random.default_rng(args.seed)
    model = initialise_model(args.hidden_size, args.seed)
    moments = {
        "first": {name: np.zeros_like(value) for name, value in model.items()},
        "second": {name: np.zeros_like(value) for name, value in model.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    observation_path = args.output / "training_observations.csv"
    observation_fields = [
        "epoch", "train_loss", "train_rmse", "train_search_loss", "train_search_rmse",
        "train_wdl_loss", "train_wdl_rmse", "validation_loss", "validation_rmse",
        "validation_search_loss", "validation_search_rmse", "validation_wdl_loss",
        "validation_wdl_rmse", "epoch_seconds", "training_seconds",
        "positions_per_second", "sampled_peak_rss_mb", "best_epoch",
        "best_validation_loss", "best_checkpoint",
    ]
    best_epoch = 0
    best_validation_loss = float("inf")
    best_checkpoint = args.output / "best.npz"
    step = 0

    with observation_path.open("w", newline="", encoding="utf-8") as observations_file:
        writer = csv.DictWriter(observations_file, fieldnames=observation_fields)
        writer.writeheader()
        observations_file.flush()

        for epoch in range(1, args.epochs + 1):
            epoch_started = time.perf_counter()
            order = rng.permutation(len(training[0]))
            training_started = time.perf_counter()

            for start in range(0, len(order), args.batch_size):
                indices = order[start:start + args.batch_size]
                step += 1
                train_batch(
                    model,
                    moments,
                    training[0][indices],
                    training[1][indices],
                    training[2][indices],
                    args,
                    step,
                )
                peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)

            training_seconds = time.perf_counter() - training_started
            train_metrics = evaluate_metrics(
                model, training, args.batch_size, args.eval_scale, args.wdl
            )
            peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)
            validation_metrics = evaluate_metrics(
                model, validation, args.batch_size, args.eval_scale, args.wdl
            ) if validation else {}
            peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)

            selection_loss = validation_metrics.get("loss", train_metrics["loss"])
            if selection_loss < best_validation_loss:
                best_validation_loss = selection_loss
                best_epoch = epoch

            metadata = {name: str(value) for name, value in vars(args).items()}
            metadata.update({"epoch": str(epoch), "best_epoch": str(best_epoch)})
            epoch_checkpoint = args.output / f"epoch-{epoch:03d}.npz"
            np.savez_compressed(
                epoch_checkpoint,
                **model,
                metadata=np.asarray(json.dumps(metadata)),
            )
            if best_epoch == epoch:
                np.savez_compressed(
                    best_checkpoint,
                    **model,
                    metadata=np.asarray(json.dumps(metadata)),
                )

            epoch_seconds = time.perf_counter() - epoch_started
            row = {
                "epoch": epoch,
                **{f"train_{name}": value for name, value in train_metrics.items()},
                **{f"validation_{name}": value for name, value in validation_metrics.items()},
                "epoch_seconds": epoch_seconds,
                "training_seconds": training_seconds,
                "positions_per_second": len(training[0]) / max(training_seconds, 1e-9),
                "sampled_peak_rss_mb": peak_rss_bytes / (1024 * 1024),
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "best_checkpoint": str(best_checkpoint),
            }
            writer.writerow(row)
            observations_file.flush()
            print(json.dumps(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path, default=Path("checkpoints-cpu"))
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--eval-scale", type=float, default=400.0)
    parser.add_argument("--wdl", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--max-positions", type=int)
    args = parser.parse_args()

    if args.hidden_size < 1 or args.epochs < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("hidden size, epochs, batch size, and threads must be positive")
    if args.eval_scale <= 0 or not 0.0 <= args.wdl <= 1.0:
        parser.error("eval scale must be positive and WDL blend must be in [0, 1]")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        parser.error("learning rate must be positive and weight decay non-negative")
    if args.max_positions is not None and args.max_positions < 1:
        parser.error("--max-positions must be positive")
    if not args.data.is_file():
        parser.error(f"training data file does not exist: {args.data}")
    if args.validation is not None and not args.validation.is_file():
        parser.error(f"validation data file does not exist: {args.validation}")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    with threadpool_limits(limits=arguments.threads):
        train(arguments)