"""Streaming residual preparation and compact graph-dataset caching."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch
from torch.utils.data import Dataset

from .baseline import FrozenBaseline, sha256_file


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_NNUE_TRAINING_PATH = _REPOSITORY_ROOT / "nnue-training"
if str(_NNUE_TRAINING_PATH) not in sys.path:
    sys.path.insert(0, str(_NNUE_TRAINING_PATH))
import train_cpu


CACHE_SCHEMA_VERSION = 1
_PACKED_ARRAYS = (
    "piece_identity",
    "control_counts",
    "ep_square",
    "global_features",
    "attack_sources",
    "defence_sources",
    "white_king_square",
    "black_king_square",
    "residual_targets",
    "baseline_predictions",
    "original_targets",
)


def _line_count(path: Path) -> int:
    count = 0
    last_byte = b""
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    return count + int(bool(last_byte) and last_byte != b"\n")


def _cache_identity(
    data_path: Path,
    baseline: FrozenBaseline,
    split: str,
    max_positions: int | None,
) -> dict[str, Any]:
    identity = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "split": split,
        "source_path": str(data_path.resolve()),
        "source_sha256": sha256_file(data_path),
        "baseline_path": str(baseline.checkpoint_path),
        "baseline_sha256": baseline.checkpoint_id,
        "max_positions": max_positions,
        "eval_scale": baseline.eval_scale,
        "wdl": baseline.wdl,
    }
    signature = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    identity["cache_signature"] = signature
    return identity


def _packed_board(board: chess.Board) -> tuple[np.ndarray, ...]:
    piece_identity = np.zeros(64, dtype=np.uint8)
    control_counts = np.zeros((64, 2), dtype=np.uint8)
    attack_sources = np.zeros(64, dtype=np.uint64)
    defence_sources = np.zeros(64, dtype=np.uint64)

    for source in chess.SQUARES:
        piece = board.piece_at(source)
        if piece is None:
            continue

        piece_identity[source] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
        source_bit = np.uint64(1) << np.uint64(source)
        color_index = 0 if piece.color == chess.WHITE else 1
        for target in board.attacks(source):
            control_counts[target, color_index] += 1
            target_piece = board.piece_at(target)
            if target_piece is not None:
                if target_piece.color == piece.color:
                    defence_sources[target] |= source_bit
                else:
                    attack_sources[target] |= source_bit

    ep_square = -1 if board.ep_square is None else board.ep_square
    global_features = np.asarray(
        [
            1 if board.turn == chess.WHITE else -1,
            int(board.has_kingside_castling_rights(chess.WHITE)),
            int(board.has_queenside_castling_rights(chess.WHITE)),
            int(board.has_kingside_castling_rights(chess.BLACK)),
            int(board.has_queenside_castling_rights(chess.BLACK)),
        ],
        dtype=np.int8,
    )
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is None or black_king is None:
        raise ValueError("dataset position must contain both kings")

    return (
        piece_identity,
        control_counts,
        np.asarray(ep_square, dtype=np.int16),
        global_features,
        attack_sources,
        defence_sources,
        np.asarray(white_king, dtype=np.uint8),
        np.asarray(black_king, dtype=np.uint8),
    )


class ResidualGraphDataset(Dataset):
    """Memory-mapped compact graph inputs with residual and validation labels."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        with (cache_path / "metadata.json").open("r", encoding="utf-8") as source:
            self.metadata = json.load(source)
        self.length = int(self.metadata["num_samples"])
        if self.length < 1:
            raise ValueError(f"residual cache is empty: {cache_path}")
        self.arrays = {
            name: np.load(cache_path / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in _PACKED_ARRAYS
        }
        for name, array in self.arrays.items():
            if array.shape[0] < self.length:
                raise ValueError(f"residual cache array {name} is shorter than its metadata")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[np.ndarray, ...]:
        return tuple(np.array(self.arrays[name][index], copy=True) for name in _PACKED_ARRAYS)

    def close(self) -> None:
        for array in self.arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.arrays.clear()

    def __enter__(self) -> ResidualGraphDataset:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def collate_residual_batch(samples: list[tuple[np.ndarray, ...]]) -> dict[str, torch.Tensor]:
    packed = [np.stack(values, axis=0) for values in zip(*samples)]
    (
        piece_identity,
        control_counts,
        ep_square,
        global_features,
        attack_sources,
        defence_sources,
        white_king_square,
        black_king_square,
        residual_targets,
        baseline_predictions,
        original_targets,
    ) = packed

    batch_size = piece_identity.shape[0]
    batch_indices = np.arange(batch_size)[:, None]
    square_indices = np.arange(64)[None, :]
    square_features = np.zeros((batch_size, 64, 16), dtype=np.float32)
    square_features[batch_indices, square_indices, piece_identity] = 1.0
    square_features[:, :, 13] = control_counts[:, :, 0].astype(np.float32) / 8.0
    square_features[:, :, 14] = control_counts[:, :, 1].astype(np.float32) / 8.0
    ep_rows = np.flatnonzero(ep_square >= 0)
    square_features[ep_rows, ep_square[ep_rows], 15] = 1.0

    source_indices = np.arange(64, dtype=np.uint64)[None, None, :]
    attack_adj = ((attack_sources[:, :, None] >> source_indices) & 1).astype(np.float32)
    defence_adj = ((defence_sources[:, :, None] >> source_indices) & 1).astype(np.float32)

    return {
        "square_features": torch.from_numpy(square_features),
        "global_features": torch.from_numpy(global_features.astype(np.float32)),
        "attack_adj": torch.from_numpy(attack_adj),
        "defence_adj": torch.from_numpy(defence_adj),
        "white_king_square": torch.from_numpy(white_king_square.astype(np.int64)),
        "black_king_square": torch.from_numpy(black_king_square.astype(np.int64)),
        "residual_targets": torch.from_numpy(residual_targets.astype(np.float32)),
        "baseline_predictions": torch.from_numpy(baseline_predictions.astype(np.float32)),
        "original_targets": torch.from_numpy(original_targets.astype(np.float32)),
    }


def _make_memmaps(directory: Path, capacity: int) -> dict[str, np.memmap]:
    specifications = {
        "piece_identity": ((capacity, 64), np.uint8),
        "control_counts": ((capacity, 64, 2), np.uint8),
        "ep_square": ((capacity,), np.int16),
        "global_features": ((capacity, 5), np.int8),
        "attack_sources": ((capacity, 64), np.uint64),
        "defence_sources": ((capacity, 64), np.uint64),
        "white_king_square": ((capacity,), np.uint8),
        "black_king_square": ((capacity,), np.uint8),
        "residual_targets": ((capacity, 1), np.float32),
        "baseline_predictions": ((capacity, 1), np.float32),
        "original_targets": ((capacity, 1), np.float32),
    }
    return {
        name: np.lib.format.open_memmap(
            directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in specifications.items()
    }


def prepare_residual_cache(
    data_path: Path,
    cache_root: Path,
    split: str,
    baseline: FrozenBaseline,
    *,
    max_positions: int | None = None,
    preparation_batch_size: int = 1024,
) -> ResidualGraphDataset:
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")
    if not data_path.is_file():
        raise FileNotFoundError(f"dataset split not found: {data_path}")
    if max_positions is not None and max_positions < 1:
        raise ValueError("max_positions must be positive when provided")
    if preparation_batch_size < 1:
        raise ValueError("preparation_batch_size must be positive")

    identity = _cache_identity(data_path, baseline, split, max_positions)
    cache_path = cache_root / f"{split}-{identity['cache_signature'][:20]}"
    if cache_path.exists():
        metadata_path = cache_path / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"incomplete residual cache exists: {cache_path}")
        with metadata_path.open("r", encoding="utf-8") as source:
            cached_metadata = json.load(source)
        expected = {key: value for key, value in identity.items() if key != "cache_signature"}
        actual = {key: cached_metadata.get(key) for key in expected}
        if (
            actual != expected
            or cached_metadata.get("cache_signature") != identity["cache_signature"]
        ):
            raise ValueError(f"residual cache metadata does not match its identity: {cache_path}")
        return ResidualGraphDataset(cache_path)

    capacity = _line_count(data_path)
    if max_positions is not None:
        capacity = min(capacity, max_positions)
    if capacity < 1:
        raise ValueError(f"dataset split is empty: {data_path}")

    cache_root.mkdir(parents=True, exist_ok=True)
    staging_path = Path(tempfile.mkdtemp(prefix=f".{cache_path.name}.tmp-", dir=cache_root))
    arrays = _make_memmaps(staging_path, capacity)
    batch_rows: list[np.ndarray] = []
    batch_scores: list[float] = []
    batch_results: list[float] = []
    batch_boards: list[chess.Board] = []
    num_samples = 0

    def flush_batch() -> None:
        nonlocal num_samples
        if not batch_rows:
            return
        end = num_samples + len(batch_rows)
        if end > capacity:
            raise RuntimeError("parsed sample count exceeded cache capacity")

        baseline_features = torch.as_tensor(
            np.stack(batch_rows).astype(np.int64, copy=False),
            dtype=torch.long,
            device=baseline.feature_weights.device,
        )
        white_to_move_np = np.asarray(
            [board.turn == chess.WHITE for board in batch_boards], dtype=np.bool_
        )
        white_to_move = torch.as_tensor(
            white_to_move_np,
            dtype=torch.bool,
            device=baseline.feature_weights.device,
        )
        baseline_predictions = baseline.predict_white(baseline_features, white_to_move)
        target_stm = train_cpu.make_target(
            np.asarray(batch_scores, dtype=np.float32),
            np.asarray(batch_results, dtype=np.float32),
            baseline.eval_scale,
            baseline.wdl,
        )
        original_targets = np.where(
            white_to_move_np, target_stm, 1.0 - target_stm
        ).astype(np.float32)

        for row_index, board in enumerate(batch_boards):
            packed_board = _packed_board(board)
            for name, values in zip(_PACKED_ARRAYS[:8], packed_board):
                arrays[name][num_samples + row_index] = values
        predictions_np = baseline_predictions.cpu().numpy().astype(np.float32, copy=False)
        arrays["baseline_predictions"][num_samples:end] = predictions_np
        arrays["original_targets"][num_samples:end, 0] = original_targets
        arrays["residual_targets"][num_samples:end, 0] = original_targets - predictions_np[:, 0]
        num_samples = end
        batch_rows.clear()
        batch_scores.clear()
        batch_results.clear()
        batch_boards.clear()

    try:
        for row, score, result, board in train_cpu.iter_data_rows(data_path, max_positions):
            batch_rows.append(row)
            batch_scores.append(score)
            batch_results.append(result)
            batch_boards.append(board)
            if len(batch_rows) == preparation_batch_size:
                flush_batch()
        flush_batch()
        if num_samples == 0:
            raise ValueError(f"dataset split has no valid positions: {data_path}")

        metadata = {
            **identity,
            "num_samples": num_samples,
            "piece_identity_encoding": "0=empty, 1-6=white, 7-12=black",
            "control_counts_encoding": "integer counts, divided by 8 when decoded",
            "adjacency_encoding": "uint64 source bitboards per target",
        }
        for array in arrays.values():
            array.flush()
            array._mmap.close()
        arrays.clear()
        with (staging_path / "metadata.json").open("w", encoding="utf-8") as output:
            json.dump(metadata, output, sort_keys=True, indent=2)
        os.replace(staging_path, cache_path)
    except Exception:
        for array in arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    return ResidualGraphDataset(cache_path)