#!/usr/bin/env python3
"""Relabel existing Bermuda FEN|score|result data with Stockfish 19."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import chess
import chess.engine


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENGINE = (
    REPOSITORY_ROOT
    / "nnue-training"
    / "tools"
    / "stockfish-19"
    / "stockfish-windows-x86-64-universal.exe"
)
DEFAULT_INPUT = REPOSITORY_ROOT / "nnue-training" / "data" / "validation.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[tuple[int, str, chess.Board, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("|")
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected FEN | score | result")
            fen, _, result_text = (field.strip() for field in fields)
            try:
                board = chess.Board(fen)
                result = float(result_text)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: invalid FEN or result: {error}") from error
            if not board.is_valid():
                raise ValueError(f"{path}:{line_number}: invalid chess position")
            if not math.isfinite(result) or not 0.0 <= result <= 1.0:
                raise ValueError(f"{path}:{line_number}: result must be between 0 and 1")
            rows.append((line_number, fen, board, result_text))
    if not rows:
        raise ValueError(f"input dataset is empty: {path}")
    return rows


def choose_rows(
    rows: list[tuple[int, str, chess.Board, str]],
    max_positions: int | None,
    seed: int,
) -> list[tuple[int, str, chess.Board, str]]:
    if max_positions is None or max_positions >= len(rows):
        return rows
    indices = sorted(random.Random(seed).sample(range(len(rows)), max_positions))
    return [rows[index] for index in indices]


def relabel(args: argparse.Namespace) -> dict[str, object]:
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    engine_path = args.engine.resolve()
    if input_path == output_path:
        raise ValueError("input and output paths must differ; source data is never overwritten")
    if not input_path.is_file():
        raise FileNotFoundError(f"input dataset does not exist: {input_path}")
    if not engine_path.is_file():
        raise FileNotFoundError(f"Stockfish executable does not exist: {engine_path}")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"output already exists (use --force to replace): {output_path}")
    if args.max_positions is not None and args.max_positions < 1:
        raise ValueError("--max-positions must be positive")
    if args.threads < 1 or args.hash_mb < 1 or args.max_score_cp < 1:
        raise ValueError("--threads, --hash-mb, and --max-score-cp must be positive")

    rows = load_rows(input_path)
    selected = choose_rows(rows, args.max_positions, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    mate_scores = 0
    outlier_scores = 0
    labelled = 0
    started = time.perf_counter()
    engine_id: dict[str, str] = {}

    try:
        with chess.engine.SimpleEngine.popen_uci(str(engine_path), timeout=120.0) as engine:
            engine_id = {str(key): str(value) for key, value in engine.id.items()}
            engine_name = engine_id.get("name", "")
            if "stockfish 19" not in engine_name.casefold():
                raise RuntimeError(
                    f"expected Stockfish 19, engine identified itself as {engine_name!r}"
                )
            config = {"Threads": args.threads, "Hash": args.hash_mb}
            if "UCI_AnalyseMode" in engine.options:
                config["UCI_AnalyseMode"] = True
            engine.configure(config)

            with temporary_path.open("w", encoding="utf-8", newline="\n") as output:
                for index, (line_number, fen, board, result_text) in enumerate(selected, start=1):
                    info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                    score = info["score"].pov(chess.WHITE).score(mate_score=None)
                    if score is None:
                        mate_scores += 1
                    elif abs(score) > args.max_score_cp:
                        outlier_scores += 1
                    else:
                        output.write(f"{fen} | {score} | {result_text}\n")
                        labelled += 1

                    if index % 100 == 0 or index == len(selected):
                        output.flush()
                        elapsed = time.perf_counter() - started
                        print(
                            f"Analysed {index:,}/{len(selected):,}; labelled {labelled:,}; "
                            f"mates {mate_scores:,}; outliers {outlier_scores:,}; "
                            f"{index / max(elapsed, 1e-9):.1f} positions/s",
                            flush=True,
                        )

        if labelled == 0:
            raise RuntimeError("Stockfish produced no usable centipawn labels")
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    metadata = {
        "format": "FEN | Stockfish white-relative centipawns | original white-relative result",
        "stockfish": engine_id,
        "engine_path": str(engine_path),
        "engine_sha256": sha256_file(engine_path),
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "depth": args.depth,
        "threads": args.threads,
        "hash_mb": args.hash_mb,
        "seed": args.seed,
        "max_positions": args.max_positions,
        "input_positions": len(rows),
        "selected_positions": len(selected),
        "labelled_positions": labelled,
        "skipped_mate_scores": mate_scores,
        "skipped_outlier_scores": outlier_scores,
        "max_score_cp": args.max_score_cp,
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".json")
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {labelled:,} labels to {output_path}")
    print(f"Run metadata: {manifest_path}")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=(8, 10), default=8)
    parser.add_argument("--max-positions", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--max-score-cp", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=1908)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.max_positions is not None and args.max_positions < 1:
        parser.error("--max-positions must be positive")
    if args.threads < 1 or args.hash_mb < 1 or args.max_score_cp < 1:
        parser.error("--threads, --hash-mb, and --max-score-cp must be positive")
    return args


if __name__ == "__main__":
    relabel(parse_args())
