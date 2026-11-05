#!/usr/bin/env python3
"""Convert Cutechess annotated PGNs into Bullet text training data."""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys
from pathlib import Path

import chess
import chess.pgn


SCORE_PATTERN = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:/(\d+))?(?:\s|$)"
)
MATE_PATTERN = re.compile(r"(?:^|\s)[+-]?M\d+(?:/\d+)?(?:\s|$)", re.IGNORECASE)
RESULTS = {"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0}


def parse_score(comment: str, min_depth: int, max_score_cp: int) -> int | None:
    if MATE_PATTERN.search(comment):
        return None

    match = SCORE_PATTERN.search(comment)
    if match is None:
        return None

    if match.group(2) is not None and int(match.group(2)) < min_depth:
        return None

    score_cp = round(float(match.group(1)) * 100)
    if abs(score_cp) > max_score_cp:
        return None
    return score_cp


def validation_group(game: chess.pgn.Game, split_ply: int) -> str:
    board = game.board()
    for ply, node in enumerate(game.mainline()):
        if ply >= split_ply:
            break
        board.push(node.move)
    return board.fen(en_passant="fen")


def choose_validation_groups(groups: list[str], seed: int, percent: int) -> set[str]:
    if percent == 0 or len(groups) < 2:
        return set()

    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest(),
    )
    count = min(len(groups) - 1, max(1, (len(groups) * percent + 99) // 100))
    return set(ordered[:count])


def extract_game(
    game: chess.pgn.Game,
    min_ply: int,
    min_depth: int,
    max_score_cp: int,
    exclude_checks: bool,
) -> tuple[list[str], int]:
    result = RESULTS.get(game.headers.get("Result", "*"))
    if result is None:
        return [], 0

    board = game.board()
    rows = []
    missing_or_filtered = 0

    for ply, node in enumerate(game.mainline()):
        if ply >= min_ply and (not exclude_checks or not board.is_check()):
            score_cp = parse_score(node.comment, min_depth, max_score_cp)
            if score_cp is not None:
                white_score_cp = score_cp if board.turn == chess.WHITE else -score_cp
                fen = board.fen(en_passant="fen")
                rows.append(f"{fen} | {white_score_cp} | {result:.1f}")
            else:
                missing_or_filtered += 1
        board.push(node.move)

    return rows, missing_or_filtered


def write_rows(path: Path, rows: list[str], rng: random.Random) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng.shuffle(rows)
    with path.open("w", encoding="ascii", newline="\n") as output:
        for row in rows:
            output.write(row + "\n")


def generate(args: argparse.Namespace) -> int:
    rows_by_group: dict[str, list[str]] = {}
    games = invalid_games = filtered_positions = 0

    for pgn_path in args.pgn:
        with pgn_path.open("r", encoding="utf-8-sig", errors="replace") as source:
            while True:
                game = chess.pgn.read_game(source)
                if game is None:
                    break
                games += 1
                if game.errors:
                    invalid_games += 1
                    continue

                group = validation_group(game, args.split_ply)
                rows, filtered = extract_game(
                    game,
                    args.min_ply,
                    args.min_depth,
                    args.max_score_cp,
                    not args.include_checks,
                )
                filtered_positions += filtered
                rows_by_group.setdefault(group, []).extend(rows)

    validation_groups = choose_validation_groups(
        list(rows_by_group), args.seed, args.validation_percent
    )
    train_rows = []
    validation_rows = []
    for group, rows in rows_by_group.items():
        (validation_rows if group in validation_groups else train_rows).extend(rows)

    rng = random.Random(args.seed)
    write_rows(args.train_out, train_rows, rng)
    write_rows(args.validation_out, validation_rows, rng)

    print(f"Games read: {games}; invalid games skipped: {invalid_games}")
    print(f"Positions without an accepted score: {filtered_positions}")
    print(f"Training positions: {len(train_rows)} -> {args.train_out}")
    print(f"Validation positions: {len(validation_rows)} -> {args.validation_out}")

    if not train_rows:
        print("No training positions were produced.", file=sys.stderr)
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", type=Path, nargs="+", required=True)
    parser.add_argument("--train-out", type=Path, default=Path("data/train.txt"))
    parser.add_argument("--validation-out", type=Path, default=Path("data/validation.txt"))
    parser.add_argument("--min-ply", type=int, default=16)
    parser.add_argument("--split-ply", type=int, default=16)
    parser.add_argument("--min-depth", type=int, default=4)
    parser.add_argument("--max-score-cp", type=int, default=2000)
    parser.add_argument("--validation-percent", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--include-checks", action="store_true")
    args = parser.parse_args()

    if args.min_ply < 0 or args.split_ply < 0:
        parser.error("--min-ply and --split-ply must be non-negative")
    if args.min_depth < 0 or args.max_score_cp < 0:
        parser.error("--min-depth and --max-score-cp must be non-negative")
    if not 0 <= args.validation_percent < 100:
        parser.error("--validation-percent must be between 0 and 99")
    for path in args.pgn:
        if not path.is_file():
            parser.error(f"PGN file does not exist: {path}")
    return args


if __name__ == "__main__":
    raise SystemExit(generate(parse_args()))