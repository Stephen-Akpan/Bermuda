#!/usr/bin/env python3
"""Merge annotated PGN files while preserving game records and comments."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import chess.pgn


def expand_inputs(patterns: list[str], output: Path) -> list[Path]:
    paths = []
    for pattern in patterns:
        matches = sorted(Path(path) for path in glob.glob(pattern))
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.extend(path for path in matches if path.resolve() != output.resolve())

    unique_paths = list(dict.fromkeys(paths))
    if not unique_paths:
        raise ValueError("No input PGN files matched")
    return unique_paths


def combine(inputs: list[Path], output_path: Path) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    game_count = 0
    error_count = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for input_path in inputs:
            file_games = 0
            with input_path.open("r", encoding="utf-8-sig", errors="replace") as source:
                while True:
                    game = chess.pgn.read_game(source)
                    if game is None:
                        break

                    file_games += 1
                    game_count += 1
                    error_count += len(game.errors)
                    exporter = chess.pgn.StringExporter(
                        headers=True,
                        variations=True,
                        comments=True,
                    )
                    output.write(game.accept(exporter))
                    output.write("\n\n")

            print(f"{input_path}: {file_games} games")

    print(f"Combined {game_count} games into {output_path}")
    print(f"PGN parse errors: {error_count}")
    return game_count, error_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pgn", nargs="+", required=True, help="input PGNs or glob patterns")
    parser.add_argument("--out", type=Path, default=Path("data/selfplay-combined.pgn"))
    args = parser.parse_args()

    try:
        inputs = expand_inputs(args.pgn, args.out)
        _, errors = combine(inputs, args.out)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())