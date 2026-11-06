#!/usr/bin/env python3
"""Generate a reproducible PGN suite of random legal chess openings."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import chess
import chess.pgn


def generate_openings(count: int, plies: int, seed: int, output_path: Path) -> None:
    rng = random.Random(seed)
    games = []

    while len(games) < count:
        board = chess.Board()
        game = chess.pgn.Game()
        node = game
        for _ in range(plies):
            if board.is_game_over():
                break
            move = rng.choice(list(board.legal_moves))
            board.push(move)
            node = node.add_variation(move)

        game.headers["Event"] = "Bermuda NNUE self-play opening"
        game.headers["Site"] = "GitHub Actions"
        game.headers["Round"] = str(len(games) + 1)
        game.headers["White"] = "Random opening"
        game.headers["Black"] = "Random opening"
        game.headers["Result"] = "*"
        games.append(game)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for game in games:
            exporter = chess.pgn.StringExporter(
                headers=True,
                variations=False,
                comments=False,
            )
            output.write(game.accept(exporter))
            output.write("\n\n")
    print(f"Wrote {len(games)} unique opening positions at {plies} plies to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--plies", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.plies < 0:
        parser.error("count must be positive and plies must be non-negative")
    generate_openings(args.count, args.plies, args.seed, args.out)


if __name__ == "__main__":
    main()