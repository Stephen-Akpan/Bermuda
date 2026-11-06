import sys
import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_openings


class OpeningGenerationTests(unittest.TestCase):
    def test_seeded_openings_are_legal_unique_and_reproducible(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.pgn"
            second = Path(temp_dir) / "second.pgn"
            generate_openings.generate_openings(8, 12, 123, first)
            generate_openings.generate_openings(8, 12, 123, second)

            self.assertEqual(first.read_text(encoding="utf-8"), second.read_text(encoding="utf-8"))
            games = []
            with first.open(encoding="utf-8") as source:
                while (game := chess.pgn.read_game(source)) is not None:
                    self.assertFalse(game.errors)
                    games.append(game)
            self.assertEqual(len(games), 8)
            endings = [game.end().board().fen(en_passant="fen") for game in games]
            self.assertEqual(len(set(endings)), len(endings))

    def test_zero_ply_openings_do_not_require_unique_positions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "startpos.pgn"
            generate_openings.generate_openings(3, 0, 5, output)

            with output.open(encoding="utf-8") as source:
                games = list(iter(lambda: chess.pgn.read_game(source), None))
            self.assertEqual(len(games), 3)


if __name__ == "__main__":
    unittest.main()