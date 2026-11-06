import io
import sys
import unittest
from pathlib import Path

import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import generate_cutechess_dataset as dataset


class CutechessDatasetTests(unittest.TestCase):
    def test_scores_are_white_relative_and_tied_to_pre_move_positions(self):
        pgn = """[Event "score perspective"]
[Result "1-0"]

1. e4 {+0.57/8 0.22s} e5 {-0.36/7 0.22s} 2. Nf3 {+0.42/8} Nc6 {-0.31/8} 1-0
"""
        game = chess.pgn.read_game(io.StringIO(pgn))

        rows, _ = dataset.extract_game(
            game,
            min_ply=0,
            min_depth=4,
            max_score_cp=2000,
            exclude_checks=False,
        )

        self.assertEqual([row.split(" | ")[1:] for row in rows], [
            ["57", "1.0"],
            ["36", "1.0"],
            ["42", "1.0"],
            ["31", "1.0"],
        ])
        self.assertIn(" w ", rows[0].split(" | ")[0])
        self.assertIn(" b ", rows[1].split(" | ")[0])

    def test_mate_and_shallow_scores_are_rejected(self):
        self.assertIsNone(dataset.parse_score("-M5/5 0.05s", 1, 2000))
        self.assertIsNone(dataset.parse_score("+0.57/3 0.05s", 4, 2000))
        self.assertEqual(dataset.parse_score("+0.57/8 0.22s", 4, 2000), 57)

    def test_small_split_holds_out_at_least_one_opening(self):
        groups = [f"opening-{index}" for index in range(10)]

        validation = dataset.choose_validation_groups(groups, seed=7, percent=10)

        self.assertEqual(len(validation), 1)
        self.assertEqual(validation, dataset.choose_validation_groups(groups, 7, 10))
        self.assertEqual(dataset.choose_validation_groups(groups, 7, 0), set())


if __name__ == "__main__":
    unittest.main()