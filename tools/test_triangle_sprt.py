import json
import tempfile
import unittest
from pathlib import Path

from triangle_sprt import (
    analyze_fixed_matches,
    analyze_matches,
    calculate_llr,
    read_pair,
    split_pgn,
    sprt_boundaries,
    update_sprt,
    write_fixed_report,
)


def write_pair(path, results):
    games = []
    for index, result in enumerate(results):
        white = "Candidate" if index == 0 else "Baseline"
        black = "Baseline" if index == 0 else "Candidate"
        games.append(
            f'[Event "Test"]\n[White "{white}"]\n[Black "{black}"]\n'
            f'[Result "{result}"]\n\n1. e4 e5 {result}\n'
        )
    path.write_text("\n".join(games), encoding="utf-8")


class TriangleSprtTests(unittest.TestCase):
    def test_fishtest_default_boundaries(self):
        lower, upper = sprt_boundaries()
        self.assertAlmostEqual(lower, -2.9444389791664403)
        self.assertAlmostEqual(upper, 2.9444389791664403)

    def test_fishtest_llr_reference_fixture(self):
        self.assertAlmostEqual(calculate_llr([1, 2, 3, 4, 5]), 0.1413399419497301)

    def test_pentanomial_categories_and_reversed_colors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair-00001.pgn"
            write_pair(path, ["1-0", "1/2-1/2"])
            self.assertEqual(read_pair(path), 3)

    def test_same_color_assignment_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pair-00001.pgn"
            text = (
                '[White "Candidate"]\n[Black "Baseline"]\n[Result "1-0"]\n\n'
                '1. e4 e5 1-0\n\n'
                '[White "Candidate"]\n[Black "Baseline"]\n[Result "1-0"]\n\n'
                '1. d4 d5 1-0\n'
            )
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "did not switch colors"):
                read_pair(path)

    def test_batch_pgn_is_split_only_after_pair_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch.pgn"
            write_pair(batch, ["1-0", "0-1"])
            text = batch.read_text(encoding="utf-8")
            batch.write_text(text + "\n" + text, encoding="utf-8")
            output = root / "matches"
            split_pgn(batch, output, first_pair=4, expected_pairs=2)
            self.assertEqual(read_pair(output / "pair-00004.pgn"), 4)
            self.assertEqual(read_pair(output / "pair-00005.pgn"), 4)

            malformed = root / "malformed.pgn"
            malformed.write_text(text + "\n" + text.replace(
                '[White "Baseline"]', '[White "Candidate"]'
            ), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must contain"):
                split_pgn(malformed, output, first_pair=6, expected_pairs=2)
            self.assertFalse((output / "pair-00006.pgn").exists())

    def test_batch_rejects_reversed_color_pair_with_different_opening_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = root / "batch.pgn"
            batch.write_text(
                '[White "Candidate"]\n[Black "Baseline"]\n[Result "1-0"]\n\n'
                '1. e4 e5 2. Nf3 Nc6 1-0\n\n'
                '[White "Baseline"]\n[Black "Candidate"]\n[Result "0-1"]\n\n'
                '1. d4 d5 2. c4 e6 0-1\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "opening prefix"):
                split_pgn(batch, root / "matches", first_pair=1,
                          expected_pairs=1, opening_plies=4)
            self.assertFalse((root / "matches" / "pair-00001.pgn").exists())

    def test_sprt_decisions_and_dynamic_overshoot(self):
        counts = [0, 0, 0, 0, 0]
        overshoot = {
            "last_update": 0,
            "ref0": 0.0,
            "m0": 0.0,
            "sq0": 0.0,
            "ref1": 0.0,
            "m1": 0.0,
            "sq1": 0.0,
        }
        first = update_sprt(counts, overshoot)
        self.assertEqual(first["status"], "CONTINUE")
        for _ in range(206):
            counts[4] += 1
            result = update_sprt(counts, overshoot)
            if result["status"] != "CONTINUE":
                break
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(overshoot["last_update"], 206)
        self.assertLess(
            result["adjusted_upper_boundary"], result["upper_boundary"]
        )

        counts = [0, 0, 0, 0, 0]
        overshoot = {
            "last_update": 0,
            "ref0": 0.0,
            "m0": 0.0,
            "sq0": 0.0,
            "ref1": 0.0,
            "m1": 0.0,
            "sq1": 0.0,
        }
        for _ in range(203):
            counts[0] += 1
            result = update_sprt(counts, overshoot)
            if result["status"] != "CONTINUE":
                break
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(overshoot["last_update"], 203)

    def test_maximum_without_boundary_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            matches = Path(directory)
            write_pair(matches / "pair-00001.pgn", ["1/2-1/2", "1/2-1/2"])
            result = analyze_matches(matches, completed_pairs=1, max_pairs=1)
            self.assertEqual(result["status"], "INCONCLUSIVE")
            self.assertEqual(result["completed_games"], 2)
            self.assertEqual(result["pentanomial_counts"], [0, 0, 1, 0, 0])

    def test_incremental_updates_match_full_rebuild(self):
        pair_results = [
            ["1-0", "0-1"],
            ["0-1", "1-0"],
            ["1/2-1/2", "1/2-1/2"],
            ["1-0", "1-0"],
            ["1/2-1/2", "0-1"],
            ["0-1", "1/2-1/2"],
            ["1-0", "0-1"],
            ["1/2-1/2", "1-0"],
        ]
        with tempfile.TemporaryDirectory() as directory:
            matches = Path(directory)
            for pair_number, results in enumerate(pair_results, start=1):
                write_pair(matches / f"pair-{pair_number:05d}.pgn", results)

            full = analyze_matches(matches, completed_pairs=8, max_pairs=10)
            previous = analyze_matches(matches, completed_pairs=0, max_pairs=10)
            for pair_number in range(1, 9):
                previous = analyze_matches(
                    matches,
                    completed_pairs=pair_number,
                    max_pairs=10,
                    previous_result=json.loads(json.dumps(previous)),
                )

            self.assertEqual(previous["pentanomial_counts"], full["pentanomial_counts"])
            self.assertEqual(previous["overshoot_state"], full["overshoot_state"])
            self.assertAlmostEqual(previous["llr"], full["llr"])
            self.assertEqual(previous["status"], full["status"])

    def test_analysis_json_shape_is_serializable(self):
        with tempfile.TemporaryDirectory() as directory:
            matches = Path(directory)
            write_pair(matches / "pair-00001.pgn", ["1/2-1/2", "1/2-1/2"])
            result = analyze_matches(matches, completed_pairs=1, max_pairs=1)
            decoded = json.loads(json.dumps(result))
            self.assertEqual(decoded["status"], "INCONCLUSIVE")

    def test_fixed_report_uses_pair_level_score_and_writes_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matches = root / "matches"
            matches.mkdir()
            for pair_number in range(1, 21):
                result_cycle = pair_number % 4
                if result_cycle == 0:
                    results = ["1-0", "0-1"]
                elif result_cycle == 1:
                    results = ["1/2-1/2", "1/2-1/2"]
                elif result_cycle == 2:
                    results = ["1-0", "1/2-1/2"]
                else:
                    results = ["0-1", "1-0"]
                write_pair(matches / f"pair-{pair_number:05d}.pgn", results)

            report = analyze_fixed_matches(matches, 20, {"seed": 14})
            self.assertEqual(report["completed_games"], 40)
            self.assertEqual(report["candidate_wins"], 15)
            self.assertEqual(report["candidate_draws"], 15)
            self.assertEqual(report["candidate_losses"], 10)
            self.assertAlmostEqual(report["score"], 0.5625)
            self.assertAlmostEqual(report["elo_difference"], 43.66, places=1)
            self.assertIsNotNone(report["elo_uncertainty_95"])
            self.assertEqual(report["average_game_length_plies"], 2.0)
            self.assertIsNone(report["average_nodes_per_game"])
            self.assertFalse(report["sprt_stopping_applied"])

            json_path = root / "report.json"
            markdown_path = root / "report.md"
            write_fixed_report(report, json_path, markdown_path)
            self.assertEqual(json.loads(json_path.read_text())["score"], report["score"])
            self.assertIn(
                "Bermuda MLP + GNN (epoch 7, \u03bb=1.0)",
                markdown_path.read_text(encoding="utf-8"),
            )

    def test_fixed_report_handles_extreme_score_without_non_finite_json(self):
        with tempfile.TemporaryDirectory() as directory:
            matches = Path(directory)
            write_pair(matches / "pair-00001.pgn", ["1-0", "0-1"])
            report = analyze_fixed_matches(matches, 1)
            self.assertEqual(report["score"], 1.0)
            self.assertIsNone(report["elo_difference"])
            self.assertIsNone(report["elo_uncertainty_95"])
            self.assertEqual(report["score_ci95_paired"], [1.0, 1.0])
            self.assertIsNone(json.loads(json.dumps(report, allow_nan=False))["elo_difference"])

    def test_fixed_markdown_handles_score_interval_reaching_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matches = root / "matches"
            matches.mkdir()
            write_pair(matches / "pair-00001.pgn", ["1-0", "0-1"])
            write_pair(matches / "pair-00002.pgn", ["1/2-1/2", "1/2-1/2"])
            report = analyze_fixed_matches(matches, 2)
            json_path = root / "report.json"
            markdown_path = root / "report.md"

            write_fixed_report(report, json_path, markdown_path)

            self.assertIn("finite Elo uncertainty is unavailable", markdown_path.read_text())


if __name__ == "__main__":
    unittest.main()
