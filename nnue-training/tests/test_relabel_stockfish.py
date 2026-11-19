import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import relabel_stockfish


class FakeStockfish:
    id = {"name": "Stockfish 19 test double", "author": "tests"}
    options = {"Threads": object(), "Hash": object()}

    def __init__(self):
        self.calls = 0
        self.config = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def configure(self, config):
        self.config = config

    def analyse(self, board, limit):
        self.calls += 1
        self.asserted_depth = limit.depth
        if self.calls == 2:
            score = chess.engine.PovScore(chess.engine.Mate(2), chess.WHITE)
        elif board.fullmove_number == 2:
            score = chess.engine.PovScore(chess.engine.Cp(42), chess.BLACK)
        else:
            score = chess.engine.PovScore(chess.engine.Cp(17), chess.WHITE)
        return {"score": score}


class StockfishRelabelTests(unittest.TestCase):
    def test_subset_selection_is_reproducible(self):
        board = chess.Board()
        rows = [(index + 1, board.fen(), board.copy(), "0.5") for index in range(20)]

        selected = relabel_stockfish.choose_rows(rows, 5, 1908)

        self.assertEqual(len(selected), 5)
        self.assertEqual(selected, relabel_stockfish.choose_rows(rows, 5, 1908))
        self.assertEqual(selected, sorted(selected, key=lambda row: row[0]))

    def test_relabel_writes_white_scores_and_skips_mates(self):
        board = chess.Board()
        rows = [
            f"{board.fen()} | 0 | 1.0",
        ]
        board.push_uci("e2e4")
        rows.append(f"{board.fen()} | 0 | 0.5")
        board.push_uci("e7e5")
        rows.append(f"{board.fen()} | 0 | 0.0")

        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            input_path = directory / "validation.txt"
            output_path = directory / "validation-stockfish.txt"
            engine_path = directory / "stockfish.exe"
            input_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            engine_path.write_bytes(b"test executable placeholder")
            fake_engine = FakeStockfish()
            args = argparse.Namespace(
                input=input_path,
                output=output_path,
                engine=engine_path,
                depth=8,
                max_positions=None,
                threads=1,
                hash_mb=64,
                max_score_cp=9999,
                seed=1908,
                force=False,
            )

            with patch(
                "relabel_stockfish.chess.engine.SimpleEngine.popen_uci",
                return_value=fake_engine,
            ):
                metadata = relabel_stockfish.relabel(args)

            output_rows = output_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual([row.split(" | ")[1:] for row in output_rows], [
                ["17", "1.0"],
                ["-42", "0.0"],
            ])
            self.assertEqual(fake_engine.config, {"Threads": 1, "Hash": 64})
            self.assertEqual(fake_engine.asserted_depth, 8)
            self.assertEqual(metadata["selected_positions"], 3)
            self.assertEqual(metadata["labelled_positions"], 2)
            self.assertEqual(metadata["skipped_mate_scores"], 1)
            manifest = json.loads(output_path.with_suffix(".txt.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["depth"], 8)
            self.assertEqual(manifest["stockfish"]["name"], "Stockfish 19 test double")

    def test_source_file_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            source = Path(temporary_directory) / "dataset.txt"
            source.write_text(f"{chess.STARTING_FEN} | 0 | 0.5\n", encoding="utf-8")
            args = argparse.Namespace(
                input=source,
                output=source,
                engine=source,
                max_positions=1,
                threads=1,
                hash_mb=64,
                max_score_cp=9999,
            )
            with self.assertRaisesRegex(ValueError, "must differ"):
                relabel_stockfish.relabel(args)


if __name__ == "__main__":
    unittest.main()
