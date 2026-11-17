from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch

from neural.export.graph_export import export_checkpoint
from neural.models.graph_evaluator import GraphEvaluator
from neural.training.train_graph import MODEL_CONFIG


ROOT = Path(__file__).resolve().parents[2]
PARITY_EXECUTABLE = ROOT / "build" / "windows" / "x64" / "release" / "graph_eval_parity.exe"


class CppPythonParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ["xmake", "build", "graph_eval_parity"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if not PARITY_EXECUTABLE.is_file():
            raise FileNotFoundError(f"parity executable was not built: {PARITY_EXECUTABLE}")

    def test_graph_encoder_and_evaluator_parity_on_reference_positions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)
            checkpoint_path = temporary_path / "parity-checkpoint.pt"
            model_path = temporary_path / "parity-model.bmgnn"
            torch.manual_seed(20261114)
            model = GraphEvaluator().eval()
            torch.save(
                {
                    "model_config": MODEL_CONFIG,
                    "graph_state_dict": model.state_dict(),
                    "residual_multiplier": 0.8,
                    "baseline_checkpoint_id": "42" * 32,
                },
                checkpoint_path,
            )
            reference_json = export_checkpoint(checkpoint_path, model_path)
            reference_bin = reference_json.with_suffix(".bin")
            reference_document = json.loads(reference_json.read_text(encoding="utf-8"))
            self.assertEqual(
                {case["name"] for case in reference_document["cases"]},
                {"opening", "middlegame", "endgame", "en-passant-castling", "pinned-rook"},
            )

            result = subprocess.run(
                [str(PARITY_EXECUTABLE), str(model_path), str(reference_bin)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            rows = list(csv.DictReader(result.stdout.splitlines()))
            self.assertEqual(len(rows), 5)

            output_errors = []
            graph_encoder_errors = []
            for row in rows:
                output_errors.append(
                    abs(float(row["cpp_output"]) - float(row["expected_output"]))
                )
                graph_encoder_errors.append(float(row["graph_max_abs"]))
                self.assertEqual(row["king_indices_match"], "1")
                self.assertEqual(float(row["graph_max_abs"]), 0.0)
                self.assertEqual(float(row["graph_mean_abs"]), 0.0)

            maximum_output_error = max(output_errors)
            mean_output_error = sum(output_errors) / len(output_errors)
            self.assertLess(maximum_output_error, 5e-5)
            self.assertLess(mean_output_error, 2e-5)
            self.assertEqual(max(graph_encoder_errors), 0.0)
            print(
                "Python/C++ GraphEvaluator parity: "
                f"max_abs={maximum_output_error:.9g}, mean_abs={mean_output_error:.9g}; "
                "GraphEncoder max/mean abs error=0"
            )


if __name__ == "__main__":
    unittest.main()
