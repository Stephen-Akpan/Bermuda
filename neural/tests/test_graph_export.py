from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from neural.export.graph_export import (
    ARCHITECTURE_VERSION,
    FORMAT_VERSION,
    PARAMETER_COUNT,
    export_checkpoint,
    load_exported_model,
)
from neural.models.graph_evaluator import GraphEvaluator
from neural.training.train_graph import MODEL_CONFIG


class GraphExportTests(unittest.TestCase):
    @staticmethod
    def write_checkpoint(path: Path) -> GraphEvaluator:
        torch.manual_seed(414)
        model = GraphEvaluator().eval()
        torch.save(
            {
                "model_config": MODEL_CONFIG,
                "graph_state_dict": model.state_dict(),
                "residual_multiplier": 0.75,
                "baseline_checkpoint_id": "ab" * 32,
                "epoch": 7,
            },
            path,
        )
        return model

    @staticmethod
    def synthetic_inputs(batch_size: int = 2) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(78)
        return {
            "square_features": torch.randn(batch_size, 64, 16, generator=generator),
            "global_features": torch.randn(batch_size, 5, generator=generator),
            "attack_adj": torch.randint(
                0, 2, (batch_size, 64, 64), generator=generator
            ).float(),
            "defence_adj": torch.randint(
                0, 2, (batch_size, 64, 64), generator=generator
            ).float(),
            "white_king_square": torch.tensor([4 + 8 * i for i in range(batch_size)]),
            "black_king_square": torch.tensor([60 - 8 * i for i in range(batch_size)]),
        }

    def test_export_reload_preserves_all_parameters_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "trained.pt"
            export_path = root / "graph.bmgnn"
            expected_model = self.write_checkpoint(checkpoint_path)
            reference_path = export_checkpoint(checkpoint_path, export_path)

            restored, metadata = load_exported_model(export_path)
            self.assertEqual(metadata["format_version"], FORMAT_VERSION)
            self.assertEqual(metadata["architecture_version"], ARCHITECTURE_VERSION)
            self.assertEqual(metadata["dtype"], "float32")
            self.assertEqual(metadata["tensor_count"], 14)
            self.assertEqual(metadata["parameter_count"], PARAMETER_COUNT)
            self.assertEqual(metadata["residual_multiplier"], 0.75)
            self.assertEqual(metadata["baseline_checkpoint_id"], "ab" * 32)

            expected_state = expected_model.state_dict()
            actual_state = restored.state_dict()
            self.assertEqual(expected_state.keys(), actual_state.keys())
            for name in expected_state:
                self.assertEqual(expected_state[name].dtype, torch.float32)
                self.assertTrue(torch.equal(expected_state[name], actual_state[name]), name)

            inputs = self.synthetic_inputs()
            with torch.inference_mode():
                expected_output = expected_model(**inputs)
                actual_output = restored(**inputs)
            torch.testing.assert_close(actual_output, expected_output, rtol=0.0, atol=0.0)
            self.assertEqual(tuple(actual_output.shape), (2, 1))
            self.assertTrue(reference_path.is_file())

    def test_reference_positions_have_full_inputs_and_expected_outputs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "trained.pt"
            export_path = root / "graph.bmgnn"
            self.write_checkpoint(checkpoint_path)
            reference_path = export_checkpoint(checkpoint_path, export_path)
            restored, _ = load_exported_model(export_path)
            document = json.loads(reference_path.read_text(encoding="utf-8"))

            self.assertEqual(document["architecture_version"], ARCHITECTURE_VERSION)
            self.assertEqual(document["baseline_checkpoint_id"], "ab" * 32)
            self.assertEqual(document["residual_multiplier"], 0.75)
            self.assertEqual(document["adjacency_convention"], "adjacency[target][source] = 1")
            self.assertEqual(len(document["cases"]), 5)
            self.assertTrue(Path(reference_path.with_suffix(".bin")).is_file())
            for case in document["cases"]:
                inputs = {
                    "square_features": torch.tensor(case["inputs"]["square_features"], dtype=torch.float32).unsqueeze(0),
                    "global_features": torch.tensor(case["inputs"]["global_features"], dtype=torch.float32).unsqueeze(0),
                    "attack_adj": torch.tensor(case["inputs"]["attack_adj"], dtype=torch.float32).unsqueeze(0),
                    "defence_adj": torch.tensor(case["inputs"]["defence_adj"], dtype=torch.float32).unsqueeze(0),
                    "white_king_square": torch.tensor([case["inputs"]["white_king_square"]], dtype=torch.long),
                    "black_king_square": torch.tensor([case["inputs"]["black_king_square"]], dtype=torch.long),
                }
                self.assertEqual(tuple(inputs["square_features"].shape), (1, 64, 16))
                self.assertEqual(tuple(inputs["attack_adj"].shape), (1, 64, 64))
                with torch.inference_mode():
                    actual = restored(**inputs).item()
                self.assertAlmostEqual(actual, case["expected_graph_output"], places=7)

    def test_export_rejects_bad_tensor_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "bad.pt"
            model = self.write_checkpoint(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint["graph_state_dict"]["attack_transform.weight"] = torch.zeros(32, 31)
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "attack_transform.weight must have shape"):
                export_checkpoint(checkpoint_path, root / "bad.bmgnn")

    def test_export_rejects_non_float32_parameters(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "bad.pt"
            self.write_checkpoint(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint["graph_state_dict"]["value_head.2.bias"] = torch.zeros(1, dtype=torch.float64)
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(TypeError, "value_head.2.bias must be float32"):
                export_checkpoint(checkpoint_path, root / "bad.bmgnn")

    def test_export_rejects_unrepresentable_residual_multiplier(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "bad.pt"
            self.write_checkpoint(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint["residual_multiplier"] = 1e100
            torch.save(checkpoint, checkpoint_path)

            with self.assertRaisesRegex(ValueError, "outside float32 range"):
                export_checkpoint(checkpoint_path, root / "bad.bmgnn")

    def test_loader_detects_payload_corruption(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint_path = root / "trained.pt"
            export_path = root / "graph.bmgnn"
            self.write_checkpoint(checkpoint_path)
            export_checkpoint(checkpoint_path, export_path)
            payload = bytearray(export_path.read_bytes())
            payload[-40] ^= 0x01
            export_path.write_bytes(payload)

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_exported_model(export_path)


if __name__ == "__main__":
    unittest.main()