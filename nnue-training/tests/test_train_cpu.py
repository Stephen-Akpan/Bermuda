import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
import struct

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train_cpu
import export_nnue


class CpuTrainerTests(unittest.TestCase):
    def test_chess768_features_follow_both_perspectives(self):
        board = chess.Board("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1")

        white_view = train_cpu.features_for(board, chess.WHITE)
        black_view = train_cpu.features_for(board, chess.BLACK)

        self.assertIn(5 * 64 + chess.E1, white_view)
        self.assertIn(11 * 64 + chess.E8, white_view)
        self.assertIn(1 * 64 + chess.C3, white_view)
        self.assertIn(5 * 64 + (chess.E8 ^ 56), black_view)
        self.assertIn(11 * 64 + (chess.E1 ^ 56), black_view)
        self.assertIn(7 * 64 + (chess.C3 ^ 56), black_view)

    def test_adam_update_changes_model_and_loss_is_finite(self):
        model = train_cpu.initialise_model(hidden_size=8, seed=3)
        moments = {
            "first": {name: np.zeros_like(value) for name, value in model.items()},
            "second": {name: np.zeros_like(value) for name, value in model.items()},
        }
        args = argparse.Namespace(
            eval_scale=400.0,
            wdl=0.75,
            learning_rate=0.001,
            weight_decay=0.0001,
        )
        features = np.full((2, 2, 32), train_cpu.PADDING_INDEX, dtype=np.uint16)
        features[:, 0, 0] = [4, 12]
        features[:, 1, 0] = [390, 402]
        before = model["feature_weights"].copy()

        loss = train_cpu.train_batch(
            model,
            moments,
            features,
            np.asarray([25.0, -40.0], dtype=np.float32),
            np.asarray([1.0, 0.5], dtype=np.float32),
            args,
            step=1,
        )

        self.assertTrue(np.isfinite(loss))
        self.assertFalse(np.array_equal(before, model["feature_weights"]))
        self.assertTrue(np.all(model["feature_weights"][train_cpu.PADDING_INDEX] == 0.0))

    def test_metrics_separate_mixed_search_and_wdl_losses(self):
        model = train_cpu.initialise_model(hidden_size=8, seed=4)
        features = np.full((2, 2, 32), train_cpu.PADDING_INDEX, dtype=np.uint16)
        features[:, 0, 0] = [4, 12]
        features[:, 1, 0] = [390, 402]
        data = (
            features,
            np.asarray([25.0, -40.0], dtype=np.float32),
            np.asarray([1.0, 0.5], dtype=np.float32),
        )

        metrics = train_cpu.evaluate_metrics(model, data, 2, 400.0, 0.75)

        self.assertAlmostEqual(metrics["rmse"] ** 2, metrics["loss"])
        self.assertAlmostEqual(metrics["search_rmse"] ** 2, metrics["search_loss"])
        self.assertAlmostEqual(metrics["wdl_rmse"] ** 2, metrics["wdl_loss"])

    def test_checkpoint_export_has_expected_header_and_float_arrays(self):
        model = train_cpu.initialise_model(hidden_size=64, seed=12)
        metadata = json.dumps({"eval_scale": 400.0})
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "best.npz"
            output_path = Path(temp_dir) / "network.nnue"
            np.savez_compressed(checkpoint_path, **model, metadata=np.asarray(metadata))

            export_nnue.export_checkpoint(checkpoint_path, output_path)

            data = output_path.read_bytes()
            magic, inputs, hidden, eval_scale = export_nnue.HEADER.unpack_from(data)
            self.assertEqual(magic, export_nnue.MAGIC)
            self.assertEqual(inputs, 768)
            self.assertEqual(hidden, 64)
            self.assertEqual(eval_scale, 400.0)
            self.assertEqual(len(data), export_nnue.HEADER.size + (768 * 64 + 64 + 128 + 1) * 4)


if __name__ == "__main__":
    unittest.main()