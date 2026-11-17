from __future__ import annotations

import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from neural.models.graph_evaluator import GraphEvaluator
from neural.training.baseline import FrozenBaseline, sha256_file
from neural.training.residual_data import (
    ResidualGraphDataset,
    collate_residual_batch,
    prepare_residual_cache,
)
from neural.training.train_graph import train


class ResidualTrainingTests(unittest.TestCase):
    @staticmethod
    def write_baseline_checkpoint(path: Path, seed: int = 5) -> None:
        rng = np.random.default_rng(seed)
        hidden_size = 8
        feature_weights = rng.normal(0.0, 0.02, (769, hidden_size)).astype(np.float32)
        feature_weights[768] = 0.0
        metadata = {
            "eval_scale": 400.0,
            "wdl": 0.75,
        }
        np.savez_compressed(
            path,
            feature_weights=feature_weights,
            feature_bias=np.zeros(hidden_size, dtype=np.float32),
            output_weights=rng.normal(0.0, 0.02, (2 * hidden_size,)).astype(np.float32),
            output_bias=np.zeros(1, dtype=np.float32),
            metadata=np.asarray(json.dumps(metadata)),
        )

    @staticmethod
    def write_dataset(path: Path, positions: list[tuple[str, int, float]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="ascii", newline="\n") as output:
            for fen, score, result in positions:
                output.write(f"{fen} | {score} | {result:.1f}\n")

    @staticmethod
    def training_args(
        root: Path,
        baseline_path: Path,
        train_path: Path,
        validation_path: Path,
        *,
        epochs: int = 2,
        resume: Path | None = None,
    ) -> argparse.Namespace:
        return argparse.Namespace(
            train_data=train_path,
            validation_data=validation_path,
            baseline_checkpoint=baseline_path,
            cache_dir=root / "cache",
            output_dir=root / "output",
            device="cpu",
            epochs=epochs,
            batch_size=2,
            preparation_batch_size=2,
            learning_rate=0.0003,
            weight_decay=0.0001,
            gradient_clip_norm=1.0,
            residual_multiplier=1.0,
            seed=19,
            max_positions=None,
            resume=resume,
        )

    def make_split_files(self, root: Path) -> tuple[Path, Path]:
        train_path = root / "train.txt"
        validation_path = root / "validation.txt"
        self.write_dataset(
            train_path,
            [
                ("4k3/8/8/8/8/2N5/8/4K3 w - - 0 1", 120, 1.0),
                ("4k3/8/8/8/8/2N5/8/4K3 b - - 0 1", -80, 0.5),
                ("4k3/8/8/8/8/3N4/8/4K3 w - - 0 1", 40, 0.5),
                ("4k3/8/8/8/8/3N4/8/4K3 b - - 0 1", -20, 0.0),
            ],
        )
        self.write_dataset(
            validation_path,
            [
                ("4k3/8/8/8/8/4N3/8/4K3 w - - 0 1", 90, 1.0),
                ("4k3/8/8/8/8/4N3/8/4K3 b - - 0 1", -60, 0.0),
            ],
        )
        return train_path, validation_path

    def test_baseline_matches_existing_numpy_forward_and_is_frozen(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nnue-training"))
        import train_cpu

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "baseline.npz"
            self.write_baseline_checkpoint(checkpoint_path)
            baseline = FrozenBaseline(checkpoint_path)
            self.assertFalse(baseline.training)
            self.assertEqual(list(baseline.parameters()), [])

            position = chess.Board("4k3/8/8/8/8/2N5/8/4K3 b - - 0 1")
            row = np.full((2, 32), 768, dtype=np.uint16)
            stm_features = train_cpu.features_for(position, position.turn)
            ntm_features = train_cpu.features_for(position, not position.turn)
            row[0, :len(stm_features)] = stm_features
            row[1, :len(ntm_features)] = ntm_features
            features = torch.from_numpy(row.astype(np.int64)).unsqueeze(0)
            before = baseline(features)
            white = baseline.predict_white(features, torch.tensor([False]))
            expected_stm = train_cpu.sigmoid(
                train_cpu.forward(
                    {
                        "feature_weights": baseline.feature_weights.cpu().numpy(),
                        "feature_bias": baseline.feature_bias.cpu().numpy(),
                        "output_weights": baseline.output_weights.cpu().numpy(),
                        "output_bias": baseline.output_bias.cpu().numpy(),
                    },
                    row[None, ...],
                )[0]
            )
            self.assertEqual(tuple(before.shape), (1, 1))
            np.testing.assert_allclose(before.cpu().numpy().reshape(-1), expected_stm, rtol=1e-6)
            torch.testing.assert_close(white, 1.0 - before)
            torch.testing.assert_close(baseline(features), before)

    def test_residual_targets_cache_round_trip_and_prediction_shapes(self):
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "nnue-training"))
        import train_cpu

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline_path = root / "baseline.npz"
            self.write_baseline_checkpoint(baseline_path)
            baseline = FrozenBaseline(baseline_path)
            train_path, _ = self.make_split_files(root)

            dataset = prepare_residual_cache(train_path, root / "cache", "train", baseline)
            loaded = prepare_residual_cache(train_path, root / "cache", "train", baseline)
            self.assertIsInstance(loaded, ResidualGraphDataset)
            self.assertEqual(dataset.metadata["cache_signature"], loaded.metadata["cache_signature"])
            self.assertEqual(len(dataset), 4)

            batch = collate_residual_batch([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["square_features"].shape), (2, 64, 16))
            self.assertEqual(tuple(batch["global_features"].shape), (2, 5))
            self.assertEqual(tuple(batch["attack_adj"].shape), (2, 64, 64))
            self.assertEqual(tuple(batch["defence_adj"].shape), (2, 64, 64))
            self.assertEqual(tuple(batch["baseline_predictions"].shape), (2, 1))
            self.assertEqual(tuple(batch["residual_targets"].shape), (2, 1))

            rows, scores, results, boards = train_cpu.load_data(
                train_path, None, include_boards=True
            )
            baseline_features = torch.from_numpy(rows[:2].astype(np.int64))
            white_to_move = torch.tensor([board.turn for board in boards[:2]], dtype=torch.bool)
            expected_baseline = baseline.predict_white(baseline_features, white_to_move)
            target_stm = train_cpu.make_target(scores[:2], results[:2], 400.0, 0.75)
            expected_target = np.where(white_to_move.numpy(), target_stm, 1.0 - target_stm)
            np.testing.assert_allclose(
                batch["baseline_predictions"].numpy()[:, 0],
                expected_baseline.numpy()[:, 0],
                rtol=1e-6,
            )
            np.testing.assert_allclose(
                batch["original_targets"].numpy()[:, 0], expected_target, rtol=1e-6
            )
            torch.testing.assert_close(
                batch["residual_targets"],
                batch["original_targets"] - batch["baseline_predictions"],
            )

            model = GraphEvaluator().eval()
            with torch.no_grad():
                graph_prediction = model(
                    batch["square_features"],
                    batch["global_features"],
                    batch["attack_adj"],
                    batch["defence_adj"],
                    batch["white_king_square"],
                    batch["black_king_square"],
                )
            combined_prediction = batch["baseline_predictions"] + graph_prediction
            self.assertEqual(tuple(graph_prediction.shape), (2, 1))
            self.assertEqual(tuple(combined_prediction.shape), (2, 1))
            torch.testing.assert_close(
                combined_prediction,
                batch["baseline_predictions"] + graph_prediction,
            )
            np.testing.assert_array_equal(
                dataset.arrays["piece_identity"][0],
                loaded.arrays["piece_identity"][0],
            )
            dataset.close()
            loaded.close()

    def test_cache_identity_changes_when_baseline_checkpoint_changes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline_path = root / "baseline.npz"
            self.write_baseline_checkpoint(baseline_path, seed=5)
            baseline_a = FrozenBaseline(baseline_path)
            train_path, _ = self.make_split_files(root)
            cache_a = prepare_residual_cache(train_path, root / "cache", "train", baseline_a)

            self.write_baseline_checkpoint(baseline_path, seed=8)
            baseline_b = FrozenBaseline(baseline_path)
            cache_b = prepare_residual_cache(train_path, root / "cache", "train", baseline_b)

            self.assertNotEqual(baseline_a.checkpoint_id, baseline_b.checkpoint_id)
            self.assertNotEqual(cache_a.cache_path, cache_b.cache_path)
            self.assertEqual(cache_b.metadata["baseline_sha256"], baseline_b.checkpoint_id)
            cache_a.close()
            cache_b.close()

    def test_training_keeps_splits_separate_baseline_frozen_and_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline_path = root / "baseline.npz"
            self.write_baseline_checkpoint(baseline_path)
            train_path, validation_path = self.make_split_files(root)
            baseline_hash_before = sha256_file(baseline_path)
            baseline_before = FrozenBaseline(baseline_path)
            sample = torch.zeros((1, 2, 32), dtype=torch.long)
            baseline_prediction_before = baseline_before(sample)

            history_path = train(self.training_args(root, baseline_path, train_path, validation_path))
            self.assertTrue(history_path.is_file())
            self.assertTrue((root / "output" / "best.pt").is_file())
            self.assertTrue((root / "output" / "last.pt").is_file())
            with history_path.open("r", newline="", encoding="utf-8") as history_file:
                history = list(csv.DictReader(history_file))
            self.assertEqual(sha256_file(baseline_path), baseline_hash_before)
            torch.testing.assert_close(FrozenBaseline(baseline_path)(sample), baseline_prediction_before)

            checkpoint = torch.load(root / "output" / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["baseline_checkpoint_id"], baseline_before.checkpoint_id)
            self.assertEqual(checkpoint["residual_multiplier"], 1.0)
            self.assertEqual(checkpoint["model_config"]["hidden_size"], 32)
            self.assertEqual(checkpoint["train_cache_signature"], json_metadata(root / "cache", "train"))
            self.assertEqual(cache_metadata(root / "cache", "train")["split"], "train")
            self.assertEqual(cache_metadata(root / "cache", "validation")["split"], "validation")
            self.assertNotEqual(
                cache_metadata(root / "cache", "train")["source_path"],
                cache_metadata(root / "cache", "validation")["source_path"],
            )
            self.assertNotEqual(
                checkpoint["train_cache_signature"],
                checkpoint["validation_cache_signature"],
            )
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertEqual(
                float(checkpoint["best_validation_metric"]["combined_mae"]),
                min(float(row["combined_mae"]) for row in history),
            )

            restored = GraphEvaluator()
            restored.load_state_dict(checkpoint["graph_state_dict"])
            self.assertEqual(checkpoint["best_validation_metric"].keys(), {"combined_mae"})

            resume_args = self.training_args(
                root,
                baseline_path,
                train_path,
                validation_path,
                epochs=3,
                resume=root / "output" / "last.pt",
            )
            train(resume_args)
            resumed = torch.load(root / "output" / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed["epoch"], 3)

    def test_same_file_cannot_be_used_for_training_and_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            baseline_path = root / "baseline.npz"
            self.write_baseline_checkpoint(baseline_path)
            train_path, _ = self.make_split_files(root)
            args = self.training_args(root, baseline_path, train_path, train_path)

            with self.assertRaisesRegex(ValueError, "must be different files"):
                train(args)


def json_metadata(cache_root: Path, split: str) -> str:
    return cache_metadata(cache_root, split)["cache_signature"]


def cache_metadata(cache_root: Path, split: str) -> dict[str, object]:
    matching = list(cache_root.glob(f"{split}-*/metadata.json"))
    if len(matching) != 1:
        raise AssertionError(f"expected one {split} cache, got {len(matching)}")
    with matching[0].open("r", encoding="utf-8") as source:
        return json.load(source)


if __name__ == "__main__":
    unittest.main()
