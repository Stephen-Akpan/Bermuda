from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from neural.models.graph_evaluator import GraphEvaluator


class GraphEvaluatorTests(unittest.TestCase):
    @staticmethod
    def make_inputs(batch_size: int, empty_adjacency: bool = False) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(17)
        square_features = torch.randn(batch_size, 64, 16, generator=generator)
        global_features = torch.randn(batch_size, 5, generator=generator)
        if empty_adjacency:
            attack_adj = torch.zeros(batch_size, 64, 64)
            defence_adj = torch.zeros(batch_size, 64, 64)
        else:
            attack_adj = torch.randint(0, 2, (batch_size, 64, 64), generator=generator).float()
            defence_adj = torch.randint(0, 2, (batch_size, 64, 64), generator=generator).float()
        return {
            "square_features": square_features,
            "global_features": global_features,
            "attack_adj": attack_adj,
            "defence_adj": defence_adj,
            "white_king_square": torch.tensor(
                [(5 + index * 13) % 64 for index in range(batch_size)], dtype=torch.long
            ),
            "black_king_square": torch.tensor(
                [(42 + index * 11) % 64 for index in range(batch_size)], dtype=torch.long
            ),
        }

    def test_single_and_batch_forward_shapes(self):
        model = GraphEvaluator().eval()
        single = model(**self.make_inputs(1))
        batch = model(**self.make_inputs(4))

        self.assertEqual(tuple(single.shape), (1, 1))
        self.assertEqual(tuple(batch.shape), (4, 1))
        self.assertTrue(torch.isfinite(single).all())
        self.assertTrue(torch.isfinite(batch).all())

    def test_empty_adjacency_produces_finite_output(self):
        model = GraphEvaluator().eval()
        output = model(**self.make_inputs(3, empty_adjacency=True))

        self.assertEqual(tuple(output.shape), (3, 1))
        self.assertTrue(torch.isfinite(output).all())

    def test_attack_adjacency_uses_target_source_orientation(self):
        model = GraphEvaluator().eval()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.square_projection.weight[0, 0] = 1.0
            model.attack_transform.weight[0, 0] = 1.0
            model.layer_norm.weight.fill_(1.0)
            model.value_head[0].weight[0, 64] = 1.0
            model.value_head[2].weight[0, 0] = 1.0

        inputs = self.make_inputs(1, empty_adjacency=True)
        inputs["square_features"].zero_()
        inputs["square_features"][0, 1, 0] = 1.0
        inputs["attack_adj"][0, 0, 1] = 1.0
        inputs["white_king_square"] = torch.tensor([0], dtype=torch.long)
        inputs["black_king_square"] = torch.tensor([2], dtype=torch.long)
        with torch.no_grad():
            target_source_output = model(**inputs)

        inputs["attack_adj"].zero_()
        inputs["attack_adj"][0, 1, 0] = 1.0
        with torch.no_grad():
            transposed_output = model(**inputs)

        self.assertGreater(target_source_output.item(), 0.0)
        self.assertEqual(transposed_output.item(), 0.0)

    def test_batched_king_indices_match_individual_examples(self):
        model = GraphEvaluator().eval()
        batch_inputs = self.make_inputs(3)
        batch_output = model(**batch_inputs)
        self.assertEqual(len(torch.unique(batch_inputs["white_king_square"])), 3)
        self.assertEqual(len(torch.unique(batch_inputs["black_king_square"])), 3)

        individual_outputs = []
        for index in range(3):
            individual_inputs = {
                name: tensor[index:index + 1]
                for name, tensor in batch_inputs.items()
            }
            individual_outputs.append(model(**individual_inputs))
        expected = torch.cat(individual_outputs, dim=0)

        torch.testing.assert_close(batch_output, expected)

    def test_backpropagation_reaches_inputs_and_parameters(self):
        model = GraphEvaluator()
        inputs = self.make_inputs(2)
        for name in ("square_features", "global_features", "attack_adj", "defence_adj"):
            inputs[name].requires_grad_()

        output = model(**inputs)
        output.square().mean().backward()

        for name in ("square_features", "global_features", "attack_adj", "defence_adj"):
            self.assertIsNotNone(inputs[name].grad, name)
            self.assertTrue(torch.isfinite(inputs[name].grad).all(), name)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_eval_inference_is_deterministic(self):
        model = GraphEvaluator().eval()
        inputs = self.make_inputs(2)

        with torch.no_grad():
            first = model(**inputs)
            second = model(**inputs)

        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_state_dict_save_load_preserves_output(self):
        model = GraphEvaluator().eval()
        inputs = self.make_inputs(2)
        with torch.no_grad():
            expected = model(**inputs)

        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "graph_evaluator.pt"
            torch.save(model.state_dict(), checkpoint)
            restored = GraphEvaluator().eval()
            restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))

        with torch.no_grad():
            actual = restored(**inputs)
        torch.testing.assert_close(actual, expected)

    def test_invalid_shapes_dtypes_and_king_indices_are_rejected(self):
        model = GraphEvaluator()
        inputs = self.make_inputs(2)

        invalid_shape = dict(inputs)
        invalid_shape["attack_adj"] = torch.zeros(2, 64, 32)
        with self.assertRaisesRegex(ValueError, "attack_adj must have shape"):
            model(**invalid_shape)

        invalid_dtype = dict(inputs)
        invalid_dtype["global_features"] = inputs["global_features"].double()
        with self.assertRaisesRegex(TypeError, "global_features must have dtype"):
            model(**invalid_dtype)

        invalid_king = dict(inputs)
        invalid_king["black_king_square"] = torch.tensor([2, 64], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "black_king_square indices"):
            model(**invalid_king)


if __name__ == "__main__":
    unittest.main()