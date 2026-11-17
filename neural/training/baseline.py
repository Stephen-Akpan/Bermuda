"""Frozen inference adapter for Bermuda's existing NumPy MLP checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenBaseline(nn.Module):
    """Torch inference adapter for the existing Chess768 NumPy MLP checkpoint."""

    input_count = 768
    padding_index = 768
    max_pieces = 32

    def __init__(self, checkpoint_path: Path, device: torch.device | str = "cpu") -> None:
        super().__init__()
        checkpoint_path = checkpoint_path.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"baseline checkpoint not found: {checkpoint_path}")

        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            feature_weights = np.asarray(checkpoint["feature_weights"], dtype=np.float32)
            feature_bias = np.asarray(checkpoint["feature_bias"], dtype=np.float32)
            output_weights = np.asarray(checkpoint["output_weights"], dtype=np.float32)
            output_bias = np.asarray(checkpoint["output_bias"], dtype=np.float32)
            metadata = json.loads(str(checkpoint["metadata"].item()))

        if feature_weights.ndim != 2 or feature_weights.shape[0] != self.input_count + 1:
            raise ValueError("baseline feature_weights must have shape [769, hidden]")
        hidden_size = feature_weights.shape[1]
        if feature_bias.shape != (hidden_size,):
            raise ValueError("baseline feature_bias has an incompatible shape")
        if output_weights.shape != (2 * hidden_size,):
            raise ValueError("baseline output_weights has an incompatible shape")
        if output_bias.shape != (1,):
            raise ValueError("baseline output_bias must contain one value")
        if any(not np.isfinite(array).all() for array in (
            feature_weights, feature_bias, output_weights, output_bias
        )):
            raise ValueError("baseline checkpoint contains non-finite weights")
        if np.any(feature_weights[self.padding_index] != 0.0):
            raise ValueError("baseline padding feature must be zero")

        self.register_buffer("feature_weights", torch.from_numpy(feature_weights.copy()))
        self.register_buffer("feature_bias", torch.from_numpy(feature_bias.copy()))
        self.register_buffer("output_weights", torch.from_numpy(output_weights.copy()))
        self.register_buffer("output_bias", torch.from_numpy(output_bias.copy()))
        self.hidden_size = hidden_size
        self.eval_scale = float(metadata.get("eval_scale", 400.0))
        self.wdl = float(metadata.get("wdl", 0.75))
        if not np.isfinite(self.eval_scale) or self.eval_scale <= 0.0:
            raise ValueError("baseline eval_scale must be positive and finite")
        if not np.isfinite(self.wdl) or not 0.0 <= self.wdl <= 1.0:
            raise ValueError("baseline wdl must be in [0, 1]")

        self.checkpoint_path = checkpoint_path
        self.checkpoint_id = sha256_file(checkpoint_path)
        self.to(device)
        self.eval()

    @torch.inference_mode()
    def forward(self, stm_features: torch.Tensor) -> torch.Tensor:
        if stm_features.dtype != torch.long:
            raise TypeError("stm_features must have dtype torch.long")
        if stm_features.ndim != 3 or tuple(stm_features.shape[1:]) != (2, self.max_pieces):
            raise ValueError("stm_features must have shape [B, 2, 32]")
        if stm_features.device != self.feature_weights.device:
            raise ValueError("stm_features and baseline must be on the same device")
        if torch.any((stm_features < 0) | (stm_features > self.padding_index)):
            raise ValueError("baseline feature indices must be in [0, 768]")

        accumulators = self.feature_weights[stm_features].sum(dim=2) + self.feature_bias
        clipped = accumulators.clamp(0.0, 1.0)
        hidden = torch.cat((clipped[:, 0].square(), clipped[:, 1].square()), dim=1)
        logits = hidden @ self.output_weights + self.output_bias[0]
        return torch.sigmoid(logits.clamp(-40.0, 40.0)).unsqueeze(1)

    @torch.inference_mode()
    def predict_white(
        self,
        stm_features: torch.Tensor,
        white_to_move: torch.Tensor,
    ) -> torch.Tensor:
        if white_to_move.dtype != torch.bool or tuple(white_to_move.shape) != (stm_features.shape[0],):
            raise ValueError("white_to_move must be a bool tensor with shape [B]")
        if white_to_move.device != self.feature_weights.device:
            raise ValueError("white_to_move and baseline must be on the same device")
        prediction_stm = self(stm_features)
        return torch.where(white_to_move.unsqueeze(1), prediction_stm, 1.0 - prediction_stm)