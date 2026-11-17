"""Basic relational graph evaluator for Bermuda chess positions."""

from __future__ import annotations

import torch
from torch import nn


class GraphEvaluator(nn.Module):
    """One-layer relational evaluator returning a white-perspective score residual."""

    square_count = 64
    square_feature_count = 16
    global_feature_count = 5
    hidden_size = 32
    pooled_size = 133

    def __init__(self) -> None:
        super().__init__()
        self.square_projection = nn.Linear(self.square_feature_count, self.hidden_size)
        self.square_embedding = nn.Embedding(self.square_count, self.hidden_size)
        self.global_projection = nn.Linear(self.global_feature_count, self.hidden_size)

        self.self_transform = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attack_transform = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.defence_transform = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        self.value_head = nn.Sequential(
            nn.Linear(self.pooled_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.square_embedding.weight, mean=0.0, std=0.02)
        nn.init.ones_(self.layer_norm.weight)
        nn.init.zeros_(self.layer_norm.bias)

    def _validate_inputs(
        self,
        square_features: torch.Tensor,
        global_features: torch.Tensor,
        attack_adj: torch.Tensor,
        defence_adj: torch.Tensor,
        white_king_square: torch.Tensor,
        black_king_square: torch.Tensor,
    ) -> None:
        tensors = (
            ("square_features", square_features, torch.float32),
            ("global_features", global_features, torch.float32),
            ("attack_adj", attack_adj, torch.float32),
            ("defence_adj", defence_adj, torch.float32),
            ("white_king_square", white_king_square, torch.long),
            ("black_king_square", black_king_square, torch.long),
        )
        for name, tensor, expected_dtype in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.dtype != expected_dtype:
                raise TypeError(f"{name} must have dtype {expected_dtype}")

        if square_features.ndim != 3:
            raise ValueError("square_features must have shape [B, 64, 16]")
        batch_size = square_features.shape[0]
        expected_shapes = (
            ("square_features", square_features, (batch_size, self.square_count, self.square_feature_count)),
            ("global_features", global_features, (batch_size, self.global_feature_count)),
            ("attack_adj", attack_adj, (batch_size, self.square_count, self.square_count)),
            ("defence_adj", defence_adj, (batch_size, self.square_count, self.square_count)),
            ("white_king_square", white_king_square, (batch_size,)),
            ("black_king_square", black_king_square, (batch_size,)),
        )
        if batch_size == 0:
            raise ValueError("batch size must be greater than zero")
        for name, tensor, expected_shape in expected_shapes:
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")

        model_device = self.square_projection.weight.device
        for name, tensor, _ in tensors:
            if tensor.device != model_device:
                raise ValueError(f"{name} must be on the model device ({model_device})")

        for name, king_squares in (
            ("white_king_square", white_king_square),
            ("black_king_square", black_king_square),
        ):
            if torch.any((king_squares < 0) | (king_squares >= self.square_count)):
                raise ValueError(f"{name} indices must be in [0, 63]")

    @staticmethod
    def _normalize_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
        incoming_degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return adjacency / incoming_degree

    def forward(
        self,
        square_features: torch.Tensor,
        global_features: torch.Tensor,
        attack_adj: torch.Tensor,
        defence_adj: torch.Tensor,
        white_king_square: torch.Tensor,
        black_king_square: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            square_features,
            global_features,
            attack_adj,
            defence_adj,
            white_king_square,
            black_king_square,
        )

        batch_size = square_features.shape[0]
        square_ids = torch.arange(self.square_count, device=square_features.device)
        global_embedding = self.global_projection(global_features).unsqueeze(1)
        h0 = torch.relu(
            self.square_projection(square_features)
            + self.square_embedding(square_ids).unsqueeze(0)
            + global_embedding
        )

        normalized_attack = self._normalize_adjacency(attack_adj)
        normalized_defence = self._normalize_adjacency(defence_adj)
        attack_messages = torch.bmm(normalized_attack, self.attack_transform(h0))
        defence_messages = torch.bmm(normalized_defence, self.defence_transform(h0))
        h1 = torch.relu(
            self.layer_norm(
                h0
                + self.self_transform(h0)
                + attack_messages
                + defence_messages
            )
        )

        batch_indices = torch.arange(batch_size, device=h1.device)
        white_king_hidden = h1[batch_indices, white_king_square]
        black_king_hidden = h1[batch_indices, black_king_square]
        pooled = torch.cat(
            (
                h1.mean(dim=1),
                h1.max(dim=1).values,
                white_king_hidden,
                black_king_hidden,
                global_features,
            ),
            dim=1,
        )
        return self.value_head(pooled)