"""Versioned little-endian export for trained Bermuda GraphEvaluator checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any

import chess
import numpy as np
import torch

from neural.models.graph_evaluator import GraphEvaluator
from neural.training.train_graph import MODEL_CONFIG


MAGIC = b"BMGRPH01"
FORMAT_VERSION = 1
ARCHITECTURE_VERSION = 1
FLOAT32_CODE = 1
FILE_HEADER = struct.Struct("<8sIIIIQfI")
TENSOR_HEADER = struct.Struct("<HB")
PARAMETER_COUNT = 14561
REFERENCE_MAGIC = b"BMGREF01"
REFERENCE_HEADER = struct.Struct("<8sI")
REFERENCE_CASE_PREFIX = struct.Struct("<I")
REFERENCE_KINGS = struct.Struct("<II")

TENSOR_ORDER = (
    ("square_projection.weight", (32, 16)),
    ("square_projection.bias", (32,)),
    ("square_embedding.weight", (64, 32)),
    ("global_projection.weight", (32, 5)),
    ("global_projection.bias", (32,)),
    ("self_transform.weight", (32, 32)),
    ("attack_transform.weight", (32, 32)),
    ("defence_transform.weight", (32, 32)),
    ("layer_norm.weight", (32,)),
    ("layer_norm.bias", (32,)),
    ("value_head.0.weight", (64, 133)),
    ("value_head.0.bias", (64,)),
    ("value_head.2.weight", (1, 64)),
    ("value_head.2.bias", (1,)),
)

REFERENCE_POSITIONS = (
    ("opening", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("middlegame", "r1bq1rk1/ppp2ppp/2npbn2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8"),
    ("endgame", "8/8/2k5/8/3K4/4P3/8/8 w - - 0 1"),
    ("en-passant-castling", "r3k2r/8/8/3pP3/8/8/8/R3K2R w Kq d6 0 1"),
    ("pinned-rook", "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1"),
)


def _validate_checkpoint(checkpoint: dict[str, Any]) -> tuple[dict[str, torch.Tensor], str, float]:
    if checkpoint.get("model_config") != MODEL_CONFIG:
        raise ValueError("checkpoint architecture metadata does not match GraphEvaluator v1")
    baseline_id = checkpoint.get("baseline_checkpoint_id")
    if not isinstance(baseline_id, str) or len(baseline_id) != 64:
        raise ValueError("checkpoint must contain a 64-character baseline checkpoint SHA-256")
    try:
        bytes.fromhex(baseline_id)
    except ValueError as error:
        raise ValueError("baseline checkpoint identifier must be hexadecimal") from error

    residual_multiplier = checkpoint.get("residual_multiplier")
    if (
        isinstance(residual_multiplier, bool)
        or not isinstance(residual_multiplier, (int, float))
        or not math.isfinite(float(residual_multiplier))
        or float(residual_multiplier) <= 0.0
    ):
        raise ValueError("checkpoint residual_multiplier must be positive and finite")
    try:
        serialized_multiplier = struct.unpack(
            "<f", struct.pack("<f", float(residual_multiplier))
        )[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("checkpoint residual_multiplier is outside float32 range") from error
    if not math.isfinite(serialized_multiplier) or serialized_multiplier <= 0.0:
        raise ValueError("checkpoint residual_multiplier is not representable as positive float32")

    state = checkpoint.get("graph_state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint is missing graph_state_dict")
    expected_names = {name for name, _ in TENSOR_ORDER}
    if set(state) != expected_names:
        missing = sorted(expected_names - set(state))
        extra = sorted(set(state) - expected_names)
        raise ValueError(f"checkpoint tensor names mismatch; missing={missing}, extra={extra}")

    validated: dict[str, torch.Tensor] = {}
    parameter_count = 0
    for name, expected_shape in TENSOR_ORDER:
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"checkpoint tensor {name} is not a torch.Tensor")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"checkpoint tensor {name} must have shape {expected_shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise TypeError(f"checkpoint tensor {name} must be float32, got {tensor.dtype}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"checkpoint tensor {name} contains non-finite values")
        validated[name] = tensor.detach().cpu().contiguous()
        parameter_count += tensor.numel()

    if parameter_count != PARAMETER_COUNT:
        raise ValueError(f"GraphEvaluator v1 parameter count mismatch: {parameter_count}")
    return validated, baseline_id.lower(), serialized_multiplier


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.numpy()
    little_endian = np.asarray(array, dtype="<f4", order="C")
    return little_endian.tobytes(order="C")


def _build_export_bytes(
    state: dict[str, torch.Tensor], baseline_id: str, residual_multiplier: float
) -> bytes:
    baseline_bytes = baseline_id.encode("ascii")
    if len(baseline_bytes) > 0xFFFFFFFF:
        raise ValueError("baseline identifier is too long")

    output = bytearray(
        FILE_HEADER.pack(
            MAGIC,
            FORMAT_VERSION,
            ARCHITECTURE_VERSION,
            FLOAT32_CODE,
            len(TENSOR_ORDER),
            PARAMETER_COUNT,
            residual_multiplier,
            len(baseline_bytes),
        )
    )
    output.extend(baseline_bytes)
    for name, expected_shape in TENSOR_ORDER:
        name_bytes = name.encode("ascii")
        tensor = state[name]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"tensor {name} has unexpected dimensions")
        if len(name_bytes) > 0xFFFF or tensor.ndim > 0xFF:
            raise ValueError(f"tensor descriptor is too large for {name}")
        output.extend(TENSOR_HEADER.pack(len(name_bytes), tensor.ndim))
        output.extend(name_bytes)
        for dimension in tensor.shape:
            output.extend(struct.pack("<I", dimension))
        output.extend(_tensor_bytes(tensor))

    output.extend(hashlib.sha256(output).digest())
    return bytes(output)


def _reference_inputs(fen: str) -> dict[str, np.ndarray]:
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError(f"invalid reference position: {fen}")

    square_features = np.zeros((64, 16), dtype=np.float32)
    attack_adj = np.zeros((64, 64), dtype=np.float32)
    defence_adj = np.zeros((64, 64), dtype=np.float32)
    for source in chess.SQUARES:
        piece = board.piece_at(source)
        if piece is None:
            square_features[source, 0] = 1.0
            continue

        identity = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
        square_features[source, identity] = 1.0
        color_control_index = 13 if piece.color == chess.WHITE else 14
        for target in board.attacks(source):
            square_features[target, color_control_index] += 1.0 / 8.0
            target_piece = board.piece_at(target)
            if target_piece is not None:
                adjacency = defence_adj if target_piece.color == piece.color else attack_adj
                adjacency[target, source] = 1.0

    global_features = np.asarray(
        [
            1.0 if board.turn == chess.WHITE else -1.0,
            float("K" in fen.split()[2]),
            float("Q" in fen.split()[2]),
            float("k" in fen.split()[2]),
            float("q" in fen.split()[2]),
        ],
        dtype=np.float32,
    )
    if board.ep_square is not None:
        square_features[board.ep_square, 15] = 1.0
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    if white_king is None or black_king is None:
        raise ValueError("reference position must contain both kings")

    return {
        "square_features": square_features,
        "global_features": global_features,
        "attack_adj": attack_adj,
        "defence_adj": defence_adj,
        "white_king_square": np.asarray(white_king, dtype=np.int64),
        "black_king_square": np.asarray(black_king, dtype=np.int64),
    }


def write_reference_vectors(
    model: GraphEvaluator,
    output_path: Path,
    *,
    export_sha256: str,
    baseline_checkpoint_id: str,
    residual_multiplier: float,
) -> None:
    model = model.to("cpu").eval()
    cases = []
    with torch.inference_mode():
        for name, fen in REFERENCE_POSITIONS:
            inputs = _reference_inputs(fen)
            tensors = {
                key: torch.from_numpy(value).unsqueeze(0)
                for key, value in inputs.items()
            }
            expected_output = model(**tensors).item()
            cases.append(
                {
                    "name": name,
                    "fen": fen,
                    "inputs": {key: value.tolist() for key, value in inputs.items()},
                    "expected_graph_output": expected_output,
                }
            )
    document = {
        "architecture_version": ARCHITECTURE_VERSION,
        "export_sha256": export_sha256,
        "baseline_checkpoint_id": baseline_checkpoint_id,
        "residual_multiplier": residual_multiplier,
        "adjacency_convention": "adjacency[target][source] = 1",
        "output_units": "white-perspective residual in baseline target units",
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")

    binary_path = output_path.with_suffix(".bin")
    with binary_path.open("wb") as output:
        output.write(REFERENCE_HEADER.pack(REFERENCE_MAGIC, len(cases)))
        for case in cases:
            fen_bytes = case["fen"].encode("ascii")
            inputs = case["inputs"]
            output.write(REFERENCE_CASE_PREFIX.pack(len(fen_bytes)))
            output.write(fen_bytes)
            for name in (
                "square_features",
                "global_features",
                "attack_adj",
                "defence_adj",
            ):
                tensor = np.asarray(inputs[name], dtype="<f4", order="C")
                output.write(tensor.tobytes(order="C"))
            output.write(REFERENCE_KINGS.pack(
                int(inputs["white_king_square"]), int(inputs["black_king_square"])
            ))
            output.write(struct.pack("<f", case["expected_graph_output"]))


def export_checkpoint(checkpoint_path: Path, output_path: Path) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state, baseline_id, residual_multiplier = _validate_checkpoint(checkpoint)
    payload = _build_export_bytes(state, baseline_id, residual_multiplier)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_bytes(payload)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    model = GraphEvaluator()
    model.load_state_dict(state, strict=True)
    reference_path = output_path.with_suffix(".reference.json")
    write_reference_vectors(
        model,
        reference_path,
        export_sha256=hashlib.sha256(payload).hexdigest(),
        baseline_checkpoint_id=baseline_id,
        residual_multiplier=residual_multiplier,
    )
    return reference_path


def load_exported_model(
    path: Path, device: torch.device | str = "cpu"
) -> tuple[GraphEvaluator, dict[str, Any]]:
    payload = path.read_bytes()
    minimum_size = FILE_HEADER.size + 32
    if len(payload) < minimum_size:
        raise ValueError("export file is truncated")
    content, expected_checksum = payload[:-32], payload[-32:]
    if hashlib.sha256(content).digest() != expected_checksum:
        raise ValueError("export payload checksum mismatch")

    offset = 0
    (
        magic,
        format_version,
        architecture_version,
        dtype_code,
        tensor_count,
        parameter_count,
        residual_multiplier,
        baseline_length,
    ) = FILE_HEADER.unpack_from(content, offset)
    offset += FILE_HEADER.size
    if magic != MAGIC:
        raise ValueError("invalid GraphEvaluator export magic")
    if format_version != FORMAT_VERSION:
        raise ValueError(f"unsupported export format version: {format_version}")
    if architecture_version != ARCHITECTURE_VERSION:
        raise ValueError(f"unsupported GraphEvaluator architecture: {architecture_version}")
    if dtype_code != FLOAT32_CODE:
        raise ValueError(f"unsupported export dtype code: {dtype_code}")
    if tensor_count != len(TENSOR_ORDER) or parameter_count != PARAMETER_COUNT:
        raise ValueError("export tensor or parameter count does not match GraphEvaluator v1")
    if not math.isfinite(residual_multiplier) or residual_multiplier <= 0.0:
        raise ValueError("export residual multiplier must be positive and finite")
    if offset + baseline_length > len(content):
        raise ValueError("export baseline identifier is truncated")
    try:
        baseline_id = content[offset:offset + baseline_length].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("export baseline identifier is not ASCII") from error
    offset += baseline_length
    if len(baseline_id) != 64:
        raise ValueError("export baseline identifier must be a SHA-256 hex string")
    try:
        bytes.fromhex(baseline_id)
    except ValueError as error:
        raise ValueError("export baseline identifier is not hexadecimal") from error

    tensors: dict[str, torch.Tensor] = {}
    actual_parameters = 0
    for expected_name, expected_shape in TENSOR_ORDER:
        if offset + TENSOR_HEADER.size > len(content):
            raise ValueError("export tensor descriptor is truncated")
        name_length, rank = TENSOR_HEADER.unpack_from(content, offset)
        offset += TENSOR_HEADER.size
        descriptor_size = name_length + rank * 4
        if offset + descriptor_size > len(content):
            raise ValueError("export tensor descriptor is truncated")
        try:
            name = content[offset:offset + name_length].decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("export tensor name is not ASCII") from error
        offset += name_length
        dimensions = tuple(
            struct.unpack_from("<I", content, offset + index * 4)[0]
            for index in range(rank)
        )
        offset += rank * 4
        if name != expected_name or dimensions != expected_shape:
            raise ValueError(
                f"export tensor descriptor mismatch: {name} {dimensions}; "
                f"expected {expected_name} {expected_shape}"
            )
        element_count = math.prod(dimensions)
        byte_count = element_count * 4
        if offset + byte_count > len(content):
            raise ValueError(f"export tensor payload is truncated: {name}")
        tensor_array = np.frombuffer(content, dtype="<f4", count=element_count, offset=offset)
        if not np.isfinite(tensor_array).all():
            raise ValueError(f"export tensor contains non-finite values: {name}")
        tensors[name] = torch.from_numpy(tensor_array.copy().reshape(dimensions))
        offset += byte_count
        actual_parameters += element_count

    if actual_parameters != parameter_count:
        raise ValueError("export parameter count does not match tensor payload")
    if offset != len(content):
        raise ValueError("export contains unexpected trailing data")

    model = GraphEvaluator()
    model.load_state_dict(tensors, strict=True)
    model.to(device).eval()
    metadata = {
        "format_version": format_version,
        "architecture_version": architecture_version,
        "dtype": "float32",
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "residual_multiplier": residual_multiplier,
        "baseline_checkpoint_id": baseline_id,
    }
    return model, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    reference_path = export_checkpoint(arguments.checkpoint, arguments.output)
    print(f"Exported GraphEvaluator to {arguments.output}")
    print(f"Reference vectors written to {reference_path}")


if __name__ == "__main__":
    main()