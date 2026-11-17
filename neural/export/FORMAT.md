# Bermuda GraphEvaluator Export Format

`BMGRPH01` is the version-1, little-endian binary format written by
`neural.export.graph_export`. It stores every learned GraphEvaluator parameter
as IEEE-754 float32. The format is independent of PyTorch and is intended for
future C++ readers; this task does not provide C++ inference.

## File Layout

All integer fields are unsigned and little-endian. The fixed 40-byte file header
uses the `struct` layout `<8sIIIIQfI`:

| Field | Type | Value / meaning |
|---|---|---|
| magic | 8 bytes | ASCII `BMGRPH01` |
| format version | u32 | `1` |
| architecture version | u32 | `1` (M2 GraphEvaluator) |
| dtype code | u32 | `1` means IEEE-754 float32 |
| tensor count | u32 | `14` |
| parameter count | u64 | `14561` scalar values |
| residual multiplier | f32 | multiplier from the training checkpoint |
| baseline ID length | u32 | byte length of the following identifier |

The baseline checkpoint identifier follows as ASCII hexadecimal SHA-256 text
(64 bytes for M3 checkpoints). Each tensor then appears in the canonical order
below. A tensor descriptor is `<HB`: name length as u16, rank as u8, ASCII name,
then `rank` u32 dimensions. The payload is contiguous row-major float32 in
little-endian order. The file ends with 32 raw bytes containing SHA-256 of every
preceding byte in the file.

## Tensor Order

| Order | Name | Dimensions |
|---:|---|---|
| 1 | `square_projection.weight` | `[32,16]` |
| 2 | `square_projection.bias` | `[32]` |
| 3 | `square_embedding.weight` | `[64,32]` |
| 4 | `global_projection.weight` | `[32,5]` |
| 5 | `global_projection.bias` | `[32]` |
| 6 | `self_transform.weight` | `[32,32]` |
| 7 | `attack_transform.weight` | `[32,32]` |
| 8 | `defence_transform.weight` | `[32,32]` |
| 9 | `layer_norm.weight` | `[32]` |
| 10 | `layer_norm.bias` | `[32]` |
| 11 | `value_head.0.weight` | `[64,133]` |
| 12 | `value_head.0.bias` | `[64]` |
| 13 | `value_head.2.weight` | `[1,64]` |
| 14 | `value_head.2.bias` | `[1]` |

Matrix values serialize in PyTorch's row-major weight order: output dimension
first, input dimension second. Embedding rows are square indices `0..63`. No
optimizer state or non-parameter training state is exported.

## Reference Vectors

Every export also writes `<export-stem>.reference.json`. The sidecar contains
the export SHA-256, baseline checkpoint identifier, residual multiplier, and
five FEN cases covering opening, middlegame, endgame, en-passant/castling, and a
pinned rook. Each case includes its full `[64,16]`, `[5]`, `[64,64]`, and
`[64,64]` graph inputs, the two king-square indices, and the expected raw `[1,1]`
GraphEvaluator residual. Adjacency matrices use `adjacency[target][source] = 1`.
The expected output is before multiplying by the header's residual multiplier.

Every export also writes `<export-stem>.reference.bin` with magic `BMGREF01`.
It starts with `<8sI` (magic and u32 case count). Each case contains a u32 FEN
byte length, ASCII FEN, then little-endian float32 arrays in this order:
`[64,16]` square features, `[5]` global features, `[64,64]` attack adjacency,
`[64,64]` defence adjacency; it ends with two u32 king indices and one float32
expected graph output. This compact form is consumed by the C++ parity runner.

The reference inputs follow Bermuda's M1 feature order and little-endian
rank-file square indices (`a1 = 0`). Consumers should check the file checksum,
versions, declared dimensions, and tensor order before using any weights.