# Bermuda GNN — M1 Development Guideline
## Position-to-Graph Feature Encoder

**Objective:** Implement a deterministic C++ component that converts a chess position into the fixed-size graph representation required by the experimental Bermuda GNN.

**M1 does not implement neural-network inference or training.**

### 1. Scope

Implement:

- 64 square nodes.
- 16 features per square.
- Five global position features.
- Directed attack relationships.
- Directed defence relationships.
- White and black king-square indices.
- Unit tests for feature extraction and graph construction.

Do not implement:

- Neural-network code.
- Tensor libraries.
- Attention.
- X-ray relationships.
- Policy or WDL features.
- Incremental graph updates.
- Search integration.
- SIMD optimizations.

### 2. First task: inspect Bermuda

Before writing code, identify:

1. The `Position` class and how pieces are accessed.
2. Square indexing and coordinate conventions.
3. Piece and color enumerations.
4. Bitboard representation and helpers.
5. Existing pawn, knight and king attack generation.
6. Existing bishop, rook and queen attack generation.
7. How occupancy and piece bitboards are accessed.
8. FEN parsing and existing test infrastructure.

Prefer existing engine primitives. Do not duplicate attack generation without a demonstrated need.

Produce a short implementation plan identifying the relevant files, classes and functions before making changes.

### 3. Data structures

Use fixed-size storage.

Suggested definitions (adapt names and types to Bermuda's conventions):

```cpp
struct GraphFeatures
{
    std::array<std::array<float, 16>, 64> squares{};
    std::array<float, 5> global{};

    std::array<uint64_t, 64> attack_sources{};
    std::array<uint64_t, 64> defence_sources{};

    uint8_t white_king_square = 0;
    uint8_t black_king_square = 0;
};
```

The adjacency convention is:

```cpp
attack_sources[target] |= (1ULL << source);
defence_sources[target] |= (1ULL << source);
```

A set bit identifies a source square whose piece attacks the target square.

Square indexing must exactly match Bermuda's existing indexing.

### 4. Square feature layout

Each square contains 16 features.

| Index | Feature |
|---:|---|
| 0 | Empty |
| 1 | White pawn |
| 2 | White knight |
| 3 | White bishop |
| 4 | White rook |
| 5 | White queen |
| 6 | White king |
| 7 | Black pawn |
| 8 | Black knight |
| 9 | Black bishop |
| 10 | Black rook |
| 11 | Black queen |
| 12 | Black king |
| 13 | White control count |
| 14 | Black control count |
| 15 | En-passant target indicator |

The first 13 features are one-hot encoded.

For example, a white knight on e4 produces:

```text
[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
```

Control counts:

- Count all white pieces geometrically attacking the square.
- Count all black pieces geometrically attacking the square.
- Include attacks on empty squares.
- Include attacks on occupied squares.
- Include pinned pieces.
- Include kings.
- Normalize each count by dividing by 8.

The en-passant feature is 1 only on the current en-passant target square. Otherwise it is 0.

Do not use legal move counts as attack counts.

### 5. Global features

Use the following order:

| Index | Feature | Encoding |
|---:|---|---|
| 0 | Side to move | White = +1, Black = -1 |
| 1 | White kingside castling | 0 or 1 |
| 2 | White queenside castling | 0 or 1 |
| 3 | Black kingside castling | 0 or 1 |
| 4 | Black queenside castling | 0 or 1 |

Do not infer side to move or castling rights from board occupancy.

### 6. Graph construction

For every occupied source square:

1. Determine its piece and color.
2. Generate its geometric attack bitboard using the current position occupancy.
3. Iterate over the attacked squares.
4. Increment the appropriate color's control count on every attacked square.
5. If the target is occupied:
   - Friendly occupant: add a defence edge.
   - Enemy occupant: add an attack edge.
6. If the target is empty, do not add an attack or defence edge in this initial design. Its control information is retained in the square features.

Use the convention:

```cpp
A[target][source] = 1;
```

Do not confuse geometric attacks with legal moves.

#### Piece-specific considerations

**Pawns**
- White pawns attack diagonally forward.
- Black pawns attack diagonally forward in the opposite direction.
- Do not count forward movement squares as pawn attacks.

**Knights**
- Use the normal eight-square attack pattern.
- Occupancy does not block knight attacks.

**Kings**
- Use adjacent-square attack patterns.
- Do not filter squares because the king would be unable to legally move there.

**Bishops, rooks and queens**
- Generate attacks using the current occupancy.
- Include the first occupied square along a ray.
- Stop beyond that blocker.
- Do not exclude friendly blockers from the attack map.

### 7. Interface

Keep feature extraction independent of neural inference.

Suggested interface:

```cpp
class GraphEncoder
{
public:
    void encode(
        const Position& position,
        GraphFeatures& output
    ) const;
};
```

Adapt this to Bermuda's existing namespace and coding conventions.

Requirements:

- No heap allocation during encoding.
- No changes to the position.
- No move making or unmaking.
- No dependence on search state.
- Deterministic output for the same position.
- Output must be fully overwritten on every call, including when reusing an existing `GraphFeatures` object.

### 8. Unit tests

Use FEN-based tests if Bermuda already supports them.

At minimum, test:

1. Initial chess position.
2. Empty squares and occupied squares.
3. White and black pawn attack directions.
4. Knight attacks.
5. King attacks.
6. Bishop attacks.
7. Rook attacks.
8. Queen attacks.
9. Sliding-piece blockers.
10. Friendly occupied targets creating defence edges.
11. Enemy occupied targets creating attack edges.
12. Empty squares receiving control counts.
13. Pinned pieces still contributing attacks.
14. Side-to-move encoding.
15. All four castling-right features.
16. En-passant target encoding.
17. Repeated encoding into the same output object.

Include a test in which a slider's first blocker is friendly and another in which it is an enemy piece.

### 9. Important invariants

For every encoded position:

- Exactly 64 nodes exist.
- Every square has exactly one piece-identity feature set.
- All other piece-identity features are zero.
- Control counts are non-negative.
- The control-count features equal independently calculated attack counts.
- No source bit refers to an unoccupied square.
- Every edge connects two valid board squares.
- Every attack edge targets an enemy-occupied square.
- Every defence edge targets a friendly-occupied square.
- No sliding attack passes through an occupied blocker.
- The original position remains unchanged.

### 10. Completion criteria

M1 is finished when:

- The encoder compiles with the existing project.
- All tests pass.
- Attack and defence relationships are verified against known positions.
- Repeated encoding is deterministic.
- No existing move-generation or search behaviour has been modified.
- A small debug representation can print each square's piece, control counts and incoming attack/defence sources.

Do not optimize prematurely. Correctness and a stable feature contract are the objectives of this milestone.