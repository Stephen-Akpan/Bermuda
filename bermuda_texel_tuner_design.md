# Bermuda Texel Tuner — Design and Implementation Specification

**Project:** Bermuda Chess Engine  
**Component:** Texel evaluation tuner  
**Status:** Implementation design  
**Language:** C++17  
**Build system:** xmake  
**Primary interface:** C++ library + command-line executable  
**Tuning method:** Texel-style logistic objective optimized with SPSA  
**Evaluation model:** Handcrafted evaluation (HCE), parameterized through `Eval::Parameters`

---

## 1. Purpose

The purpose of this refactor is to turn Bermuda's current single-file Texel tuner into a maintainable, reusable tuning subsystem without changing the tuning mathematics during the initial migration.

The tuner must:

1. Import positions and game results from PGN or prepared datasets.
2. Keep all positions from a game in the same dataset split.
3. evaluate positions using arbitrary `Eval::Parameters`, without mutating Bermuda's active global evaluation parameters.
4. Minimize a Texel logistic mean-squared-error objective.
5. Optimize selected evaluation parameters using SPSA.
6. retain the best candidate according to validation loss.
7. save enough state to resume an experiment deterministically.
8. produce identifiable, reproducible experiment outputs.
9. remain independent of Bermuda's UCI executable and playing-strength test harness.

The design deliberately separates **data preparation**, **optimization**, and **playing-strength testing**. A lower validation MSE is evidence that a candidate fits the chosen game data better; it is not proof that the engine is stronger. TriangleTest remains the independent playing-strength gate.

## 2. Existing implementation: baseline to preserve

The latest `tools/texel_tuner.cpp` is the behavioral reference for the first migration. It currently contains the following functionality:

- PGN tokenization, tag extraction, SAN move reconstruction and pre-move FEN extraction.
- Support for PGN comments, semicolon comments, nested variations, NAGs, castling, promotions, SAN disambiguation and FEN-started games.
- Dataset loading from rows formatted as `GAME_ID RESULT FEN`.
- White-perspective results: `1-0` / `1`, `1/2-1/2` / `0.5`, `0-1` / `0`.
- Reproducible game-level train/validation splitting.
- Prepared train and validation dataset files.
- Parameter conversion between `Eval::Parameters` and a 19-element array.
- Logistic prediction and MSE objective.
- SPSA with a random ±1 perturbation direction, minibatches, decaying perturbation and learning rate.
- Parameter-family selection (`all`, `material`, `piece-values`).
- Validation-best parameter retention.
- Checkpoint save/resume, including the `std::mt19937_64` state and a backup file.
- Dataset fingerprinting.
- Loading initial parameters and saving the best candidate.

**Migration rule:** first extract and reorganize this behavior without changing its mathematical behavior. Improvements described later in this document should be implemented only after the extraction has a regression baseline.

### 2.1 Existing evaluation parameter vector

The tuner currently operates on 19 parameters in this exact order:

| Index | Name | Meaning |
|---:|---|---|
| 0 | `material` | Overall material component multiplier |
| 1 | `piece_square` | Piece-square evaluation multiplier |
| 2 | `pawn_structure` | Pawn-structure multiplier |
| 3 | `passed_pawns` | Passed-pawn multiplier |
| 4 | `mobility` | Mobility multiplier |
| 5 | `activity` | Activity multiplier |
| 6 | `threats` | Threat evaluation multiplier |
| 7 | `king_safety` | King-safety multiplier |
| 8 | `tempo` | Tempo multiplier |
| 9 | `pawn_mg` | Pawn material multiplier, middlegame |
| 10 | `knight_mg` | Knight material multiplier, middlegame |
| 11 | `bishop_mg` | Bishop material multiplier, middlegame |
| 12 | `rook_mg` | Rook material multiplier, middlegame |
| 13 | `queen_mg` | Queen material multiplier, middlegame |
| 14 | `pawn_eg` | Pawn material multiplier, endgame |
| 15 | `knight_eg` | Knight material multiplier, endgame |
| 16 | `bishop_eg` | Bishop material multiplier, endgame |
| 17 | `rook_eg` | Rook material multiplier, endgame |
| 18 | `queen_eg` | Queen material multiplier, endgame |

The parameter order is part of the experiment/checkpoint compatibility contract. Do not silently reorder it. Prefer named parameter descriptors and explicit conversion functions over relying on array indices throughout the code.

The current implementation clamps indices 0–8 to `[0.0, 3.0]`, and indices 9–18 to `[0.7, 1.3]`.

**Identifiability warning:** `material` multiplies the material component after the piece-type MG/EG multipliers have been applied. Therefore the effective piece values depend on products such as `material * knight_mg`. Simultaneously tuning the global material multiplier and all piece-value multipliers introduces redundant degrees of freedom. Parameter-family definitions should prevent this by default: tune global material separately from piece values, or hold one group fixed.

## 3. Repository layout

Keep the existing engine layout intact. Add a tuner subdirectory under `tools/`.

```text
Bermuda/
├── engine/
│   └── ...                         # Existing engine code
├── binary/
│   └── ...                         # Existing UCI executable
├── tests/
│   └── ...                         # Existing engine tests
├── tools/
│   ├── magic_gen.cpp
│   ├── texel_tuner.cpp             # Thin CLI entry point
│   └── texel_tuner/
│       ├── include/
│       │   └── bermuda_tuner/
│       │       ├── types.h
│       │       ├── config.h
│       │       ├── dataset.h
│       │       ├── pgn_parser.h
│       │       ├── dataset_split.h
│       │       ├── parameter_space.h
│       │       ├── objective.h
│       │       ├── optimizer.h
│       │       ├── checkpoint.h
│       │       ├── experiment.h
│       │       └── report.h
│       └── src/
│           ├── dataset/
│           │   ├── dataset.cpp
│           │   ├── pgn_parser.cpp
│           │   └── dataset_split.cpp
│           ├── objective/
│           │   └── texel_objective.cpp
│           ├── optimizer/
│           │   ├── parameter_space.cpp
│           │   ├── spsa.cpp
│           │   └── checkpoint.cpp
│           └── experiment/
│               ├── experiment.cpp
│               └── report.cpp
└── xmake.lua
```

The directory structure is a logical separation, not a requirement to create empty files in advance. Add files as their responsibilities are extracted.

### 3.1 Dependency direction

```text
                 bermuda_engine
                 /      |       \
                /       |        \
           bermuda  bermuda_tests  bermuda_tuner_lib
                                         |
                                    bermuda_tuner
```

- `bermuda_engine` must not depend on tuner code.
- `bermuda_tuner_lib` depends on `bermuda_engine`.
- `bermuda_tuner` depends on `bermuda_tuner_lib`.
- The UCI executable and tests remain independent consumers of the engine library.
- The tuner library must not depend on the UCI executable or invoke it to evaluate positions.

## 4. xmake integration

The project currently uses C++17 and defines `bermuda_engine` as a static library. Preserve that configuration.

Replace the existing single-file tuner target with:

```lua
-- ============================================================================
-- Texel Tuner Library
-- ============================================================================

target("bermuda_tuner_lib")
    set_kind("static")
    add_files("tools/texel_tuner/src/**.cpp")
    add_headerfiles("tools/texel_tuner/include/**.h")
    add_includedirs("tools/texel_tuner/include", {public = true})
    add_includedirs("engine")
    add_deps("bermuda_engine")
    set_group("Tools")

-- ============================================================================
-- Texel Tuner CLI
-- ============================================================================

target("bermuda_tuner")
    set_kind("binary")
    add_files("tools/texel_tuner.cpp")
    add_includedirs("tools/texel_tuner/include")
    add_deps("bermuda_tuner_lib")
    set_group("Tools")
```

Keep `magic_gen`, `bermuda`, `bermuda_tests`, and the existing custom tasks unchanged.

The library's public headers may include engine headers where necessary. Keep implementation-only engine includes private to the library when possible. If the public headers expose `Eval::Parameters`, the consumer must be able to resolve the engine include path; adjust xmake's public include visibility accordingly if required by the actual header dependencies.

Useful build checks:

```powershell
xmake
xmake build bermuda_engine
xmake build bermuda_tuner_lib
xmake build bermuda_tuner
xmake build bermuda_tests
xmake run bermuda_tests
```

---

## 5. Core data model

Use namespace:

```cpp
namespace Bermuda::Tuner {
    // ...
}
```

The exact class layout can evolve, but public types should be stable, explicit and testable.

### 5.1 Sample

A sample represents one position before a played move, with the final result of its source game.

```cpp
struct Sample {
    Position position;
    double result;              // White perspective: 1.0, 0.5, 0.0
    std::size_t game_index;     // Index into Dataset::games
};
```

The sample's result is always stored from White's perspective. Do not store a side-to-move result in the dataset.

### 5.2 Game metadata

The current implementation retains only generated game IDs. The refactor should introduce a metadata record so that future filtering and dataset analysis do not require reparsing the PGN.

```cpp
struct GameMetadata {
    std::string id;
    std::string event;
    std::string site;
    std::string date;
    std::string round;
    std::string white;
    std::string black;
    std::string result;
    std::string eco;
    std::string opening;
    std::string time_control;
    std::string termination;
    std::string variant;
    std::string source;
    // Optional normalized fields can be added later:
    // ratings, rated status, time-control category, OTB/online, etc.
};
```

Preserve original tag strings where practical. Do not infer missing metadata. A field should remain empty/unknown when the PGN does not provide it.

### 5.3 Dataset

```cpp
struct Dataset {
    std::vector<Sample> samples;
    std::vector<GameMetadata> games;

    std::string source_description;
    std::uint64_t content_fingerprint = 0;

    bool validate(std::string& error) const;
};
```

Invariants:

- Every sample's `game_index` is within `games`.
- Every sample result is finite and exactly one of `0.0`, `0.5`, `1.0`.
- Every position is valid and restorable.
- Every game represented in the dataset has a consistent result.
- A game is not split into multiple game records merely because it has multiple positions.
- Empty datasets and datasets with no usable positions are rejected.
- Training and validation datasets must contain distinct game IDs.

The in-memory dataset can initially continue storing `Position` objects as the current implementation does. A later binary format can store FEN or a compact position representation and reconstruct positions during loading. Do not prematurely optimize storage before measuring it.

### 5.4 Dataset identity

A fingerprint must identify the *actual experiment input*, not merely a sequence of FEN/result pairs.

Include, in a stable serialization, at least:

- Each game's stable ID.
- Each game's result.
- Every sample's game ID/reference.
- Each sample's FEN (including side to move, castling rights, en-passant square and clocks as represented by `Position::to_fen()`).
- The order or a canonical ordering policy.
- Relevant metadata used by filtering.
- Dataset format/version.

Use a documented hash algorithm. The existing FNV-1a-like 64-bit implementation is adequate for accidental mismatch detection but is not a cryptographic integrity guarantee. If collision resistance becomes important, use a stronger digest. Do not describe a non-cryptographic hash as cryptographic.

The train split, validation split and final holdout must each have their own identity. Also compute a combined experiment dataset identity from the split identities and split configuration.

---

## 6. PGN import subsystem

### 6.1 Responsibilities

`pgn_parser` owns:

1. Reading PGN input.
2. Tokenizing tags and movetext.
3. Ignoring comments and side variations for mainline dataset generation.
4. Extracting game metadata.
5. Reconstructing legal moves from SAN.
6. Recording the FEN before each mainline move.
7. Assigning the completed game result to each recorded position.
8. Reporting parse errors with file and line context.

`dataset` owns the resulting samples and validation. The parser should not know about SPSA, checkpoints or evaluation weights.

### 6.2 Existing parser behavior to retain

The current parser handles:

- Brace comments.
- Semicolon-to-end-of-line comments.
- Nested parenthesized variations.
- PGN tags.
- Castling (`O-O`, `O-O-O`, and zero-based notation).
- Piece moves and pawn moves.
- File/rank SAN disambiguation.
- Captures and en passant.
- Promotions.
- Check/checkmate suffixes and common annotation suffixes.
- NAG tokens.
- FEN tags for non-standard starting positions.
- Result tag/movetext disagreement detection.

Keep these capabilities during extraction.

### 6.3 Parser hardening

The current parser is custom. It must be treated as a component requiring tests, not assumed to implement every PGN edge case.

Add tests for:

- Standard short and long games.
- Multiple games in one file.
- Missing result tags and `*` results.
- Result tag/movetext disagreement.
- FEN-started games.
- Castling on both sides.
- Promotions and underpromotions.
- En passant.
- Ambiguous SAN requiring file/rank disambiguation.
- Comments containing punctuation that resembles PGN tokens.
- Nested variations.
- Escaped quotes in tag values.
- CRLF and LF files.
- Malformed tags, malformed comments and malformed variations.
- Illegal SAN and ambiguous legal SAN.
- Games with no moves.
- Games with unknown/unfinished results.

Unknown-result games (`*`) should not be assigned a fabricated result. The current importer skips games without a recognized result. Retain that behavior unless a future explicit policy says otherwise.

### 6.4 Metadata extraction

Parse all tags into a tag map or structured metadata record. At minimum preserve:

`Event`, `Site`, `Date`, `Round`, `White`, `Black`, `Result`, `ECO`, `Opening`, `TimeControl`, `Termination`, `Variant`, `FEN`.

Do not hard-code assumptions about Hikaru's archive or any other source. Different PGN collections have inconsistent tags.

### 6.5 Import reporting

At completion, report:

- Games encountered.
- Games accepted.
- Games skipped (for example, unknown result).
- Games rejected due to parse errors.
- Positions generated.
- Positions per game statistics (minimum, median, maximum, mean if practical).
- Missing metadata counts.
- Parse warnings.

A strict mode should fail on malformed games. A future permissive mode may skip an invalid game and continue, but must report exactly what was skipped and why. Do not silently discard parse errors.

---

## 7. Dataset preparation and splitting

### 7.1 Game-level splitting

Never split positions randomly. Every position from one game must belong to exactly one split.

The current implementation already performs game-level splitting. Preserve this property.

Support three distinct partitions in the design:

- **Training:** used for SPSA objective calculations and parameter updates.
- **Validation:** used to select the best candidate and monitor overfitting.
- **Final test:** held back until experiment selection is complete; never used to tune hyperparameters or choose weights.

The current implementation has training and validation only. Final-test support is a planned extension, not a claim about current behavior.

### 7.2 Independent seeds

The current `Config` has one `seed`, and `--split-seed` currently writes into that same field. This means split randomness and optimizer randomness are not actually independent.

Refactor to:

```cpp
std::uint64_t split_seed = 1;
std::uint64_t optimizer_seed = 1;
```

CLI:

- `--split-seed N`
- `--seed N` (optimizer seed)

Changing the optimizer seed must not change the split. Changing the split seed must not change the optimizer's random sequence after the dataset has been fixed.

For prepared train/validation files, store split metadata in a manifest, including seed and game counts. Do not rely on remembering a command line from terminal history.

### 7.3 Stratification

Initially use reproducible random game-level splitting, matching the current implementation. Then consider optional stratification by result, time control and source. If stratifying, keep all positions from a game together and document the algorithm. Do not stratify by individual positions.

### 7.4 Source-specific experiments

Keep source collections separate initially so their effects can be measured:

1. Engine self-play dataset (the user's existing collection of roughly 2,060 engine games).
2. Hikaru classical/OTB games.
3. Hikaru rapid games.
4. Hikaru blitz games.
5. A deliberately constructed combined dataset.

The Hikaru archive is a practical human-game source, not ground-truth evaluation data. Human games contain practical errors, and the archive may mix online and OTB conditions and time controls. Preserve metadata and do not assume all games have the same quality or statistical meaning.

Do not simply replace engine games with GM games. Run controlled comparisons and retain separate experiment records.

---

## 8. Prepared dataset format and storage

### 8.1 Current text format

The existing text dataset format is:

```text
GAME_ID RESULT FEN
```

Example:

```text
game-0001 1-0 rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
game-0001 1-0 rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1
game-0002 0-1 rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
```

Retain text input compatibility. It is useful for debugging and inspecting prepared positions.

### 8.2 Binary dataset format (`.btd`)

A versioned binary format is a planned optimization. It should avoid repeated PGN parsing and reduce the cost of loading large datasets.

Requirements:

- Magic signature, e.g. `BMTD`.
- Explicit format version.
- Explicit endianness or a defined byte order.
- Header with counts and dataset identity.
- Game metadata table.
- Position records with game reference and result.
- Validation of lengths and offsets before allocation.
- No unchecked allocation from untrusted counts.
- Corruption detection (checksum or digest).
- Forward-compatible version rejection with a useful error.
- Atomic output creation.

Do not serialize raw C++ object memory for `Position`; its layout is not a stable file format. Serialize a defined representation (FEN initially, or a deliberate compact board-state schema).

Implement `.btd` only after the library extraction and regression tests are complete. Text remains the human-readable interchange format.

### 8.3 Storage policy

Large PGN archives and derived datasets can consume significant storage. Keep source PGNs, prepared datasets, checkpoints and reports in separate directories. Avoid creating duplicate full datasets for every experiment when a shared immutable dataset plus split manifests is sufficient.

Suggested layout:

```text
tuning/
├── sources/
├── datasets/
├── experiments/
│   └── <experiment-id>/
│       ├── config.json
│       ├── checkpoint.bin
│       ├── checkpoint.bin.bak
│       ├── best-params.txt
│       ├── report.json
│       └── progress.csv
└── candidates/
```

Do not automatically delete source data or the last known-good checkpoint.

---

## 9. Evaluation interface and parameter space

### 9.1 Evaluation call

The tuner must use the existing parameterized engine API:

```cpp
Eval::evaluate(sample.position, parameters);
```

Do not change global parameters on each evaluation. Do not route each position through UCI. Do not create a second implementation of Bermuda's evaluation terms inside the tuner.

Initialize engine tables once before importing or evaluating positions:

```cpp
BB::init();
Zobrist::init();
```

The initialization currently occurs in `main`. Move it to a clearly documented application initialization step, not into every objective evaluation.

### 9.2 Parameter conversion

Centralize conversion:

```cpp
ParameterVector to_vector(const Eval::Parameters&);
Eval::Parameters to_parameters(const ParameterVector&);
```

The mapping must be tested in both directions. A round-trip should preserve all 19 values within floating-point equality expectations.

### 9.3 Parameter descriptors

Replace index-specific conditionals with descriptors:

```cpp
struct ParameterDescriptor {
    std::string_view name;
    double minimum;
    double maximum;
    double default_value;
    ParameterGroup group;
};
```

Use a fixed, documented descriptor table matching the 19 parameter names and order.

Possible groups:

- `evaluation_terms`
- `material_global`
- `piece_values_mg`
- `piece_values_eg`

The descriptor table owns parameter names, bounds, defaults and group membership. `ParameterSpace` owns which descriptors are active in a particular experiment.

### 9.4 Parameter families

Preserve existing families for compatibility:

- `all`
- `material`
- `piece-values`

Clarify existing behavior:
- `material` currently enables only index 0, the global material multiplier.
- `piece-values` currently enables indices 9–18, all piece MG/EG multipliers.
- `all` enables all 19.

Add more granular families after the baseline extraction, for example:

- `positional` — indices 1–8.
- `piece-values-mg` — indices 9–13.
- `piece-values-eg` — indices 14–18.
- Individual evaluation terms such as `mobility`, `activity`, `threats`, `king-safety`, `passed-pawns`, `pawn-structure`, `piece-square`, `tempo`.

Prefer explicit named groups over string comparisons against raw indices.

### 9.5 Bounds and initialization

Keep existing bounds in the first migration. Validate every initial parameter for finiteness and bounds before optimization.

A future configuration may allow custom bounds, but bounds must be included in the experiment configuration and checkpoint identity.

Do not silently clamp an invalid initial parameter file and proceed. Report the parameter name and invalid value. Clamping is appropriate for optimizer-generated candidates, not necessarily for user-supplied configuration.

---

## 10. Texel objective

### 10.1 Prediction

The current implementation converts Bermuda's side-to-move evaluation to White's perspective:

```cpp
const double sign =
    sample.position.side_to_move() == WHITE ? 1.0 : -1.0;

const double centipawns =
    sign * Eval::evaluate(sample.position, parameters);
```

The predicted White score is:

\[
P = \frac{1}{1 + 10^{-K \cdot cp / 400}}
\]

Equivalent implementation:

```cpp
const double exponent = std::clamp(
    -K * centipawns * std::log(10.0) / 400.0,
    -60.0, 60.0);

const double prediction = 1.0 / (1.0 + std::exp(exponent));
```

The clamp protects `exp` from overflow. Retain the current default `K = 1.13` for the baseline.

### 10.2 Loss

For sample \(i\):

\[
e_i = (P_i - R_i)^2
\]

where:
- \(P_i\) is the predicted White score.
- \(R_i\) is the actual game result from White's perspective.

Current objective:

\[
MSE = \frac{1}{N}\sum_{i=1}^{N}(P_i-R_i)^2
\]

This is the exact baseline objective: every position has equal weight.

### 10.3 Weighting modes (planned)

Long games currently contribute more because they contain more positions. Add explicit weighting modes after baseline reproduction:

- `position`: current behavior; every position has equal weight.
- `game`: each game contributes equal total weight, so a game with many positions does not dominate a shorter game.

For game weighting, if game \(g\) has \(n_g\) included positions, each position receives weight proportional to \(1/n_g\). Normalize by total game weight so the objective remains a weighted mean.

Report the selected weighting mode. Do not change the default silently.

### 10.4 Numerical and input validation

- Reject empty index ranges.
- Reject invalid sample indices.
- Reject non-finite K or K <= 0.
- Reject non-finite evaluation values.
- Guard against invalid objective results.
- Return a structured error/status rather than silently producing NaN.
- Keep evaluation score units consistent with the logistic K convention.

### 10.5 Objective API

A possible interface:

```cpp
struct ObjectiveConfig {
    double logistic_k = 1.13;
    WeightingMode weighting = WeightingMode::Position;
};

class TexelObjective {
public:
    double evaluate(
        const Dataset& dataset,
        const IndexSet& indices,
        const Eval::Parameters& parameters) const;
};
```

The objective should be deterministic for fixed dataset, parameters and configuration.

---

## 11. SPSA optimizer

### 11.1 Current algorithm

The current optimizer uses:

- `std::mt19937_64`.
- A minibatch sampled by shuffling the training indices.
- A random ±1 direction for every enabled parameter.
- Two objective evaluations per iteration: plus and minus perturbations.
- A finite-difference SPSA gradient estimate.
- Decaying perturbation and learning rate.
- Clamping to parameter bounds.

Current schedules:

\[
c_t = \frac{0.1}{(t+1)^{0.101}}
\]

\[
a_t = \frac{0.5}{(A+t+1)^{0.602}}
\]

where:

\[
A = \max(10,\ 0.1 \cdot \text{iterations})
\]

The current gradient estimate for parameter \(i\) is:

\[
\hat g_i =
\frac{L(\theta+c_t\Delta)-L(\theta-c_t\Delta)}
{2c_t\Delta_i}
\]

and the update is:

\[
\theta_i \leftarrow \theta_i-a_t\hat g_i
\]

Disabled parameters have \(\Delta_i=0\) and are not updated.

**Do not alter these equations or constants during the initial extraction.**

### 11.2 SPSA configuration

```cpp
struct SPSAConfig {
    std::size_t iterations = 300;
    std::size_t batch_size = 4096;
    std::uint64_t seed = 1;
    double perturbation_a = 0.1;
    double perturbation_exponent = 0.101;
    double learning_rate_a = 0.5;
    double learning_rate_exponent = 0.602;
    double stability_fraction = 0.1;
    std::size_t report_interval = 10;
};
```

The defaults above reproduce the current hard-coded schedule. The implementation may initially keep constants internal, but centralizing them is necessary for complete experiment identity and later tuning.

### 11.3 Batch sampling

Current behavior shuffles all training indices each iteration and takes the first `batch_size` positions. If the batch size exceeds the training set size, the whole training set is used.

Preserve this behavior initially.

Later, consider sampling by game to avoid a minibatch being dominated by a few long games. If introduced, make the policy explicit and record it in the experiment configuration.

### 11.4 Optimizer state

```cpp
struct SPSAState {
    std::size_t completed_iterations = 0;
    ParameterVector current_weights;
    ParameterVector best_weights;

    double initial_validation_loss = 0.0;
    double best_validation_loss = 0.0;
    double current_training_loss = 0.0;

    std::mt19937_64 random;
};
```

The optimizer should not own PGN parsing or file formats. It receives an objective, parameter space and training index set.

### 11.5 Optimizer API

```cpp
class SPSAOptimizer {
public:
    OptimizationResult run(
        const TexelObjective& objective,
        const Dataset& dataset,
        const IndexSet& training,
        const IndexSet& validation,
        const ParameterSpace& parameter_space,
        const SPSAConfig& config,
        SPSAState& state,
        ProgressCallback callback = {});
};
```

The exact API can change, but keep responsibilities separate:
- optimizer updates current weights;
- experiment evaluates validation and chooses the best candidate;
- checkpoint component persists state.

This separation allows a future optimizer to be added without rewriting dataset or report code.

---

## 12. Validation and candidate selection

The current implementation evaluates validation loss every 10 iterations and at the final iteration. It stores weights only when validation MSE improves.

Retain that policy in the baseline extraction.

Requirements:

- Initial weights receive a baseline training and validation measurement.
- Candidate validation must use the same fixed validation set throughout an experiment.
- Best weights are separate from current optimizer weights.
- Save the best validation candidate, not automatically the final iteration.
- Report current and best validation losses separately.
- Record the iteration at which the best candidate was found.
- If validation loss never improves, the initial weights remain the best candidate.

### 12.1 Final holdout

Add a final-test split as a later milestone. It is not used:
- for SPSA updates;
- for choosing a best iteration;
- for choosing a parameter family;
- for selecting hyperparameters;
- for deciding which candidate to promote during iterative experimentation.

Evaluate the selected candidate on final test only after the experiment design and candidate have been fixed. If results are used to change the candidate, that holdout has become validation data and a new holdout is required for an unbiased final assessment.

### 12.2 Overfitting signals

Report:
- initial train loss;
- current train loss;
- initial validation loss;
- current validation loss;
- best validation loss;
- best iteration;
- train/validation loss gap.

A falling training loss with a worsening validation loss is a warning that the candidate is fitting the training games rather than generalizing.

---

## 13. Checkpoint and resume

### 13.1 Current checkpoint behavior

The existing checkpoint contains:
- magic/version (`TRIANGLE_OPTIMIZER 1`);
- dataset fingerprint;
- parameter family;
- iteration count;
- batch size;
- logistic K;
- completed iteration;
- best, initial and current losses;
- current weights;
- best weights;
- serialized `mt19937_64` state.

It writes a `.tmp` file, rotates the previous checkpoint to `.bak`, then renames the temporary file into place. Resume tries the primary checkpoint and then the backup.

Preserve this resilience, but improve the format and experiment identity.

### 13.2 New checkpoint contents

Use a versioned checkpoint containing:

**Identity**
- checkpoint magic and format version;
- tuner version / source revision if available;
- engine evaluation version or explicit evaluator identity;
- parameter schema version;
- dataset fingerprint;
- training, validation and final-test split fingerprints;
- split seed and split policy;
- source and filter description.

**Configuration**
- objective K;
- weighting mode;
- optimizer name and version;
- iterations and batch size;
- optimizer seed;
- SPSA schedule parameters;
- active parameter names;
- parameter bounds;
- validation/report interval.

**State**
- completed iteration;
- current parameter vector;
- best parameter vector;
- best iteration;
- initial/current/best losses;
- PRNG state;
- any optimizer-specific accumulators needed by future algorithms.

### 13.3 Resume validation

Resume must reject a checkpoint if any state-defining input differs. At minimum compare:
- dataset/split identity;
- parameter schema;
- active parameter set;
- bounds;
- optimizer;
- objective settings;
- SPSA schedule;
- batch size;
- optimizer seed.

A requested larger total iteration limit may be allowed as an explicit extension if the schedule remains mathematically consistent. Since the current stability constant depends on the configured total iteration count, changing the total can change the schedule. The simplest safe policy is to require an exact match initially.

Do not initialize a new state and accidentally overwrite checkpoint values before loading. Construct initial state, load checkpoint, validate it, then continue.

### 13.4 Atomic persistence

- Write to a temporary file in the same directory.
- Flush and close successfully before promotion.
- Preserve the last valid checkpoint as backup.
- On failed promotion, restore the backup where possible.
- Validate the complete checkpoint before accepting it.
- Do not delete both primary and backup after a failed write.
- Consider filesystem durability (`fsync`/platform equivalent) later if power-loss safety is required.

Checkpoint writes should happen at a configurable interval and on graceful shutdown if practical. A process killed without warning can only resume from the most recent completed checkpoint.

---

## 14. Experiment orchestration

`Experiment` coordinates the other modules. It should not contain PGN tokenization or implement the SPSA equations itself.

### 14.1 Experiment configuration

```cpp
struct ExperimentConfig {
    DatasetPaths datasets;
    ObjectiveConfig objective;
    SPSAConfig optimizer;
    ParameterSpaceConfig parameter_space;

    std::string output_directory;
    std::string initial_parameters_path;
    std::string checkpoint_path;

    bool resume = false;
};
```

A production config should be serializable to JSON or another stable, human-readable format.

### 14.2 Experiment lifecycle

1. Validate configuration.
2. Initialize engine tables.
3. Load initial evaluation parameters or use engine defaults.
4. Load or import datasets.
5. Validate game/sample references.
6. Build or load fixed train/validation/final-test splits.
7. Compute all dataset and split identities.
8. Construct parameter space and validate initial weights.
9. Calculate initial train and validation losses.
10. Load and validate checkpoint if resuming.
11. Run SPSA.
12. At each report interval, evaluate training and validation.
13. Retain best validation weights.
14. Persist checkpoint and progress report.
15. Save best candidate parameters.
16. Write final experiment report.
17. Leave playing-strength testing to TriangleTest.

### 14.3 Library API example

The intended use should be simple for another C++ program:

```cpp
#include <bermuda_tuner/experiment.h>

int main() {
    Bermuda::Tuner::ExperimentConfig config;
    config.optimizer.iterations = 1000;
    config.optimizer.batch_size = 4096;
    config.optimizer.seed = 42;
    config.output_directory = "tuning/experiments/run-001";

    Bermuda::Tuner::Experiment experiment(config);
    const auto result = experiment.run();

    return result.success ? 0 : 1;
}
```

This is illustrative. The actual constructors and result type should be designed around error handling and dataset ownership during implementation.

The library must not print directly to `std::cout` throughout its implementation. Provide a progress callback or logger interface. The CLI decides how progress is displayed.

---

## 15. CLI design

Keep `tools/texel_tuner.cpp` as the executable entry point. It should:
- parse command-line arguments;
- construct configuration;
- call the library;
- render errors and progress;
- return meaningful exit codes.

It should not contain PGN, objective, optimizer or checkpoint implementation.

### 15.1 Preserve current commands

The current CLI supports:

```text
--prepare-pgn FILE
--train-out FILE
--validation-out FILE
--validation-percent N
--split-seed N
--pgn FILE
--data FILE
--train-data FILE
--validation-data FILE
--out FILE
--init FILE
--checkpoint FILE
--resume
--family NAME
--iterations N
--batch N
--seed N
--k VALUE
--help
```

Preserve these options during the first extraction. Fix the split/optimizer seed collision so `--split-seed` and `--seed` are genuinely independent.

### 15.2 Planned CLI additions

Add only after the baseline CLI works:

```text
--test-data FILE
--weighting position|game
--report-dir DIR
--report-interval N
--checkpoint-interval N
--format text|btd
--metadata-filter KEY=VALUE
--list-parameters
--dry-run
```

Exact names can be adjusted, but document all accepted options and reject unknown options.

### 15.3 Input mode validation

Retain the current mutually exclusive input modes:
- one `--data FILE`;
- one `--pgn FILE`;
- both `--train-data FILE` and `--validation-data FILE`;
- preparation mode with `--prepare-pgn`, `--train-out` and `--validation-out`.

Extend the validation rules carefully when final-test and binary inputs are introduced. Do not allow contradictory input modes.

---

## 16. Experiment reports and progress logs

Each experiment should be self-describing and reproducible without relying on shell history.

### 16.1 `config.json`

Record:
- experiment ID and creation time;
- Bermuda revision/build identity if available;
- tuner version;
- engine evaluation identity;
- dataset source and fingerprints;
- game and position counts;
- split policy, split seed and split fingerprints;
- optimizer name and all optimizer settings;
- objective K and weighting mode;
- active parameters and bounds;
- initial parameter file identity;
- output paths.

### 16.2 `progress.csv`

Suggested columns:

```text
iteration,training_loss,validation_loss,best_validation_loss,best_iteration
```

Append one row per report interval. Make sure a resumed experiment does not duplicate or corrupt earlier progress rows.

### 16.3 `report.json`

Include:
- complete configuration;
- initial parameter vector;
- final current vector;
- best validation vector;
- initial and best train/validation losses;
- best iteration;
- elapsed time;
- samples/games evaluated;
- completion status;
- warnings and errors;
- final-test metrics, if a final-test dataset was used after selection.

### 16.4 Parameter output

Continue using Bermuda's existing `Eval::save_parameters` format for candidate weights. The output should contain all 19 parameters, including inactive parameters unchanged from their initial values.

Write candidate files atomically. Do not overwrite the baseline parameter file by default.

---

## 17. Error handling and observability

Use structured errors inside the library. A useful error should identify:
- operation;
- file path, where applicable;
- game ID and PGN line, where applicable;
- parameter name, where applicable;
- expected and actual checkpoint identity, where applicable.

Do not rely on `std::cerr` deep inside library code. The CLI or embedding application should decide how to display errors.

Suggested exit codes for the CLI:

| Code | Meaning |
|---:|---|
| 0 | Success |
| 1 | Experiment completed but returned a non-success status, if applicable |
| 2 | Invalid arguments, input data, configuration or checkpoint |
| 3 | File I/O or persistence failure |
| 4 | Internal evaluation/optimizer failure |

The exact mapping may be refined, but use consistent documented codes.

---

## 18. Testing strategy

The refactor is not complete when it compiles. It is complete when behavior is verified.

### 18.1 Unit tests

**Parameter space**
- 19 descriptors exist.
- Names and ordering match the engine parameter schema.
- Vector-to-parameters round trip.
- Family selection activates the expected parameters.
- Bounds and invalid values are handled correctly.

**PGN parser**
- Test cases listed in Section 6.3.
- Verify exact number of generated pre-move positions.
- Verify each generated FEN is the position before the corresponding SAN move.
- Verify result perspective and game grouping.

**Dataset**
- Valid and invalid row formats.
- Duplicate game IDs with consistent results.
- Conflicting results for the same game rejected.
- Invalid FEN rejected with line number.
- Game/sample references remain valid.
- Fingerprints change when game IDs, results, positions or relevant metadata change.

**Objective**
- White-to-move and Black-to-move sign conversion.
- Known score-to-probability examples.
- Draw target is 0.5.
- K validation.
- Position-weighted MSE matches a hand-calculated result.
- Game-weighted MSE matches a hand-calculated result when implemented.

**SPSA**
- Disabled parameters remain unchanged.
- Parameters remain within bounds.
- Same seed/config/dataset produces the same result.
- Empty or invalid batches fail safely.
- Best weights correspond to best validation loss.

**Checkpoint**
- Save and load round trip.
- PRNG state round trip.
- Resume continues with the same next random sequence.
- Mismatched dataset/config/schema is rejected.
- Corrupt primary checkpoint falls back to backup.
- Truncated checkpoint is rejected.
- Failed write does not destroy the last valid checkpoint.

### 18.2 Integration tests

Create a small deterministic PGN fixture containing several games. Run:

1. PGN import.
2. Game-level split.
3. Dataset save.
4. Dataset reload.
5. Objective evaluation.
6. A short SPSA run.
7. Checkpoint save.
8. Resume to completion.
9. Candidate parameter save.
10. Report generation.

Verify all expected files and identities.

### 18.3 Refactor regression test

Before changing optimization behavior, capture a baseline run from the current tuner:
- exact source dataset;
- exact split;
- initial parameters;
- family;
- iterations;
- batch size;
- optimizer seed;
- K;
- initial train/validation MSE;
- final/best weights;
- best validation MSE.

After extraction, repeat with the same compiler/build mode and inputs. Results should match exactly where the same floating-point operation order and PRNG sequence are preserved. If not, investigate before accepting the refactor.

---

## 19. TriangleTest integration and promotion policy

TriangleTest remains an external playing-strength test. Do not merge it into the Texel objective or make it a hidden optimizer signal.

Current TriangleTest process, as described by the project owner:
- Baseline and candidate play varied openings.
- Each opening is played twice, with colours reversed.
- A recent test used 250 opening pairs.
- A previous candidate showed an estimated +9 Elo but a lower confidence bound below zero, so it was not promoted.

Recommended candidate lifecycle:

```text
Baseline parameters
       |
       v
Texel tuning experiment
       |
       v
Best validation candidate
       |
       v
Final holdout evaluation
       |
       v
TriangleTest (paired openings, reversed colours)
       |
       +---- insufficient evidence / regression ---> Reject or retest
       |
       v
Candidate accepted
       |
       v
Promote parameters and preserve experiment artifacts
```

Promotion must be conservative:
- A lower Texel validation loss is not enough.
- A small positive Elo point estimate with a confidence interval crossing zero is inconclusive, not a proven improvement.
- Keep the baseline available.
- Record the exact engine binary, parameter files, opening suite, time control, hardware/thread settings and test result.
- Avoid changing engine code between baseline and candidate tests.

TriangleTest can later be automated by a separate orchestration script, but the tuner library should not own match scheduling.

---

## 20. Performance and future scaling

The optimizer currently evaluates two parameter candidates per iteration on a minibatch. Most work is repeated engine evaluation.

Priorities:

1. Preserve direct in-process evaluation.
2. Profile before adding concurrency.
3. Avoid repeated FEN parsing during every SPSA iteration; positions should be parsed once and retained in memory.
4. Consider compact binary datasets to reduce startup and storage cost.
5. Consider precomputed immutable position features only if profiling demonstrates a worthwhile gain and the features remain valid for all parameter vectors.
6. Add parallel evaluation only after verifying the engine's evaluation and position APIs are safe for concurrent use. `Eval::evaluate(position, parameters)` is parameterized, but that alone does not prove every underlying engine function is thread-safe.
7. Keep experiment state and random-number generation deterministic.
8. Do not make GPU or paid-cloud dependencies a requirement.

For free-cloud automation, keep the core tuner as a normal command-line program that can run on a local machine or a compatible free runner. Scheduling, job orchestration and artifact synchronization should remain outside the C++ library.

---

## 21. Implementation phases and acceptance criteria

### Phase 0 — Baseline capture

- Build the current tuner.
- Run a known dataset/configuration.
- Save command line, initial parameters, metrics, candidate output and checkpoint.
- Record compiler and build mode.

**Acceptance:** baseline artifacts exist and can be used for regression comparison.

### Phase 1 — Library extraction

- Create `bermuda_tuner_lib`.
- Move implementation out of `tools/texel_tuner.cpp`.
- Keep CLI behavior and option names.
- Keep current objective, SPSA schedule, bounds and validation interval.
- Make CLI a thin consumer.

**Acceptance:** both targets build; old commands still work; regression output matches.

### Phase 2 — Module separation

- Separate PGN parser.
- Separate dataset loader and writer.
- Separate objective.
- Separate parameter space.
- Separate SPSA.
- Separate checkpoint.
- Add unit tests for each.

**Acceptance:** modules have clear interfaces and tests; no tuner implementation remains in the CLI.

### Phase 3 — Experiment identity and reliability

- Separate split and optimizer seeds.
- Strengthen dataset and split fingerprints.
- Version checkpoint schema.
- Validate all state-defining configuration on resume.
- Add atomic candidate/report persistence.
- Add best iteration and progress logging.

**Acceptance:** interrupted run resumes deterministically; mismatched experiments cannot accidentally resume.

### Phase 4 — Metadata and source analysis

- Preserve PGN metadata.
- Add import statistics.
- Add metadata filters.
- Build separate engine, classical/OTB, rapid and blitz datasets.
- Save split manifests.

**Acceptance:** each experiment can be reproduced and its source cohort understood.

### Phase 5 — Objective and parameter-space extensions

- Add game-weighted objective.
- Add explicit positional and individual parameter families.
- Add parameter descriptor table and custom bounds if needed.
- Keep existing defaults unchanged unless a deliberate experiment changes them.

**Acceptance:** weighting and parameter-family behavior have hand-calculated tests and are fully recorded.

### Phase 6 — Binary dataset and reporting

- Add versioned `.btd` format.
- Add JSON config/report.
- Add CSV progress.
- Add storage and artifact policies.

**Acceptance:** a prepared dataset can be reused without reparsing PGN and every experiment is self-describing.

### Phase 7 — Candidate evaluation workflow

- Add final holdout support.
- Export best candidate in engine parameter format.
- Run TriangleTest externally.
- Record promotion decision.

**Acceptance:** a candidate cannot be promoted solely because its Texel loss improved.

---

## 22. Explicit non-goals

The first implementation does **not** include:

- NNUE training or inference.
- Neural-network evaluation.
- Rewriting Bermuda's evaluation terms.
- Changing the engine's UCI protocol.
- Replacing TriangleTest.
- Automatically promoting weights based on Texel MSE.
- Requiring Python for optimization.
- Requiring paid cloud services.
- Parallel evaluation before thread-safety is established.
- A generic plugin system for arbitrary optimizers.
- Automatic deletion of old datasets/checkpoints.

The architecture should leave room for future optimizers and automation without implementing speculative infrastructure now.

---

## 23. Final design principles

1. **Preserve behavior before improving it.** Extraction and algorithm changes are separate milestones.
2. **Keep dependencies one-way.** The tuner depends on Bermuda; Bermuda does not depend on the tuner.
3. **Use the engine's actual evaluation.** No duplicated evaluation implementation and no UCI subprocess loop.
4. **Split by game.** Never leak positions from one game across training and validation.
5. **Separate data fit from playing strength.** Texel loss selects candidates; TriangleTest evaluates engine strength.
6. **Make experiments reproducible.** Data, splits, seeds, parameters, bounds and optimizer settings are part of experiment identity.
7. **Make interruption safe.** Checkpoints must preserve optimizer and random state and reject incompatible resumes.
8. **Keep metadata.** Human games from different formats and time controls should not be treated as indistinguishable.
9. **Measure before optimizing infrastructure.** Keep the implementation simple until profiling justifies more complexity.
10. **Keep the first version implementable.** Do not block the library extraction on binary datasets, cloud scheduling or advanced statistics.

The first concrete coding task is Phase 0 followed by Phase 1: capture a baseline, create the library target, extract the existing code with its behavior intact, and prove that the refactored executable reproduces the baseline.
