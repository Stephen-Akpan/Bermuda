# Bermuda Chess Engine

A UCI-compatible chess engine written in C++, built from scratch without external dependencies.

## Features

- **Search**: Iterative-deepening PVS with aspiration windows, transposition-table cutoffs, null-move pruning, and late-move reductions
- **Move Ordering**: Transposition-table moves, static exchange evaluation, killer moves, and history scores
- **Board Representation**: Bitboards for fast move generation and evaluation
- **Evaluation**: Tapered handcrafted evaluation with material, piece placement, mobility, pawn structure, passed pawns, threats, and king safety
- **Transposition Table**: Caches previously evaluated positions to avoid redundant computation
- **Interface**: UCI (Universal Chess Interface) protocol

For a detailed description of the current evaluation, search, and time
management implementations, see the [technical overview](./ENGINE_TECHNICAL_OVERVIEW.md).

Browse the [Bermuda documentation website](./docs/index.html) for a visual guide
to building, running, and experimenting with the engine.

For repeatable data preparation, resumable Texel/SPSA tuning, and paired
Cutechess validation, see the [TriangleTest workflow](./TRIANGLE_TEST.md).

## Getting Started

### Prerequisites

- [xmake](https://xmake.io) build system
- A UCI-compatible chess GUI (e.g., [Arena](http://www.playwitharena.de/), [Cute Chess](https://cutechess.com/))

### Building

```bash
git clone https://github.com/yourusername/bermuda.git
cd bermuda
xmake
```

The compiled binary will be placed in the `build/` directory.

## Usage

Bermuda communicates via the UCI protocol and is designed to be used with a chess GUI. Simply point your GUI to the compiled binary and it will handle the rest.

## Running Tests

Tests are written using [Catch2](https://github.com/catchorg/Catch2). To run them:

```bash
xmake test
```

## Texel Evaluation Tuning

The `bermuda_tuner_lib` static library fits the evaluation's feature-family
weights with SPSA against game outcomes, and the standalone `bermuda_tuner`
executable provides its command-line interface. C++ consumers can include
`<bermuda_tuner/texel_tuner.h>`, populate `Bermuda::Tuner::Config`, and call
`Bermuda::Tuner::run`. The tuner does not depend on Stockfish source.

The library separates PGN parsing and dataset loading, parameter-space
conversion, the Texel objective, SPSA optimization, checkpoint persistence,
and experiment orchestration into independent modules under
`tools/texel_tuner/`.

Build the tuner and run it directly against a Cutechess PGN:

```powershell
xmake build bermuda_tuner
.\build\windows\x64\release\bermuda_tuner.exe --pgn games.pgn --out eval-params.txt
```

The PGN importer extracts each game's pre-move positions and labels them with
the final result. It handles standard SAN moves, comments, variations, and FEN
setup tags, and preserves PGN tags for filtering and later analysis. Unfinished
games without a result are skipped and counted in the import summary. Repeat
`--metadata-filter KEY=VALUE` to form source cohorts, for example
`--metadata-filter TimeControl=600+0`. Filters match exact tag values. You can
also prepare a dataset directly if you already have FENs and outcomes; each
non-comment line has this format:

```text
GAME_ID RESULT FEN
```

`RESULT` is the game's result from White's perspective: `1-0`, `1/2-1/2`,
`0-1`, or the equivalent numbers `1`, `0.5`, `0`. Include multiple positions
from each game with the same `GAME_ID`; the tuner splits by game, not by
position, to keep positions from one game out of both training and validation.
Every row for a game must have the same result.
Use many games and hold back separate games for a final check. The dataset
should be representative of Bermuda's play; evaluation labels from game
results are noisy.

Tuning options include `--iterations`, `--batch`, `--seed`, `--split-seed`,
`--k`, `--init`, `--family`, and `--weighting`. `--seed` controls SPSA
randomness and `--split-seed` controls the reproducible game-level split
independently. The default validation set is 10% of games. In addition to
nine feature-family scales, the tuner can adjust separate middlegame/endgame
material values for pawns, knights, bishops, rooks, and queens. Feature-family
scales are bounded to `[0, 3]`; piece-value scales are bounded to `[0.7, 1.3]`.
Available families include `all`, `material`, `positional`, `piece-values`,
`piece-values-mg`, `piece-values-eg`, and individual parameter names such as
`mobility`, `king-safety`, or `rook-mg`. Use `--weighting game` to give each
game equal influence instead of allowing longer games to contribute more
positions; the default `position` weighting preserves the original objective.

Prepare reusable splits in the versioned binary format (text remains the
default and human-readable option):

```powershell
.\build\windows\x64\release\bermuda_tuner.exe --prepare-pgn games.pgn `
  --train-out training.btd --validation-out validation.btd --format btd `
  --split-seed 42
```

`.btd` files use the `BMTD` signature, explicit version and little-endian
integer fields, FEN position records, retained game metadata, and an FNV-1a
64-bit accidental-corruption checksum. The checksum is not cryptographic.
Text prepared datasets retain metadata in backward-compatible comment records.
Both prepared formats produce a split manifest containing the seed, split
fingerprints, game IDs, and game/position counts.

Each tuning run writes `config.json`, `progress.csv`, and `report.json` to
`<output-path>.experiment` by default; override this with `--report-dir`.
Reports record the data and split identities, import statistics, active
parameters and bounds, objective weighting, losses, and best iteration.
Checkpoints and candidate parameter files are written through temporary files
with a backup retained. Keep source PGNs, prepared datasets, reports, and
checkpoints in separate directories; the tuner does not delete source data or
the last known-good checkpoint.

The tuner keeps the candidate with the lowest held-out validation error;
compare that error with the initial baseline, then validate promising weights
in paired Cutechess matches before adopting them.

Load the saved weights in the engine by setting the UCI option:

```text
setoption name EvalParams value eval-params.txt
```

An empty `EvalParams` value restores the built-in evaluation defaults. Invalid
parameter files are rejected and reported rather than silently ignored.

For tournament time controls, Bermuda reserves clock time and uses a
conservative sudden-death allocation; the UCI `MoveOverhead` option defaults
to 50 ms. If multiple games run concurrently or the machine is heavily loaded,
try increasing it to 100-200 ms. Match time losses should still be monitored
under the exact tournament concurrency and control you intend to use.

## Platform Support

| Platform | Status |
|----------|--------|
| Windows  | ✅ Tested |
| Linux    | ❓ Untested |
| macOS    | ❓ Untested |

## Roadmap

- [ ] ELO benchmarking against other engines
- [ ] Linux & macOS support

## License

All rights reserved. This project is not currently open for redistribution or commercial use.

---

> *Bermuda is a personal project with commercial ambitions. Contributions are not currently accepted.*
