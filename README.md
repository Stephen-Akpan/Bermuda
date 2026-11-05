# Bermuda

Bermuda is a UCI-compatible chess engine written in C++17. It is an ongoing
project for exploring engine search, evaluation, and playing strength.

## Features

- Bitboard-based position representation and move generation
- Iterative-deepening alpha-beta search with PVS and aspiration windows
- Transposition-table cutoffs, null-move pruning, futility pruning, and LMR
- Move ordering with TT moves, SEE, killer moves, and history scores
- Tapered handcrafted evaluation with material, piece-square, mobility, pawn structure, passed-pawn, threat, and king-safety terms
- UCI support for common chess GUIs and command-line match tools

For details on the current evaluation and search implementation, see the
[technical overview](ENGINE_TECHNICAL_OVERVIEW.md).

## Build

Requirements:

- [xmake](https://xmake.io)
- A C++17 compiler supported by xmake

From the repository root, build a release binary:

```sh
xmake f -m release
xmake build bermuda
```

xmake places the executable under `build/<platform>/<architecture>/release/`.
On Windows, the executable has an `.exe` extension.

## Run

Open the executable in a UCI-compatible GUI, or start it from a terminal and
send UCI commands. For example:

```text
uci
isready
position startpos
go depth 12
quit
```

The engine supports the UCI options `Hash`, `MoveOverhead`, `Clear Hash`, and
`EvalParams`. Use `uci` to see the options advertised by the running build.

## Tests and tuning

Build and run the engine and tuner tests with:

```sh
xmake build bermuda_tests
xmake build bermuda_tuner_tests
xmake test
```

The Texel/SPSA tuner and the paired tuning-and-strength workflow are documented
in [TRIANGLE_TEST.md](TRIANGLE_TEST.md). The tuner design and implementation
details are in [bermuda_texel_tuner_design.md](bermuda_texel_tuner_design.md).
Parameter tuning is experimental: validate candidate weights with paired games
before adopting them.

## Documentation

The [technical overview](ENGINE_TECHNICAL_OVERVIEW.md) describes the current
evaluation, search, score conventions, and implementation structure.

## Acknowledgements

- The [Chess Programming Wiki](https://www.chessprogramming.org/) for educational articles and reference material on chess-engine algorithms.
- The [Stockfish project](https://github.com/official-stockfish/Stockfish) and its contributors for their open-source engine and for providing a valuable reference for studying modern chess-engine techniques.
- The Bermuda project logos were AI-generated using Google Gemini.

Bermuda is licensed under the Apache License 2.0; see [LICENSE](LICENSE). Third-party projects and components retain their own licenses.
