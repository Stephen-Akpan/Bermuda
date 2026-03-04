# Bermuda Chess Engine

A UCI-compatible chess engine written in C++, built from scratch without external dependencies.

## Features

- **Search**: Minimax with Alpha-Beta Pruning
- **Move Ordering**: Improves pruning efficiency by searching likely best moves first
- **Board Representation**: Bitboards for fast move generation and evaluation
- **Transposition Table**: Caches previously evaluated positions to avoid redundant computation
- **Interface**: UCI (Universal Chess Interface) protocol

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

## Platform Support

| Platform | Status |
|----------|--------|
| Windows  | ✅ Tested |
| Linux    | ❓ Untested |
| macOS    | ❓ Untested |

## Roadmap

- [ ] ELO benchmarking against other engines
- [ ] Linux & macOS support
- [ ] Further search optimizations (null move pruning, late move reductions)
- [ ] Neural network evaluation (NNUE)

## License

All rights reserved. This project is not currently open for redistribution or commercial use.

---

> *Bermuda is a personal project with commercial ambitions. Contributions are not currently accepted.*

