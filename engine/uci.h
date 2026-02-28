#pragma once

#include "search.h"
#include "position.h"
#include "tt.h"
#include <string>

// ============================================================================
// UCI Protocol Handler
// ============================================================================
// Implements the Universal Chess Interface protocol.
// https://www.shredderchess.com/chess-features/uci-universal-chess-interface.html
//
// The main loop reads lines from stdin and dispatches to handlers.
// The engine communicates back via stdout.
//
// Thread model (current — single threaded):
//   - All search runs on the main thread
//   - "stop" is handled by setting the stop flag before the next node check
//
// UCI commands handled:
//   uci
//   isready
//   setoption name <id> [value <x>]
//   ucinewgame
//   position [startpos | fen <fenstring>] [moves <move1> ... <moveN>]
//   go [depth <x>] [movetime <x>] [wtime <x>] [btime <x>]
//      [winc <x>] [binc <x>] [movestogo <x>] [infinite]
//   stop
//   quit
//   -- Non-standard debug commands --
//   d                  (print current board)
//   eval               (print static evaluation)
//   perft <depth>      (run perft from current position)

namespace UCI {

// ============================================================================
// Engine Options
// ============================================================================

struct Options {
    int  hash_mb    = 64;     // TT size in megabytes
    int  move_overhead = 10;  // Safety buffer subtracted from time (ms)
};

// Global options instance
extern Options options;

// ============================================================================
// Entry Point
// ============================================================================

// Start the UCI main loop. Blocks until "quit" is received.
void loop();

// ============================================================================
// Individual Command Handlers (exposed for testing)
// ============================================================================

// Handle "position" command — sets up the board
void cmd_position(SearchInfo& info, const std::string& line);

// Handle "go" command — starts search
void cmd_go(SearchInfo& info, const std::string& line);

// Handle "setoption" command — updates engine options
void cmd_setoption(const std::string& line);

} // namespace UCI
