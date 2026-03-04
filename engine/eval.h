#pragma once

#include "position.h"
#include "nnue.h"
#include <string>


namespace Eval {

// ============================================================================
// Evaluation
// ============================================================================
// Wraps NNUE inference. The search calls Eval::evaluate() as its sole
// evaluation entry point — all NNUE details are hidden inside eval.cpp.
//
// Eval::evaluate() returns centipawns from the side-to-move's perspective,
// which is what the search expects for negamax correctness.

// Load the NNUE network file from disk.
// Must be called once at startup before any search begins.
// Delegates to nnue_init(). Prints status to stdout.
bool init(const std::string& path);

// Evaluate the position using the loaded NNUE network.
// Returns centipawns from the side-to-move's perspective.
// Returns 0 if no network is loaded — this should not happen in normal use
// but prevents undefined behaviour if init() was not called.
i32 evaluate(const Position& pos, NNUE_Data* nnue_curr, NNUE_Data* nnue_prev1, NNUE_Data* nnue_prev2);

} // namespace Eval
