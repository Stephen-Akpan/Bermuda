#pragma once

#include "position.h"
#include <cstdint>
#include <string>

// ============================================================================
// Perft
// ============================================================================
// Counts leaf nodes at a given depth to verify move generation correctness.
// A move is only counted if it is fully legal (does not leave king in check).

namespace Perft {

// Result of a perft run broken down by move category
struct Result {
    ui64 nodes      = 0;
    ui64 captures   = 0;
    ui64 ep         = 0;
    ui64 castles    = 0;
    ui64 promotions = 0;
    ui64 checks     = 0;
};

// Run perft to a given depth and return the result
Result perft(Position& pos, i32 depth);

// Run perft and print a divide breakdown (each root move and its node count)
// Useful for pinpointing which move is wrong when a count doesn't match
void divide(Position& pos, i32 depth);

} // namespace Perft
