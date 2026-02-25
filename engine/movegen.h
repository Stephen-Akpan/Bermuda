#pragma once

#include "position.h"
#include "move.h"

// ============================================================================
// Move Generation
// ============================================================================
// Generates all pseudo-legal moves for the side to move.
// Pseudo-legal means moves are geometrically valid but may leave the king
// in check. Legality must be verified by the caller via make_move + in_check.

namespace MoveGen {

// Generate all pseudo-legal moves for the current side to move.
// Moves are appended to the provided MoveList.
void generate_moves(const Position& pos, MoveList& list);

// Generate only captures and queen promotions (for quiescence search).
void generate_captures(const Position& pos, MoveList& list);

} // namespace MoveGen
