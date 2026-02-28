#pragma once

#include "types.h"

// ============================================================================
// Zobrist Hashing
// ============================================================================
// A Zobrist hash is a pseudo-random 64-bit key that uniquely (in practice)
// identifies a chess position. It is computed incrementally — each make_move
// and unmake_move XORs in/out only the keys that changed, rather than
// recomputing from scratch.
//
// The hash encodes:
//   - Which piece is on each square
//   - Side to move
//   - Castling rights (4 bits)
//   - En passant file (if any)
//
// Call Zobrist::init() once at startup (after BB::init()).

namespace Zobrist {

// -------------------------------------------------------------------------
// Key Tables
// -------------------------------------------------------------------------

// piece_keys[piece][square] — one key per (piece, square) combination
// piece index: use the Piece enum directly (0-13, NO_PIECE not used)
extern ui64 piece_keys[14][64];

// side_key — XORed in when it is black's turn to move
extern ui64 side_key;

// castling_keys[0..15] — one key per castling rights bitmask value
// Index directly with the 4-bit castling rights value
extern ui64 castling_keys[16];

// ep_keys[0..7] — one key per en passant file (only file matters, not rank)
extern ui64 ep_keys[8];

// -------------------------------------------------------------------------
// Initialisation
// -------------------------------------------------------------------------

// Seed and fill all Zobrist key tables with pseudo-random values.
// Must be called once at startup before any Position is created.
void init();

// -------------------------------------------------------------------------
// Full hash computation (used once when setting up a position from FEN)
// -------------------------------------------------------------------------

// Compute the full Zobrist hash of a position from scratch.
// Prefer incremental updates via make_move/unmake_move in normal operation.
ui64 compute(
    const ui64  piece_bb[2][6],   // piece_bb[colour][piece_type]
    ui64        side_to_move,     // WHITE=0, BLACK=1
    ui8         castling_rights,
    Square      ep_sq
);

} // namespace Zobrist
