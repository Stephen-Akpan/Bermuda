#pragma once

#include "position.h"

// ============================================================================
// Evaluation
// ============================================================================
// Returns a score in centipawns from the perspective of the side to move.
// Positive = good for side to move, negative = bad.
//
// Phase 1: Material only — enough to drive search correctly.
// All piece values are in centipawns (pawn = 100).

namespace Eval {

// -------------------------------------------------------------------------
// Piece Values
// -------------------------------------------------------------------------

constexpr i32 PAWN_VALUE   = 100;
constexpr i32 KNIGHT_VALUE = 320;
constexpr i32 BISHOP_VALUE = 330;
constexpr i32 ROOK_VALUE   = 500;
constexpr i32 QUEEN_VALUE  = 900;
constexpr i32 KING_VALUE   = 0;     // Kings are never captured — no material value

constexpr i32 PIECE_VALUES[6] = {
    PAWN_VALUE,
    KNIGHT_VALUE,
    BISHOP_VALUE,
    ROOK_VALUE,
    QUEEN_VALUE,
    KING_VALUE
};

// -------------------------------------------------------------------------
// Piece Square Tables
// -------------------------------------------------------------------------
// Bonuses/penalties based on piece location.
// Indexed [square] from white's perspective (a1=0).
// For black pieces, mirror the rank (square ^ 56).

// Pawns: encourage advancement and central control
constexpr i32 PAWN_PST[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

// Knights: prefer centre, penalise edges and corners
constexpr i32 KNIGHT_PST[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};

// Bishops: prefer long diagonals, penalise corners
constexpr i32 BISHOP_PST[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};

// Rooks: prefer open files and 7th rank
constexpr i32 ROOK_PST[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
};

// Queens: avoid early development, prefer centre in middlegame
constexpr i32 QUEEN_PST[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};

// King middlegame: encourage castling, avoid centre
constexpr i32 KING_MG_PST[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
};

// -------------------------------------------------------------------------
// Main evaluation entry point
// -------------------------------------------------------------------------

// Evaluate the position from the side to move's perspective.
i32 evaluate(const Position& pos);

// Raw material balance (white - black) in centipawns.
// Positive = white is ahead.
i32 material(const Position& pos);

} // namespace Eval
