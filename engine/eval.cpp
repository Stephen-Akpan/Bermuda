#include "eval.h"
#include "nnue.h"
#include "bitboard.h"
#include <iostream>

// ============================================================================
// Piece code mapping: Position → NNUE
// ============================================================================
//
// Your Position uses:
//   PieceType enum: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5
//   Colour enum:    WHITE=0, BLACK=1
//   Piece = make_piece(colour, type) — combined value
//
// nnue.h NN_Pieces enum:
//   BLANK=0,
//   W_KING=1,  W_QUEEN=2,  W_ROOK=3,  W_BISHOP=4,  W_KNIGHT=5,  W_PAWN=6,
//   B_KING=7,  B_QUEEN=8,  B_ROOK=9,  B_BISHOP=10, B_KNIGHT=11, B_PAWN=12
//
// Note the ordering: the NNUE piece codes rank pieces high-to-low
// (queen first, pawn last) whereas PieceType goes low-to-high
// (pawn first, king last). The table below handles this remapping.
//
// Indexed as PIECE_TO_NNUE[colour][piece_type] — both dimensions 0-indexed.

static constexpr i32 PIECE_TO_NNUE[2][6] = {
    //  PAWN        KNIGHT      BISHOP      ROOK        QUEEN       KING
    {   6,     5,   4,   3,     2,    1  },  // WHITE
    {   12,     11,   10,   9,     8,    7  },  // BLACK
};

namespace Eval {

// ============================================================================
// State
// ============================================================================

static bool network_loaded = false;

// ============================================================================
// init
// ============================================================================

bool init(const std::string& path) {
    // nnue_init prints its own "NNUE loaded" / "NNUE not found" message
    const i32 result = nnue_init(path.c_str());
    network_loaded = (result == 0);  // nnue_init returns fflush result; 0 = success

    if (!network_loaded) {
        std::cerr << "info string Eval::init failed for: " << path << "\n";
    }

    return network_loaded;
}

// ============================================================================
// Build NNUE input arrays from Position
// ============================================================================
//
// nnue_evaluate() expects:
//   pieces[]  — NNUE piece codes, kings at [0] and [1], rest from [2], 0-terminated
//   squares[] — matching square indices (a1=0 … h8=63), same encoding as ours
//
// Kings must be first — the inference code reads pieces[0]/pieces[1] specifically
// to locate the king squares for HalfKP feature index computation.
// Placing any other piece before the kings produces silently wrong evaluations.
//
// Maximum legal pieces on a board: 32. Arrays are sized 33 to hold terminator.

static void build_nnue_input(const Position& pos,
                              i32 pieces[33],
                              i32 squares[33]) {
    i32 idx = 2;  // index 0 and 1 are reserved for kings

    // Kings first — mandatory ordering for nnue_evaluate
    const Square wk = pos.king_square(WHITE);
    const Square bk = pos.king_square(BLACK);

    pieces[0]  = W_KING;
    squares[0] = static_cast<i32>(wk);

    pieces[1]  = B_KING;
    squares[1] = static_cast<i32>(bk);

    // All non-king pieces in bitboard iteration order (order doesn't matter
    // for correctness, only kings must be first)
    Bitboard occupied = pos.all_pieces()
                      & ~pos.pieces(WHITE, KING)
                      & ~pos.pieces(BLACK, KING);

    while (occupied) {
        const Square  sq    = static_cast<Square>(pop_lsb(occupied));
        const Piece   piece = pos.piece_on(sq);
        const Colour  c     = colour_of(piece);
        const PieceType pt  = type_of(piece);

        pieces[idx]  = PIECE_TO_NNUE[c][pt];
        squares[idx] = static_cast<i32>(sq);
        idx++;
    }

    // Terminator — nnue_evaluate stops reading when it sees a 0 piece code
    pieces[idx]  = 0;
    squares[idx] = 0;
}

// ============================================================================
// evaluate
// ============================================================================

i32 evaluate(const Position& pos, NNUE_Data* nnue_curr, NNUE_Data* nnue_prev1, NNUE_Data* nnue_prev2) {
    if (!network_loaded)
        return 0;

    i32 pieces[33]{};
    i32 squares[33]{};

    build_nnue_input(pos, pieces, squares);

    // nnue_evaluate player convention: 0 = white to move, 1 = black to move
    const i32 player = (pos.side_to_move() == WHITE) ? 0 : 1;

    NNUE_Position nnue_pos{};
    nnue_pos.player = player;
    nnue_pos.pieces = pieces;
    nnue_pos.squares = squares;
    nnue_pos.nnue[0] = nnue_curr;
    nnue_pos.nnue[1] = nnue_prev1;
    nnue_pos.nnue[2] = nnue_prev2;

    return nnue_evaluate_pos(&nnue_pos);
}

} // namespace Eval
