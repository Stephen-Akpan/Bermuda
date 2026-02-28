#include "eval.h"
#include "bitboard.h"

namespace Eval {

// Returns the PST value for a piece of the given type on the given square,
// from white's perspective. For black pieces, the rank is mirrored.
static i32 pst_value(PieceType pt, Square sq, Colour c) {
    // Mirror square for black (flip rank)
    i32 index = (c == WHITE) ? sq : (sq ^ 56);

    switch (pt) {
        case PAWN:   return PAWN_PST[index];
        case KNIGHT: return KNIGHT_PST[index];
        case BISHOP: return BISHOP_PST[index];
        case ROOK:   return ROOK_PST[index];
        case QUEEN:  return QUEEN_PST[index];
        case KING:   return KING_MG_PST[index];
        default:     return 0;
    }
}

static bool is_insufficient_material(const Position& pos) {
    if (pos.pieces(WHITE, PAWN) || pos.pieces(BLACK, PAWN)) return false;
    if (pos.pieces(WHITE, ROOK) || pos.pieces(BLACK, ROOK))  return false;
    if (pos.pieces(WHITE, QUEEN) || pos.pieces(QUEEN))  return false;

    i32 wN = popcount(pos.pieces(WHITE, KNIGHT));
    i32 wB = popcount(pos.pieces(WHITE, BISHOP));
    i32 bN = popcount(pos.pieces(BLACK, KNIGHT));
    i32 bB = popcount(pos.pieces(BLACK, BISHOP));

    i32 w_minors = wN + wB;
    i32 b_minors = bN + bB;

    if (w_minors == 0 && b_minors == 0) return true;

    if (w_minors == 1 && b_minors == 0) return true;
    if (w_minors == 0 && b_minors == 1) return true;

    if (wN == 2 && wB == 0 && b_minors == 0) return true;
    if (bN == 2 && bB == 0 && w_minors == 0) return true;

    return false;
}

// ============================================================================
// Material
// ============================================================================

i32 material(const Position& pos) {
    i32 score = 0;

    for (i32 pt = PAWN; pt <= QUEEN; pt++) {
        i32 white_count = popcount(pos.pieces(WHITE, (PieceType)pt));
        i32 black_count = popcount(pos.pieces(BLACK, (PieceType)pt));
        score += (white_count - black_count) * PIECE_VALUES[pt];
    }

    return score;
}

// ============================================================================
// Evaluate
// ============================================================================

i32 evaluate(const Position& pos) {
    i32 score = 0;

    if (is_insufficient_material(pos)) {
        return score;
    }

    // Material + PST for all pieces of both colours
    for (i32 c = 0; c < 2; c++) {
        Colour colour = (Colour)c;
        i32 side_score = 0;

        for (i32 pt = PAWN; pt <= KING; pt++) {
            Bitboard bb = pos.pieces(colour, (PieceType)pt);

            while (bb) {
                Square sq = (Square)pop_lsb(bb);
                side_score += PIECE_VALUES[pt];
                side_score += pst_value((PieceType)pt, sq, colour);
            }
        }

        // Add for white, subtract for black
        score += (colour == WHITE) ? side_score : -side_score;
    }

    // Bishop pair bonus — two bishops are worth slightly more together
    if (popcount(pos.pieces(WHITE, BISHOP)) >= 2) score += 30;
    if (popcount(pos.pieces(BLACK, BISHOP)) >= 2) score -= 30;

    // Return score from side to move's perspective
    return (pos.side_to_move() == WHITE) ? score : -score;
}

} // namespace Eval
