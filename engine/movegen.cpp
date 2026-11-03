#include "movegen.h"
#include "bitboard.h"

namespace MoveGen {

// ============================================================================
// Internal Helpers
// ============================================================================

// Extract all set bits from a bitboard and push a move for each,
// using a fixed 'from' square (used for sliding/knight/king moves).
static void push_all(MoveList& list, Square from, Bitboard targets) {
    while (targets) {
        Square to = (Square)pop_lsb(targets);
        list.push(from, to);
    }
}

// Push all moves from a shifted pawn bitboard where 'to - delta = from'.
static void push_pawn_moves(MoveList& list, Bitboard targets, i32 delta) {
    while (targets) {
        Square to   = (Square)pop_lsb(targets);
        Square from = (Square)((i32)to - delta);
        list.push(from, to);
    }
}

// Push all promotions from a shifted pawn bitboard where 'to - delta = from'.
static void push_pawn_promos(MoveList& list, Bitboard targets, i32 delta) {
    while (targets) {
        Square to   = (Square)pop_lsb(targets);
        Square from = (Square)((i32)to - delta);
        list.push_promo(from, to);
    }
}

// ============================================================================
// Pawn Move Generation
// ============================================================================

static void generate_pawn_moves(const Position& pos, MoveList& list, bool captures_only) {
    const Colour us   = pos.side_to_move();
    const Colour them = flip(us);

    const Bitboard pawns    = pos.pieces(us, PAWN);
    const Bitboard enemies  = pos.pieces(them);
    const Bitboard empty    = ~pos.all_pieces();

    // Promotion and non-promotion ranks
    const Bitboard promo_rank    = (us == WHITE) ? BB::RANK_8 : BB::RANK_1;
    const Bitboard non_promo     = ~promo_rank;
    const Bitboard start_rank    = (us == WHITE) ? BB::RANK_2 : BB::RANK_7;

    // Push directions
    const i32 push_delta  = (us == WHITE) ? 8  : -8;
    const i32 push2_delta = (us == WHITE) ? 16 : -16;

    // Attack deltas: left and right relative to direction of travel
    const i32 atk_left_delta  = (us == WHITE) ? 7  : -9;
    const i32 atk_right_delta = (us == WHITE) ? 9  : -7;

    if (!captures_only) {
        // -------------------------------------------------------------------------
        // Single pawn pushes (non-promotion)
        // -------------------------------------------------------------------------
        Bitboard single_push = BB::pawn_push(pawns, us) & empty & non_promo;
        push_pawn_moves(list, single_push, push_delta);

        // -------------------------------------------------------------------------
        // Double pawn pushes
        // -------------------------------------------------------------------------
        // Only pawns on the starting rank, and the intermediate square must be empty
        Bitboard double_push = BB::pawn_push(pawns & start_rank, us) & empty;
        double_push          = BB::pawn_push(double_push, us) & empty;
        push_pawn_moves(list, double_push, push2_delta);
    }

    // -------------------------------------------------------------------------
    // Pawn captures (non-promotion)
    // -------------------------------------------------------------------------
    // Left captures
    Bitboard atk_left  = BB::pawn_push(pawns, us);
    Bitboard atk_right = BB::pawn_push(pawns, us);


    // Split into left/right by re-deriving per direction
    if (us == WHITE) {
        atk_left  = ((pawns << 7) & ~BB::FILE_H) & enemies & non_promo;
        atk_right = ((pawns << 9) & ~BB::FILE_A) & enemies & non_promo;
    } else {
        atk_left  = ((pawns >> 9) & ~BB::FILE_H) & enemies & non_promo;
        atk_right = ((pawns >> 7) & ~BB::FILE_A) & enemies & non_promo;
    }

    push_pawn_moves(list, atk_left,  atk_left_delta);
    push_pawn_moves(list, atk_right, atk_right_delta);

    // -------------------------------------------------------------------------
    // Promotions (push and capture)
    // -------------------------------------------------------------------------
    Bitboard promo_pawns = pawns & ((us == WHITE) ? BB::RANK_7 : BB::RANK_2);

    if (promo_pawns) {
        // Push promotions
        {
            Bitboard push_promo = BB::pawn_push(promo_pawns, us) & empty;
            if (captures_only) {
                while (push_promo) {
                    Square to = static_cast<Square>(pop_lsb(push_promo));
                    Square from = static_cast<Square>((i32)to - push_delta);
                    list.push(Move::make_promo(from, to, PROMO_QUEEN));
                }
            } else {
                push_pawn_promos(list, push_promo, push_delta);
            }
        }

        // Capture promotions
        Bitboard cap_promo_left, cap_promo_right;

        if (us == WHITE) {
            cap_promo_left  = ((promo_pawns << 7) & ~BB::FILE_H) & enemies;
            cap_promo_right = ((promo_pawns << 9) & ~BB::FILE_A) & enemies;
        } else {
            cap_promo_left  = ((promo_pawns >> 9) & ~BB::FILE_H) & enemies;
            cap_promo_right = ((promo_pawns >> 7) & ~BB::FILE_A) & enemies;
        }

        push_pawn_promos(list, cap_promo_left,  atk_left_delta);
        push_pawn_promos(list, cap_promo_right, atk_right_delta);
    }

    // -------------------------------------------------------------------------
    // En passant
    // -------------------------------------------------------------------------
    Square ep = pos.ep_square();

    if (ep != NO_SQUARE) {
        Bitboard ep_bb      = BB::square_bb(ep);
        Bitboard ep_attacks = BB::pawn_attacks(ep_bb, them) & pawns;

        while (ep_attacks) {
            Square from = (Square)pop_lsb(ep_attacks);
            list.push_ep(from, ep);
        }
    }
}

// ============================================================================
// Knight Move Generation
// ============================================================================

static void generate_knight_moves(const Position& pos, MoveList& list, bool captures_only) {
    const Colour us      = pos.side_to_move();
    const Bitboard own   = pos.pieces(us);
    const Bitboard enemy = pos.pieces(flip(us));

    Bitboard knights = pos.pieces(us, KNIGHT);

    while (knights) {
        Square from    = (Square)pop_lsb(knights);
        Bitboard atk   = BB::knight_attack(from) & ~own;

        if (captures_only)
            atk &= enemy;

        push_all(list, from, atk);
    }
}

// ============================================================================
// Bishop Move Generation
// ============================================================================

static void generate_bishop_moves(const Position& pos, MoveList& list, bool captures_only) {
    const Colour us      = pos.side_to_move();
    const Bitboard own   = pos.pieces(us);
    const Bitboard enemy = pos.pieces(flip(us));
    const Bitboard occ   = pos.all_pieces();

    Bitboard bishops = pos.pieces(us, BISHOP);

    while (bishops) {
        Square from  = (Square)pop_lsb(bishops);
        Bitboard atk = BB::bishop_attack(from, occ) & ~own;

        if (captures_only)
            atk &= enemy;

        push_all(list, from, atk);
    }
}

// ============================================================================
// Rook Move Generation
// ============================================================================

static void generate_rook_moves(const Position& pos, MoveList& list, bool captures_only) {
    const Colour us      = pos.side_to_move();
    const Bitboard own   = pos.pieces(us);
    const Bitboard enemy = pos.pieces(flip(us));
    const Bitboard occ   = pos.all_pieces();

    Bitboard rooks = pos.pieces(us, ROOK);

    while (rooks) {
        Square from  = (Square)pop_lsb(rooks);
        Bitboard atk = BB::rook_attack(from, occ) & ~own;

        if (captures_only)
            atk &= enemy;

        push_all(list, from, atk);
    }
}

// ============================================================================
// Queen Move Generation
// ============================================================================

static void generate_queen_moves(const Position& pos, MoveList& list, bool captures_only) {
    const Colour us      = pos.side_to_move();
    const Bitboard own   = pos.pieces(us);
    const Bitboard enemy = pos.pieces(flip(us));
    const Bitboard occ   = pos.all_pieces();

    Bitboard queens = pos.pieces(us, QUEEN);

    while (queens) {
        Square from  = (Square)pop_lsb(queens);
        Bitboard atk = BB::queen_attack(from, occ) & ~own;

        if (captures_only)
            atk &= enemy;

        push_all(list, from, atk);
    }
}

// ============================================================================
// King Move Generation
// ============================================================================

static void generate_king_moves(const Position& pos, MoveList& list, bool captures_only) {
    const Colour us      = pos.side_to_move();
    const Bitboard own   = pos.pieces(us);
    const Bitboard enemy = pos.pieces(flip(us));

    Square from    = pos.king_square(us);
    Bitboard atk   = BB::king_attack(from) & ~own;

    if (captures_only)
        atk &= enemy;

    push_all(list, from, atk);
}

// ============================================================================
// Castling Move Generation
// ============================================================================

static void generate_castling_moves(const Position& pos, MoveList& list) {
    const Colour us  = pos.side_to_move();
    const Bitboard occ = pos.all_pieces();

    if (us == WHITE) {
        // White kingside: e1-f1-g1 must be empty, e1-f1-g1 not attacked
        if (pos.can_castle(WHITE_OO)) {
            constexpr Bitboard path    = (1ULL << f1) | (1ULL << g1);
            constexpr Bitboard king_path = (1ULL << e1) | (1ULL << f1) | (1ULL << g1);

            if (!(occ & path)
                && !pos.is_attacked_by(e1, BLACK)
                && !pos.is_attacked_by(f1, BLACK)
                && !pos.is_attacked_by(g1, BLACK))
            {
                list.push_castling(e1, g1);
            }
        }

        // White queenside: b1-c1-d1 must be empty, e1-d1-c1 not attacked
        if (pos.can_castle(WHITE_OOO)) {
            constexpr Bitboard path      = (1ULL << b1) | (1ULL << c1) | (1ULL << d1);
            constexpr Bitboard king_path = (1ULL << e1) | (1ULL << d1) | (1ULL << c1);

            if (!(occ & path)
                && !pos.is_attacked_by(e1, BLACK)
                && !pos.is_attacked_by(d1, BLACK)
                && !pos.is_attacked_by(c1, BLACK))
            {
                list.push_castling(e1, c1);
            }
        }
    } else {
        // Black kingside: e8-f8-g8 must be empty, e8-f8-g8 not attacked
        if (pos.can_castle(BLACK_OO)) {
            constexpr Bitboard path    = (1ULL << f8) | (1ULL << g8);

            if (!(occ & path)
                && !pos.is_attacked_by(e8, WHITE)
                && !pos.is_attacked_by(f8, WHITE)
                && !pos.is_attacked_by(g8, WHITE))
            {
                list.push_castling(e8, g8);
            }
        }

        // Black queenside: b8-c8-d8 must be empty, e8-d8-c8 not attacked
        if (pos.can_castle(BLACK_OOO)) {
            constexpr Bitboard path = (1ULL << b8) | (1ULL << c8) | (1ULL << d8);

            if (!(occ & path)
                && !pos.is_attacked_by(e8, WHITE)
                && !pos.is_attacked_by(d8, WHITE)
                && !pos.is_attacked_by(c8, WHITE))
            {
                list.push_castling(e8, c8);
            }
        }
    }
}

// ============================================================================
// Public Interface
// ============================================================================

void generate_moves(const Position& pos, MoveList& list) {
    generate_pawn_moves(pos, list, false);
    generate_knight_moves(pos, list, false);
    generate_bishop_moves(pos, list, false);
    generate_rook_moves(pos, list, false);
    generate_queen_moves(pos, list, false);
    generate_king_moves(pos, list, false);
    generate_castling_moves(pos, list);
}

void generate_captures(const Position& pos, MoveList& list) {
    generate_pawn_moves(pos, list, true);
    generate_knight_moves(pos, list, true);
    generate_bishop_moves(pos, list, true);
    generate_rook_moves(pos, list, true);
    generate_queen_moves(pos, list, true);
    generate_king_moves(pos, list, true);
    // No castling in captures — castling is never a capture
}

} // namespace MoveGen
