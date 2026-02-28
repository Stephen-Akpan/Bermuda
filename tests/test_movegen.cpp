#include "catch_amalgamated.hpp"
#include "movegen.h"
#include "position.h"
#include "move.h"

inline void init_bb() {
  static bool initialized = false;
  if (initialized == false) {
    BB::init();
    Zobrist::init();
    initialized = true;
  }
}

// ============================================================================
// Helpers
// ============================================================================

static Position make_pos(const std::string& fen) {
    init_bb();
    return Position::from_fen(fen);
}

// Count how many moves in the list match a given from/to pair
static i32 count_move(const MoveList& list, Square from, Square to) {
    i32 n = 0;
    for (i32 i = 0; i < list.size(); i++)
        if (list[i].from() == from && list[i].to() == to)
            n++;
    return n;
}

// Check whether a specific move exists in the list
static bool has_move(const MoveList& list, Square from, Square to) {
    return count_move(list, from, to) >= 1;
}

// Check whether a move with a specific flag exists
static bool has_move(const MoveList& list, Square from, Square to, MoveFlag flag) {
    for (i32 i = 0; i < list.size(); i++)
        if (list[i].from() == from && list[i].to() == to && list[i].flag() == flag)
            return true;
    return false;
}

// Check whether a promotion to a specific piece exists
static bool has_promo(const MoveList& list, Square from, Square to, PromoPiece pp) {
    for (i32 i = 0; i < list.size(); i++)
        if (list[i].from() == from && list[i].to() == to
            && list[i].flag() == PROMOTION && list[i].promo() == pp)
            return true;
    return false;
}

// Count total moves from a specific square
static i32 count_from(const MoveList& list, Square from) {
    i32 n = 0;
    for (i32 i = 0; i < list.size(); i++)
        if (list[i].from() == from)
            n++;
    return n;
}

// ============================================================================
// Pawn Moves — White
// ============================================================================

TEST_CASE("movegen pawns: white single push", "[movegen][pawn]") {
    Position pos = make_pos("8/8/8/8/8/8/4P3/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e2, e3));
}

TEST_CASE("movegen pawns: white double push from rank 2", "[movegen][pawn]") {
    Position pos = make_pos("8/8/8/8/8/8/4P3/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e2, e4));
}

TEST_CASE("movegen pawns: white no double push if blocked on rank 3", "[movegen][pawn]") {
    // Black pawn blocking on e3
    Position pos = make_pos("8/8/8/8/8/4p3/4P3/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE_FALSE(has_move(list, e2, e3));
    REQUIRE_FALSE(has_move(list, e2, e4));
}

TEST_CASE("movegen pawns: white no double push if blocked on rank 4", "[movegen][pawn]") {
    // Black pawn blocking on e4 but e3 is empty
    Position pos = make_pos("8/8/8/8/4p3/8/4P3/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e2, e3));
    REQUIRE_FALSE(has_move(list, e2, e4));
}

TEST_CASE("movegen pawns: white capture left and right", "[movegen][pawn]") {
    Position pos = make_pos("8/8/8/8/8/3p1p2/4P3/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e2, d3));
    REQUIRE(has_move(list, e2, f3));
}

TEST_CASE("movegen pawns: white cannot capture own piece", "[movegen][pawn]") {
    Position pos = make_pos("8/8/8/8/8/3N1N2/4P3/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE_FALSE(has_move(list, e2, d3));
    REQUIRE_FALSE(has_move(list, e2, f3));
}

TEST_CASE("movegen pawns: white pawn cannot push to occupied square", "[movegen][pawn]") {
    Position pos = make_pos("8/8/8/8/4p3/4P3/6K1/7k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE_FALSE(has_move(list, e2, e3));
}

TEST_CASE("movegen pawns: white en passant", "[movegen][pawn]") {
    // Black just pushed d7-d5, white pawn on e5
    Position pos = make_pos("8/8/8/3pP3/8/8/8/4K2k w - d6 0 2");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e5, d6, EN_PASSANT));
}

TEST_CASE("movegen pawns: white en passant both sides", "[movegen][pawn]") {
    // White pawn on e5, black pawns on both d5 and f5
    Position pos = make_pos("8/8/8/3pPp2/8/8/8/4K2k w - f6 0 2");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e5, f6, EN_PASSANT));
    // d6 is not the ep square so should not be generated as ep
    REQUIRE_FALSE(has_move(list, e5, d6, EN_PASSANT));
}

TEST_CASE("movegen pawns: white promotion push generates all four", "[movegen][pawn]") {
    Position pos = make_pos("8/4P3/8/8/8/8/8/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_promo(list, e7, e8, PROMO_QUEEN));
    REQUIRE(has_promo(list, e7, e8, PROMO_ROOK));
    REQUIRE(has_promo(list, e7, e8, PROMO_BISHOP));
    REQUIRE(has_promo(list, e7, e8, PROMO_KNIGHT));
    // Exactly 4 moves from e7 to e8
    REQUIRE(count_move(list, e7, e8) == 4);
}

TEST_CASE("movegen pawns: white promotion capture generates all four", "[movegen][pawn]") {
    Position pos = make_pos("3r4/4P3/8/8/8/8/8/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_promo(list, e7, d8, PROMO_QUEEN));
    REQUIRE(has_promo(list, e7, d8, PROMO_ROOK));
    REQUIRE(has_promo(list, e7, d8, PROMO_BISHOP));
    REQUIRE(has_promo(list, e7, d8, PROMO_KNIGHT));
}

TEST_CASE("movegen pawns: white pawn on rank 7 generates only promotions", "[movegen][pawn]") {
    Position pos = make_pos("8/4P3/8/8/8/8/8/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    // Should not generate e7e6 or any non-promotion
    for (i32 i = 0; i < list.size(); i++) {
        if (list[i].from() == e7)
            REQUIRE(list[i].is_promotion());
    }
}

// ============================================================================
// Pawn Moves — Black
// ============================================================================

TEST_CASE("movegen pawns: black single push", "[movegen][pawn]") {
    Position pos = make_pos("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e7, e6));
}

TEST_CASE("movegen pawns: black double push from rank 7", "[movegen][pawn]") {
    Position pos = make_pos("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e7, e5));
}

TEST_CASE("movegen pawns: black capture left and right", "[movegen][pawn]") {
    Position pos = make_pos("4k3/4p3/3P1P2/8/8/8/8/4K3 b - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e7, d6));
    REQUIRE(has_move(list, e7, f6));
}

TEST_CASE("movegen pawns: black en passant", "[movegen][pawn]") {
    // White just pushed e2-e4, black pawn on d4
    Position pos = make_pos("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 2");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, d4, e3, EN_PASSANT));
}

TEST_CASE("movegen pawns: black promotion push generates all four", "[movegen][pawn]") {
    Position pos = make_pos("4k3/8/8/8/6K1/8/4p3/8 b - - 0 1 ");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_promo(list, e2, e1, PROMO_QUEEN));
    REQUIRE(has_promo(list, e2, e1, PROMO_ROOK));
    REQUIRE(has_promo(list, e2, e1, PROMO_BISHOP));
    REQUIRE(has_promo(list, e2, e1, PROMO_KNIGHT));
    REQUIRE(count_move(list, e2, e1) == 4);
}

TEST_CASE("movegen pawns: black promotion capture generates all four", "[movegen][pawn]") {
    Position pos = make_pos("4k3/8/8/8/8/8/4p3/3RK3 b - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_promo(list, e2, d1, PROMO_QUEEN));
    REQUIRE(has_promo(list, e2, d1, PROMO_ROOK));
    REQUIRE(has_promo(list, e2, d1, PROMO_BISHOP));
    REQUIRE(has_promo(list, e2, d1, PROMO_KNIGHT));
}

// ============================================================================
// Knight Moves
// ============================================================================

TEST_CASE("movegen knights: all 8 moves from centre", "[movegen][knight]") {
    Position pos = make_pos("8/8/8/8/4N3/8/8/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e4, d6));
    REQUIRE(has_move(list, e4, f6));
    REQUIRE(has_move(list, e4, c5));
    REQUIRE(has_move(list, e4, g5));
    REQUIRE(has_move(list, e4, c3));
    REQUIRE(has_move(list, e4, g3));
    REQUIRE(has_move(list, e4, d2));
    REQUIRE(has_move(list, e4, f2));
    REQUIRE(count_from(list, e4) == 8);
}

TEST_CASE("movegen knights: corner knight has 2 moves", "[movegen][knight]") {
    Position pos = make_pos("N7/8/8/8/8/8/8/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(count_from(list, a8) == 2);
    REQUIRE(has_move(list, a8, b6));
    REQUIRE(has_move(list, a8, c7));
}

TEST_CASE("movegen knights: cannot jump to own piece square", "[movegen][knight]") {
    // Put white pieces on all knight destinations from e4
    Position pos = make_pos("8/8/3N1N2/2N3N1/4N3/2N3N1/3N1N2/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(count_from(list, e4) == 0);
}

TEST_CASE("movegen knights: can capture enemy piece", "[movegen][knight]") {
    Position pos = make_pos("8/8/8/8/4N3/8/3p4/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e4, d2));
}

// ============================================================================
// Bishop Moves
// ============================================================================

TEST_CASE("movegen bishops: full diagonals from centre on empty board", "[movegen][bishop]") {
    Position pos = make_pos("8/6k1/8/8/4B3/8/5K2/8 w - - 0 1 ");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    // NE diagonal
    REQUIRE(has_move(list, e4, f5));
    REQUIRE(has_move(list, e4, g6));
    REQUIRE(has_move(list, e4, h7));
    // NW diagonal
    REQUIRE(has_move(list, e4, d5));
    REQUIRE(has_move(list, e4, c6));
    REQUIRE(has_move(list, e4, b7));
    REQUIRE(has_move(list, e4, a8));
    // SE diagonal
    REQUIRE(has_move(list, e4, f3));
    REQUIRE(has_move(list, e4, g2));
    REQUIRE(has_move(list, e4, h1));
    // SW diagonal
    REQUIRE(has_move(list, e4, d3));
    REQUIRE(has_move(list, e4, c2));
    REQUIRE(has_move(list, e4, b1));
}

TEST_CASE("movegen bishops: blocked by own piece", "[movegen][bishop]") {
    // White pawn on f5 blocks NE ray
    Position pos = make_pos("8/8/5P2/4B3/8/8/6K1/7k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE_FALSE(has_move(list, e4, f5));  // own piece on f5
    REQUIRE_FALSE(has_move(list, e4, g6));  // behind own piece
    REQUIRE_FALSE(has_move(list, e4, h7));
}

TEST_CASE("movegen bishops: can capture enemy but not go past", "[movegen][bishop]") {
    // Black pawn on f5
    Position pos = make_pos("8/5k2/8/5p2/4B3/8/8/6K1 w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e4, f5));        // can capture
    REQUIRE_FALSE(has_move(list, e4, g6));  // cannot go past
}

// ============================================================================
// Rook Moves
// ============================================================================

TEST_CASE("movegen rooks: full rays on empty board", "[movegen][rook]") {
    Position pos = make_pos("8/8/8/4R3/8/6K1/8/6k1 w - - 2 2");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    // North
    REQUIRE(has_move(list, e5, e6));
    REQUIRE(has_move(list, e5, e8));
    // South
    REQUIRE(has_move(list, e5, e3));
    REQUIRE(has_move(list, e5, e1));
    // East
    REQUIRE(has_move(list, e5, f5));
    REQUIRE(has_move(list, e5, h5));
    // West
    REQUIRE(has_move(list, e5, d5));
    REQUIRE(has_move(list, e5, a5));
}

TEST_CASE("movegen rooks: blocked by own piece", "[movegen][rook]") {
    // White pawn on e6 blocks north ray
    Position pos = make_pos("7k/8/4P3/8/4R3/8/8/6K1 w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e4, e5));
    REQUIRE_FALSE(has_move(list, e4, e6));
    REQUIRE_FALSE(has_move(list, e4, e7));
    REQUIRE_FALSE(has_move(list, e4, e8));
}

TEST_CASE("movegen rooks: can capture enemy but not go past", "[movegen][rook]") {
    Position pos = make_pos("6k1/8/4p3/8/4R3/8/8/5K2 w - - 0 1 ");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e4, e5));
    REQUIRE(has_move(list, e4, e6));        // capture
    REQUIRE_FALSE(has_move(list, e4, e7));  // past blocker
    REQUIRE_FALSE(has_move(list, e4, e8));
}

// ============================================================================
// Queen Moves
// ============================================================================

TEST_CASE("movegen queens: combines rook and bishop rays", "[movegen][queen]") {
    Position pos = make_pos("8/8/8/8/4Q3/7k/5K2/8 w - - 8 5");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    // Rook-like
    REQUIRE(has_move(list, e4, e8));
    REQUIRE(has_move(list, e4, e1));
    REQUIRE(has_move(list, e4, a4));
    REQUIRE(has_move(list, e4, h4));
    // Bishop-like
    REQUIRE(has_move(list, e4, a8));
    REQUIRE(has_move(list, e4, h7));
    REQUIRE(has_move(list, e4, h1));
    REQUIRE(has_move(list, e4, b1));
}

// ============================================================================
// King Moves
// ============================================================================

TEST_CASE("movegen king: all 8 moves from centre", "[movegen][king]") {
    Position pos = make_pos("8/8/8/8/4K3/8/8/7k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e4, d5));
    REQUIRE(has_move(list, e4, e5));
    REQUIRE(has_move(list, e4, f5));
    REQUIRE(has_move(list, e4, d4));
    REQUIRE(has_move(list, e4, f4));
    REQUIRE(has_move(list, e4, d3));
    REQUIRE(has_move(list, e4, e3));
    REQUIRE(has_move(list, e4, f3));
    REQUIRE(count_from(list, e4) == 8);
}

TEST_CASE("movegen king: cannot move to own piece square", "[movegen][king]") {
    // Surround king with own pieces except one square
    Position pos = make_pos("8/8/8/3PPP2/3PKP2/3PPP2/8/7k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(count_from(list, e4) == 0);
}

TEST_CASE("movegen king: corner king has 3 moves", "[movegen][king]") {
    Position pos = make_pos("K7/8/8/8/8/8/8/7k w - - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(count_from(list, a8) == 3);
    REQUIRE(has_move(list, a8, b8));
    REQUIRE(has_move(list, a8, a7));
    REQUIRE(has_move(list, a8, b7));
}

// ============================================================================
// Castling
// ============================================================================

TEST_CASE("movegen castling: white kingside", "[movegen][castling]") {
    Position pos = make_pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e1, g1, CASTLING));
}

TEST_CASE("movegen castling: white queenside", "[movegen][castling]") {
    Position pos = make_pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e1, c1, CASTLING));
}

TEST_CASE("movegen castling: black kingside", "[movegen][castling]") {
    Position pos = make_pos("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e8, g8, CASTLING));
}

TEST_CASE("movegen castling: black queenside", "[movegen][castling]") {
    Position pos = make_pos("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e8, c8, CASTLING));
}

TEST_CASE("movegen castling: no kingside if right missing", "[movegen][castling]") {
    Position pos = make_pos("r3k2r/8/8/8/8/8/8/R3K2R w Qkq - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE_FALSE(has_move(list, e1, g1, CASTLING));
    REQUIRE(has_move(list, e1, c1, CASTLING));
}

TEST_CASE("movegen castling: no queenside if right missing", "[movegen][castling]") {
    Position pos = make_pos("r3k2r/8/8/8/8/8/8/R3K2R w Kkq - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e1, g1, CASTLING));
    REQUIRE_FALSE(has_move(list, e1, c1, CASTLING));
}

TEST_CASE("movegen castling: blocked by piece between king and rook kingside", "[movegen][castling]") {
    Position pos = make_pos("r3k2r/8/8/8/8/8/8/R3KN1R w KQkq - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE_FALSE(has_move(list, e1, g1, CASTLING));
    REQUIRE(has_move(list, e1, c1, CASTLING));
}

TEST_CASE("movegen castling: blocked by piece between king and rook queenside", "[movegen][castling]") {
    Position pos = make_pos("r3k2r/8/8/8/8/8/8/RN2K2R w KQkq - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e1, g1, CASTLING));
    REQUIRE_FALSE(has_move(list, e1, c1, CASTLING));
}

TEST_CASE("movegen castling: not allowed when king is in check", "[movegen][castling]") {
    // Black rook on e8 gives check to white king on e1
    Position pos = make_pos("4r3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE_FALSE(has_move(list, e1, g1, CASTLING));
    REQUIRE_FALSE(has_move(list, e1, c1, CASTLING));
}

TEST_CASE("movegen castling: not allowed when king passes through attacked square kingside", "[movegen][castling]") {
    // Black rook on f8 attacks f1 — king would pass through f1
    Position pos = make_pos("5r2/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE_FALSE(has_move(list, e1, g1, CASTLING));
    REQUIRE(has_move(list, e1, c1, CASTLING));
}

TEST_CASE("movegen castling: not allowed when king lands on attacked square", "[movegen][castling]") {
    // Black rook on g8 attacks g1 — king would land on g1
    Position pos = make_pos("6r1/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE_FALSE(has_move(list, e1, g1, CASTLING));
    REQUIRE(has_move(list, e1, c1, CASTLING));
}

TEST_CASE("movegen castling: queenside rook attacked but b1 not on king path", "[movegen][castling]") {
    // Black rook attacks b1 — but king only passes through d1 and c1
    // so queenside castling should still be legal
    Position pos = make_pos("1r6/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(has_move(list, e1, c1, CASTLING));
}

// ============================================================================
// Generate Captures Only
// ============================================================================

TEST_CASE("movegen captures: only captures returned", "[movegen][captures]") {
    // White knight on e4, black pawn on d6, otherwise empty
    Position pos = make_pos("8/8/3p4/8/4N3/8/8/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_captures(pos, list);

    // Every move in the list must be a capture (target must be an enemy piece)
    for (i32 i = 0; i < list.size(); i++) {
        Square to = list[i].to();
        bool is_enemy = (pos.piece_on(to) != NO_PIECE
                         && pos.piece_on(to) != pos.piece_on(list[i].from()))
                        || list[i].is_en_passant();
        // Simplified: just check the destination has an enemy piece or is ep
        Colour us = pos.side_to_move();
        bool valid = (pos.pieces(flip(us)) & BB::square_bb(to))
                     || list[i].is_en_passant();
        REQUIRE(valid);
    }
}

TEST_CASE("movegen captures: en passant included in captures", "[movegen][captures]") {
    Position pos = make_pos("8/8/8/3pP3/8/8/8/4K2k w - d6 0 2");
    MoveList list;
    MoveGen::generate_captures(pos, list);

    REQUIRE(has_move(list, e5, d6, EN_PASSANT));
}

TEST_CASE("movegen captures: promotion captures included", "[movegen][captures]") {
    Position pos = make_pos("3r4/4P3/8/8/8/8/8/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_captures(pos, list);

    REQUIRE(has_promo(list, e7, d8, PROMO_QUEEN));
    REQUIRE(has_promo(list, e7, d8, PROMO_ROOK));
    REQUIRE(has_promo(list, e7, d8, PROMO_BISHOP));
    REQUIRE(has_promo(list, e7, d8, PROMO_KNIGHT));
}

TEST_CASE("movegen captures: quiet moves not included", "[movegen][captures]") {
    Position pos = make_pos("8/8/8/8/4N3/8/8/4K2k w - - 0 1");
    MoveList list;
    MoveGen::generate_captures(pos, list);

    // No enemies on the board so captures list should be empty
    // (king captures also empty since no adjacent enemies)
    REQUIRE(list.empty());
}

// ============================================================================
// Starting Position Move Count
// ============================================================================

TEST_CASE("movegen: starting position has 20 pseudo-legal moves", "[movegen][startpos]") {
    Position pos = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    // 16 pawn moves (8 single + 8 double) + 4 knight moves
    REQUIRE(list.size() == 20);
}

TEST_CASE("movegen: starting position black has 20 pseudo-legal moves", "[movegen][startpos]") {
    Position pos = make_pos("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    MoveList list;
    MoveGen::generate_moves(pos, list);

    REQUIRE(list.size() == 20);
}
