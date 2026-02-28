#include "catch_amalgamated.hpp"
#include "eval.h"
#include "position.h"

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

// Evaluate from white's perspective regardless of side to move
static i32 eval_white(const Position& pos) {
    i32 score = Eval::evaluate(pos);
    return (pos.side_to_move() == WHITE) ? score : -score;
}

// ============================================================================
// Symmetry
// ============================================================================

TEST_CASE("eval: starting position is zero", "[eval]") {
    Position pos = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    // Starting position is perfectly symmetric — eval should be 0
    REQUIRE(Eval::evaluate(pos) == 0);
}

TEST_CASE("eval: symmetric position is zero", "[eval]") {
    // Custom symmetric position
    Position pos = make_pos("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    REQUIRE(Eval::evaluate(pos) == 0);
}

TEST_CASE("eval: mirrored position scores equal for both sides", "[eval]") {
    // White to move in one, black to move in the other — should both return same magnitude
    Position white_pos = make_pos("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    Position black_pos = make_pos("4k3/4p3/8/8/8/8/8/4K3 b - - 0 1");

    // Both are "one pawn up for side to move" — scores should match
    REQUIRE(Eval::evaluate(white_pos) == Eval::evaluate(black_pos));
}

// ============================================================================
// Material Imbalance — White Ahead
// ============================================================================

TEST_CASE("eval: white up a queen", "[eval][material]") {
    // White has queen, black does not
    Position pos = make_pos("4k3/8/8/8/8/8/8/3QK3 w - - 0 1");
    i32 score = eval_white(pos);
    REQUIRE(score > Eval::QUEEN_VALUE - 50);   // at least queen value minus small PST variance
    REQUIRE(score < Eval::QUEEN_VALUE + 50);
}

TEST_CASE("eval: white up a rook", "[eval][material]") {
    Position pos = make_pos("4k3/8/8/8/8/8/8/3RK3 w - - 0 1");
    i32 score = eval_white(pos);
    REQUIRE(score > Eval::ROOK_VALUE - 50);
    REQUIRE(score < Eval::ROOK_VALUE + 50);
}

TEST_CASE("eval: white up a bishop", "[eval][material]") {
    Position pos = make_pos("rn1qkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    i32 score = eval_white(pos);
    REQUIRE(score > Eval::BISHOP_VALUE - 50);
    REQUIRE(score < Eval::BISHOP_VALUE + 50);
}

TEST_CASE("eval: white up a knight", "[eval][material]") {
    Position pos = make_pos("r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    i32 score = eval_white(pos);
    REQUIRE(score > Eval::KNIGHT_VALUE - 50);
    REQUIRE(score < Eval::KNIGHT_VALUE + 50);
}

TEST_CASE("eval: white up a pawn", "[eval][material]") {
    Position pos = make_pos("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    i32 score = eval_white(pos);
    REQUIRE(score > Eval::PAWN_VALUE - 50);
    REQUIRE(score < Eval::PAWN_VALUE + 100);
}

// ============================================================================
// Material Imbalance — Black Ahead
// ============================================================================

TEST_CASE("eval: black up a queen — white perspective is negative", "[eval][material]") {
    Position pos = make_pos("3qk3/8/8/8/8/8/8/4K3 w - - 0 1");
    i32 score = eval_white(pos);
    REQUIRE(score < -Eval::QUEEN_VALUE + 50);
}

TEST_CASE("eval: black up a rook — white perspective is negative", "[eval][material]") {
    Position pos = make_pos("3rk3/8/8/8/8/8/8/4K3 w - - 0 1");
    i32 score = eval_white(pos);
    REQUIRE(score < -Eval::ROOK_VALUE + 50);
}

TEST_CASE("eval: black up a pawn — white perspective is negative", "[eval][material]") {
    Position pos = make_pos("4k3/4p3/8/8/8/8/8/4K3 w - - 0 1");
    i32 score = eval_white(pos);
    REQUIRE(score < -Eval::PAWN_VALUE + 50);
}

// ============================================================================
// Large Imbalances
// ============================================================================

TEST_CASE("eval: white up queen and rook", "[eval][material]") {
    Position pos = make_pos("4k3/8/8/8/8/8/8/2RQK3 w - - 0 1");
    i32 score = eval_white(pos);
    i32 expected = Eval::QUEEN_VALUE + Eval::ROOK_VALUE;
    REQUIRE(score > expected - 100);
    REQUIRE(score < expected + 100);
}

TEST_CASE("eval: white up full army vs bare king", "[eval][material]") {
    // White has all pieces except king, black has bare king
    Position pos = make_pos("4k3/8/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1");
    i32 score = eval_white(pos);
    // Should be massively positive
    REQUIRE(score > 3000);
}

TEST_CASE("eval: black up full army vs bare king", "[eval][material]") {
    Position pos = make_pos("rnbqkbnr/pppppppp/8/8/8/8/8/4K3 w - - 0 1");
    i32 score = eval_white(pos);
    // Should be massively negative
    REQUIRE(score < -3000);
}

TEST_CASE("eval: white up two rooks vs queen", "[eval][material]") {
    // Two rooks (1000) slightly more than queen (900)
    Position pos = make_pos("4k3/8/8/3q4/8/8/8/2RRK3 w - - 0 1");
    i32 score = eval_white(pos);
    // 2 rooks - 1 queen = 1000 - 900 = 100 centipawns approximately
    REQUIRE(score > 0);
}

TEST_CASE("eval: material function matches piece counts", "[eval][material]") {
    // White has queen+rook, black has two knights — material difference
    Position pos = make_pos("4k3/8/8/2nn4/8/8/8/2RQK3 w - - 0 1");
    i32 mat = Eval::material(pos);
    i32 expected = Eval::QUEEN_VALUE + Eval::ROOK_VALUE
                 - 2 * Eval::KNIGHT_VALUE;
    REQUIRE(mat == expected);
}

// ============================================================================
// Side to Move Perspective
// ============================================================================

TEST_CASE("eval: score flips with side to move", "[eval]") {
    // Same position, different side to move — scores should be negations
    Position white = make_pos("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    Position black = make_pos("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1");

    REQUIRE(Eval::evaluate(white) == -Eval::evaluate(black));
}

// ============================================================================
// Bishop Pair
// ============================================================================

TEST_CASE("eval: bishop pair bonus applied for white", "[eval]") {
    // White has two bishops, black has two knights — same nominal value but bishop pair bonus
    Position pos = make_pos("1n2k1n1/8/8/8/8/8/8/2B1KB2 w - - 0 1 ");
    i32 score = eval_white(pos);
    // Bishops (330 each) vs Knights (320 each) = 20cp material + 30cp bishop pair
    REQUIRE(score > 0);
}

TEST_CASE("eval: bishop pair bonus applied for black", "[eval]") {
    // Black has two bishops, white has two knights
    Position pos = make_pos("4k1bb/8/8/8/8/8/8/2NNK3 w - - 0 1");
    i32 score = eval_white(pos);
    // Black should be slightly ahead due to bishop pair
    REQUIRE(score < 0);
}

TEST_CASE("eval: single bishop gets no pair bonus", "[eval]") {
    Position with_pair    = make_pos("4k3/8/8/8/8/8/8/2BBK3 w - - 0 1");
    Position without_pair = make_pos("4k3/8/8/8/8/8/8/3BK3 w - - 0 1");

    i32 with_score    = eval_white(with_pair);
    i32 without_score = eval_white(without_pair);

    // With pair should score higher than single bishop by approximately
    // one bishop value + bishop pair bonus
    REQUIRE(with_score > without_score);
    REQUIRE(with_score - without_score > Eval::BISHOP_VALUE);
}
