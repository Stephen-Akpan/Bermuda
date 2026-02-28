#include "catch_amalgamated.hpp"
#include "search.h"
#include "position.h"
#include "tt.h"
#include "zobrist.h"

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

static Move run_search(const std::string& fen, int depth) {
    init_bb();
    TT.clear();
    TT.new_search();

    SearchInfo info;
    info.pos          = Position::from_fen(fen);
    info.limits.depth = depth;
    info.limits.infinite = true;

    return Search::start(info);
}

static int run_score(const std::string& fen, int depth) {
    init_bb();
    TT.clear();
    TT.new_search();

    SearchInfo info;
    info.pos          = Position::from_fen(fen);
    info.limits.depth = depth;
    info.limits.infinite = true;

    Search::start(info);
    return info.best_score;
}

// ============================================================================
// Mate in One
// ============================================================================

TEST_CASE("search: mate in one — back rank", "[search][mate]") {
    // White rook delivers back rank mate
    init_bb();
    Position pos = Position::from_fen("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1 ");

    SearchInfo info;
    info.pos          = pos;
    info.limits.depth = 2;
    info.limits.infinite = true;

    TT.clear();
    TT.new_search();
    Move best = Search::start(info);

    // Best move must deliver checkmate — rook to a8
    REQUIRE(best.to_string() == "a1a8");
    REQUIRE(is_mate_score(info.best_score));
    REQUIRE(info.best_score > 0);
}

TEST_CASE("search: mate in one — queen delivers checkmate", "[search][mate]") {
    // Qa7# — black king is trapped on a8
    // White: Ka6, Qa7 delivers mate
    init_bb();
    Position pos = Position::from_fen("3k4/8/3K4/8/8/8/8/7Q w - - 0 1 ");

    SearchInfo info;
    info.pos          = pos;
    info.limits.depth = 2;
    info.limits.infinite = true;

    TT.clear();
    TT.new_search();
    Search::start(info);

    REQUIRE(is_mate_score(info.best_score));
    REQUIRE(info.best_score > 0);
}

// ============================================================================
// Mate in Two
// ============================================================================

TEST_CASE("search: mate in two — finds forced mate", "[search][mate]") {
    // Classic ladder mate setup — two rooks
    // White to move, mate in 2
    Position pos = Position::from_fen("6k1/8/8/8/8/8/1RR5/6K1 w - - 0 1");
    init_bb();

    SearchInfo info;
    info.pos          = pos;
    info.limits.depth = 4;
    info.limits.infinite = true;

    TT.clear();
    TT.new_search();
    Search::start(info);

    // Score should be mate — exact move sequence doesn't matter, just that it finds mate
    REQUIRE(is_mate_score(info.best_score));
    REQUIRE(info.best_score > 0);
}

// ============================================================================
// Stalemate Avoidance
// ============================================================================

TEST_CASE("search: avoids stalemate when winning", "[search][draw]") {
    init_bb();
    Position pos = Position::from_fen("k7/8/1QK5/8/8/8/8/8 w - - 0 1");

    SearchInfo info;
    info.pos          = pos;
    info.limits.depth = 4;
    info.limits.infinite = true;

    TT.clear();
    TT.new_search();
    Search::start(info);

    // Engine should find mate, not stalemate (score must be positive mate score)
    REQUIRE(is_mate_score(info.best_score));
    REQUIRE(info.best_score > 0);
}

// ============================================================================
// Draw Detection
// ============================================================================

TEST_CASE("search: detects draw by fifty-move rule", "[search][draw]") {
    // Position where fifty-move rule applies immediately
    // Halfmove clock already at 100 — this is a draw
    Position pos = Position::from_fen("4k3/4p3/8/8/8/8/4P3/4K3 w - - 100 1");
    init_bb();

    SearchInfo info;
    info.pos          = pos;
    info.limits.depth = 4;
    info.limits.infinite = true;

    TT.clear();
    TT.new_search();
    Search::start(info);

    // Score should be 0 — pure draw
    REQUIRE(info.best_score == 0);
}

// ============================================================================
// Material Win
// ============================================================================

TEST_CASE("search: finds winning capture", "[search][tactical]") {
    // White can capture a free queen with rook
    Position pos = Position::from_fen("3q2k1/8/8/8/8/8/8/3RK3 w - - 0 1 ");
    init_bb();

    SearchInfo info;
    info.pos          = pos;
    info.limits.depth = 4;
    info.limits.infinite = true;

    TT.clear();
    TT.new_search();
    Move best = Search::start(info);

    // Should capture the queen
    REQUIRE(best.to() == d8);
    REQUIRE(info.best_score > Eval::QUEEN_VALUE / 2);
}

// ============================================================================
// Repetition
// ============================================================================

TEST_CASE("search: KVK is drawn", "[search][draw]") {
    init_bb();
    TT.clear();
    TT.new_search();

    SearchInfo info;
    info.pos = Position::from_fen("6k1/8/8/8/8/8/8/4K3 w - - 0 1 ");
    info.limits.depth = 4;
    info.limits.infinite = true;

    Search::start(info);

    REQUIRE(info.best_score == 0);
}
