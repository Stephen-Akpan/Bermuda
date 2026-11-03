#include "catch_amalgamated.hpp"
#include "bitboard.h"
#include "movegen.h"
#include "search.h"
#include "zobrist.h"

namespace {

void initialize_search_tables() {
    static const bool initialized = [] {
        BB::init();
        Zobrist::init();
        return true;
    }();
    (void)initialized;
}

bool is_legal_move(Position& pos, Move move) {
    MoveList moves;
    MoveGen::generate_moves(pos, moves);
    for (i32 i = 0; i < moves.size(); ++i) {
        if (moves[i] != move)
            continue;
        const UndoInfo undo = pos.make_move(move);
        const bool legal = !pos.is_attacked_by(
            pos.king_square(flip(pos.side_to_move())),
            pos.side_to_move());
        pos.unmake_move(move, undo);
        if (legal)
            return true;
    }
    return false;
}

} // namespace

TEST_CASE("search returns a legal evasion from check", "[search]") {
    initialize_search_tables();
    SearchInfo info;
    info.pos = Position::from_fen("4r1k1/8/8/8/8/8/8/4K3 w - - 0 1");
    info.limits.depth = 2;
    REQUIRE(info.pos.in_check());

    const Move best = Search::start(info);

    REQUIRE_FALSE(best.is_null());
    REQUIRE(is_legal_move(info.pos, best));
}

TEST_CASE("search returns no move in checkmate", "[search]") {
    initialize_search_tables();
    SearchInfo info;
    info.pos = Position::from_fen("8/8/8/8/8/2k5/1q6/K7 w - - 0 1");
    info.limits.depth = 2;
    REQUIRE(info.pos.in_check());

    REQUIRE(Search::start(info).is_null());
}

TEST_CASE("search avoids a queen capture that loses the queen", "[search]") {
    initialize_search_tables();
    SearchInfo info;
    info.pos = Position::from_fen("r7/1b5k/8/8/8/8/8/Q3K3 w - - 0 1");
    info.limits.depth = 4;

    const Move best = Search::start(info);

    REQUIRE_FALSE(best.is_null());
    REQUIRE_FALSE((best.from() == a1 && best.to() == a8));
}

TEST_CASE("time management reserves clock and caps search time", "[search]") {
    initialize_search_tables();
    SearchInfo info;
    info.pos.set_startpos();

    info.limits.wtime = 10'000;
    info.limits.btime = 10'000;
    info.limits.winc = 100;
    info.limits.binc = 100;
    info.compute_allocated_time();
    REQUIRE(info.soft_limit_ms > 0);
    REQUIRE(info.hard_limit_ms >= info.soft_limit_ms);
    REQUIRE(info.hard_limit_ms <= 9'800);

    info.limits.wtime = 30'000;
    info.limits.btime = 30'000;
    info.limits.winc = 0;
    info.limits.binc = 0;
    info.limits.movestogo = 0;
    info.compute_allocated_time();
    REQUIRE(info.soft_limit_ms <= 600);
    REQUIRE(info.hard_limit_ms <= 900);

    info.limits.movestogo = 10;
    info.compute_allocated_time();
    const i32 control_soft_limit = info.soft_limit_ms;
    info.limits.movestogo = 0;
    info.compute_allocated_time();
    REQUIRE(control_soft_limit > info.soft_limit_ms);

    info.limits = SearchLimits{};
    info.limits.movetime = 1;
    info.compute_allocated_time();
    REQUIRE(info.soft_limit_ms == 1);
    REQUIRE(info.hard_limit_ms == 1);

    info.limits = SearchLimits{};
    info.limits.wtime = 0;
    info.limits.btime = 5'000;
    info.compute_allocated_time();
    REQUIRE(info.soft_limit_ms == 1);
    REQUIRE(info.hard_limit_ms == 1);
}
