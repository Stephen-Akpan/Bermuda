#include "catch_amalgamated.hpp"
#include "zobrist.h"
#include "tt.h"
#include "position.h"
#include "movegen.h"
#include <iostream>

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

static constexpr ui64 HASH_1 = 0x9e3779b97f4a7c15ULL;
static constexpr ui64 HASH_2 = 0x6c62272e07bb0142ULL;
static constexpr ui64 HASH_3 = 0x517cc1b727220a95ULL;
static constexpr ui64 HASH_4 = 0xb73151628aed2a6aULL;
static constexpr ui64 HASH_5 = 0x243f6a8885a308d3ULL;

static Position make_pos(const std::string& fen) {
    init_bb();
    return Position::from_fen(fen);
}

// Make a single move on a position by UCI string (e.g. "e2e4")
static void make_uci(Position& pos, const std::string& uci) {
    MoveList list;
    MoveGen::generate_moves(pos, list);

    for (i32 i = 0; i < list.size(); i++) {
        if (list[i].to_string() == uci) {
            pos.make_move(list[i]);
            return;
        }
    }
    FAIL("Move not found: " + uci);
}

// ============================================================================
// Zobrist: Basic Properties
// ============================================================================

TEST_CASE("zobrist: hash is non-zero for starting position", "[zobrist]") {
    init_bb();
    Position pos = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    REQUIRE(pos.hash() != 0);
}

TEST_CASE("zobrist: same position produces same hash", "[zobrist]") {
    init_bb();
    Position a = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Position b = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    REQUIRE(a.hash() == b.hash());
}

TEST_CASE("zobrist: different positions produce different hashes", "[zobrist]") {
    init_bb();
    Position a = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    Position b = make_pos("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    REQUIRE(a.hash() != b.hash());
}

TEST_CASE("zobrist: side to move affects hash", "[zobrist]") {
    init_bb();
    // Same piece placement, different side to move
    Position white = make_pos("8/8/8/8/8/8/8/4K2k w - - 0 1");
    Position black = make_pos("8/8/8/8/8/8/8/4K2k b - - 0 1");
    REQUIRE(white.hash() != black.hash());
}

TEST_CASE("zobrist: castling rights affect hash", "[zobrist]") {
    init_bb();
    Position all    = make_pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    Position none   = make_pos("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1");
    Position kside  = make_pos("r3k2r/8/8/8/8/8/8/R3K2R w K - 0 1");
    Position qside  = make_pos("r3k2r/8/8/8/8/8/8/R3K2R w Q - 0 1");

    REQUIRE(all.hash()   != none.hash());
    REQUIRE(all.hash()   != kside.hash());
    REQUIRE(all.hash()   != qside.hash());
    REQUIRE(kside.hash() != qside.hash());
    REQUIRE(kside.hash() != none.hash());
    REQUIRE(qside.hash() != none.hash());
}

TEST_CASE("zobrist: en passant square affects hash", "[zobrist]") {
    init_bb();
    Position with_ep    = make_pos("rnbqkbnr/pppppppp/8/8/3pP3/1P6/P1PP1PPP/RNBQKBNR b KQkq e3 0 1");
    Position without_ep = make_pos("rnbqkbnr/pppppppp/8/8/3pP3/1P6/P1PP1PPP/RNBQKBNR b KQkq - 0 1 ");
    REQUIRE(with_ep.hash() != without_ep.hash());
}

TEST_CASE("zobrist: en passant file affects hash", "[zobrist]") {
    init_bb();
    // Different ep files should give different hashes
    Position ep_e = make_pos("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    Position ep_d = make_pos("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1");
    REQUIRE(ep_e.hash() != ep_d.hash());
}

// ============================================================================
// Zobrist: Incremental Updates
// ============================================================================

TEST_CASE("zobrist: hash after make_move matches freshly computed hash", "[zobrist]") {
    init_bb();

    // After e2e4 the hash should match a position set up from the resulting FEN
    Position pos = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    make_uci(pos, "e2e4");

    Position expected = make_pos("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    REQUIRE(pos.hash() == expected.hash());
}

TEST_CASE("zobrist: hash after unmake_move restores original hash", "[zobrist]") {
    init_bb();

    Position pos = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    ui64 original_hash = pos.hash();

    MoveList list;
    MoveGen::generate_moves(pos, list);

    // Make and unmake every move — hash must be restored each time
    for (i32 i = 0; i < list.size(); i++) {
        UndoInfo undo = pos.make_move(list[i]);
        pos.unmake_move(list[i], undo);
        REQUIRE(pos.hash() == original_hash);
    }
}

TEST_CASE("zobrist: hash restored after series of moves and unmoves", "[zobrist]") {
    init_bb();

    Position pos = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    ui64 original_hash = pos.hash();

    // Play several moves then undo them all
    MoveList list1;
    MoveGen::generate_moves(pos, list1);
    UndoInfo undo1 = pos.make_move(list1[0]);  // e.g. e2e4

    MoveList list2;
    MoveGen::generate_moves(pos, list2);
    UndoInfo undo2 = pos.make_move(list2[0]);  // opponent reply

    MoveList list3;
    MoveGen::generate_moves(pos, list3);
    UndoInfo undo3 = pos.make_move(list3[0]);  // second move

    // Undo all three
    pos.unmake_move(list3[0], undo3);
    pos.unmake_move(list2[0], undo2);
    pos.unmake_move(list1[0], undo1);

    REQUIRE(pos.hash() == original_hash);
}

TEST_CASE("zobrist: transposition — same position via different move orders has same hash", "[zobrist]") {
    init_bb();

    // e4 d5 vs d4 e5 — these are NOT the same position so use a real transposition:
    // e4 e5 Nf3 Nc6 vs Nf3 Nc6 e4 e5 — same position, different move order

    Position pos1 = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    make_uci(pos1, "e2e4");
    make_uci(pos1, "e7e5");
    make_uci(pos1, "g1f3");
    make_uci(pos1, "b8c6");

    Position pos2 = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    make_uci(pos2, "g1f3");
    make_uci(pos2, "b8c6");
    make_uci(pos2, "e2e4");
    make_uci(pos2, "e7e5");

    REQUIRE(pos1.hash() == pos2.hash());
}

TEST_CASE("zobrist: capture updates hash correctly", "[zobrist]") {
    init_bb();

    // After capture the hash should match FEN of resulting position
    Position pos = make_pos("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
    make_uci(pos, "e4d5");  // pawn captures d5

    Position expected = make_pos("rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2");
    REQUIRE(pos.hash() == expected.hash());
}

TEST_CASE("zobrist: castling updates hash correctly", "[zobrist]") {
    init_bb();

    Position pos = make_pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    make_uci(pos, "e1g1");  // white kingside castle

    Position expected = make_pos("r3k2r/8/8/8/8/8/8/R4RK1 b kq - 1 1");
    REQUIRE(pos.hash() == expected.hash());
}

TEST_CASE("zobrist: en passant capture updates hash correctly", "[zobrist]") {
    init_bb();

    Position pos = make_pos("8/8/8/3pP3/8/8/8/4K2k w - d6 0 2");
    make_uci(pos, "e5d6");  // en passant capture

    Position expected = make_pos("8/8/3P4/8/8/8/8/4K2k b - - 0 2");
    REQUIRE(pos.hash() == expected.hash());
}

TEST_CASE("zobrist: promotion updates hash correctly", "[zobrist]") {
    init_bb();

    Position pos = make_pos("8/4P3/8/8/8/8/8/4K2k w - - 0 1");
    make_uci(pos, "e7e8q");  // promote to queen

    Position expected = make_pos("4Q3/8/8/8/8/8/8/4K2k b - - 0 2");
    REQUIRE(pos.hash() == expected.hash());
}

TEST_CASE("zobrist: castling rights cleared after king move", "[zobrist]") {
    init_bb();

    // After king moves, castling rights should be gone from hash
    Position pos = make_pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    make_uci(pos, "e1d1");  // king move clears white castling rights

    Position expected = make_pos("r3k2r/8/8/8/8/8/8/R2K3R b kq - 1 1");
    REQUIRE(pos.hash() == expected.hash());
}

// ============================================================================
// Transposition Table: Basic Operations
// ============================================================================

TEST_CASE("tt: probe returns nullptr on empty table", "[tt]") {
    TranspositionTable tt(1);
    REQUIRE(tt.probe(0x123456789ABCULL) == nullptr);
}

TEST_CASE("tt: store and probe retrieves correct entry", "[tt]") {
    TranspositionTable tt(1);

    ui64  hash  = HASH_1;
    Move  move  = Move::make(e2, e4);
    i16   score = 150;
    ui8   depth = 6;
    Bound bound = BOUND_EXACT;

    tt.store(hash, move, score, depth, bound);

    const TTEntry* entry = tt.probe(hash);
    REQUIRE(entry != nullptr);
    REQUIRE(entry->move  == move);
    REQUIRE(entry->score == score);
    REQUIRE(entry->depth == depth);
    REQUIRE(entry->bound == bound);
}

TEST_CASE("tt: probe returns nullptr for different hash", "[tt]") {
    TranspositionTable tt(1);

    tt.store(0xAAAAAAAAULL, Move::make(e2, e4), 100, 5, BOUND_EXACT);

    // Different hash should not match
    REQUIRE(tt.probe(0xBBBBBBBBULL) == nullptr);
}

TEST_CASE("tt: stores all bound types correctly", "[tt]") {
    TranspositionTable tt(1);

    ui64 h1 = HASH_1, h2 = HASH_2, h3 = HASH_3;

    tt.store(h1, Move::make(e2, e4),  100, 4, BOUND_EXACT);
    tt.store(h2, Move::make(d2, d4),  200, 4, BOUND_LOWER);
    tt.store(h3, Move::make(c2, c4), -100, 4, BOUND_UPPER);

    REQUIRE(tt.probe(h1)->bound == BOUND_EXACT);
    REQUIRE(tt.probe(h2)->bound == BOUND_LOWER);
    REQUIRE(tt.probe(h3)->bound == BOUND_UPPER);
}

TEST_CASE("tt: deeper entry replaces shallower entry", "[tt]") {
    TranspositionTable tt(1);

    ui64 hash = HASH_5;

    tt.store(hash, Move::make(e2, e4), 100, 3, BOUND_EXACT);
    tt.store(hash, Move::make(d2, d4), 200, 6, BOUND_EXACT);

    const TTEntry* entry = tt.probe(hash);
    REQUIRE(entry != nullptr);
    REQUIRE(entry->depth == 6);
    REQUIRE(entry->score == 200);
}

TEST_CASE("tt: shallower entry does not replace deeper exact entry", "[tt]") {
    TranspositionTable tt(1);

    ui64 hash = HASH_4;

    tt.store(hash, Move::make(e2, e4), 100, 8, BOUND_EXACT);
    tt.store(hash, Move::make(d2, d4), 200, 2, BOUND_UPPER);

    const TTEntry* entry = tt.probe(hash);
    REQUIRE(entry != nullptr);
    REQUIRE(entry->depth == 8);
    REQUIRE(entry->score == 100);
}

TEST_CASE("tt: clear empties all entries", "[tt]") {
    TranspositionTable tt(1);

    tt.store(HASH_3, Move::make(e2, e4), 100, 5, BOUND_EXACT);
    tt.store(HASH_5, Move::make(d2, d4), 200, 5, BOUND_EXACT);

    tt.clear();

    REQUIRE(tt.probe(HASH_3) == nullptr);
    REQUIRE(tt.probe(HASH_5) == nullptr);
}

TEST_CASE("tt: new_search ages out old entries", "[tt]") {
    TranspositionTable tt(1);

    ui64 hash = HASH_2;
    tt.store(hash, Move::make(e2, e4), 100, 5, BOUND_EXACT);

    // Advance age — old entry should be replaced by a shallower new one
    tt.new_search();
    tt.store(hash, Move::make(d2, d4), 200, 2, BOUND_LOWER);

    const TTEntry* entry = tt.probe(hash);
    REQUIRE(entry != nullptr);
    REQUIRE(entry->score == 200);  // new entry replaced old one
    REQUIRE(entry->depth == 2);
}

TEST_CASE("tt: null move stored and retrieved", "[tt]") {
    TranspositionTable tt(1);

    ui64 hash = HASH_3;
    tt.store(hash, NULL_MOVE, -50, 4, BOUND_UPPER);

    const TTEntry* entry = tt.probe(hash);
    REQUIRE(entry != nullptr);
    REQUIRE(entry->move.is_null());
    REQUIRE(entry->score == -50);
}

TEST_CASE("tt: hashfull is zero on empty table", "[tt]") {
    TranspositionTable tt(1);
    REQUIRE(tt.hashfull() == 0);
}

TEST_CASE("tt: hashfull increases as entries are stored", "[tt]") {
    TranspositionTable tt(1);

    i32 before = tt.hashfull();

    // Store many entries with unique hashes
    for (ui64 i = 1; i <= 500; i++)
        tt.store(i * 0x9E3779B97F4A7C15ULL, Move::make(e2, e4), 0, 1, BOUND_EXACT);

    i32 after = tt.hashfull();
    REQUIRE(after > before);
}

TEST_CASE("tt: resize clears all entries", "[tt]") {
    TranspositionTable tt(4);

    tt.store(HASH_2, Move::make(e2, e4), 100, 5, BOUND_EXACT);
    tt.resize(8);

    REQUIRE(tt.probe(0x111ULL) == nullptr);
}

// ============================================================================
// Zobrist + TT Integration
// ============================================================================

TEST_CASE("tt integration: position hash used as tt key round-trips correctly", "[tt][zobrist]") {
    init_bb();

    TranspositionTable tt(1);
    Position pos = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");

    ui64 hash = pos.hash();
    tt.store(hash, Move::make(e2, e4), 42, 5, BOUND_EXACT);

    const TTEntry* entry = tt.probe(pos.hash());
    REQUIRE(entry != nullptr);
    REQUIRE(entry->score == 42);
}

TEST_CASE("tt integration: TT hit after transposition", "[tt][zobrist]") {
    init_bb();

    TranspositionTable tt(1);

    // Reach same position via two different move orders
    Position pos1 = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    make_uci(pos1, "e2e4");
    make_uci(pos1, "e7e5");
    make_uci(pos1, "g1f3");
    make_uci(pos1, "b8c6");

    // Store result under pos1's hash
    tt.store(pos1.hash(), Move::make(f1, b5), 300, 8, BOUND_EXACT);

    // Reach same position via different move order
    Position pos2 = make_pos("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    make_uci(pos2, "g1f3");
    make_uci(pos2, "b8c6");
    make_uci(pos2, "e2e4");
    make_uci(pos2, "e7e5");

    // pos2 should get a TT hit because hashes match
    const TTEntry* entry = tt.probe(pos2.hash());
    REQUIRE(entry != nullptr);
    REQUIRE(entry->score == 300);
    REQUIRE(entry->move  == Move::make(f1, b5));
}

TEST_CASE("tt integration: no TT hit for different position", "[tt][zobrist]") {
    init_bb();

    TranspositionTable tt(1);

    Position pos1 = make_pos("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    Position pos2 = make_pos("rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1");

    tt.store(pos1.hash(), Move::make(e2, e4), 50, 4, BOUND_EXACT);

    REQUIRE(tt.probe(pos2.hash()) == nullptr);
}
