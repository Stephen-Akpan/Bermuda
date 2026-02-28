#include "catch_amalgamated.hpp"
#include "perft.h"
#include "position.h"
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

static Perft::Result run(const std::string& fen, i32 depth) {
    init_bb();
    Position pos = Position::from_fen(fen);
    return Perft::perft(pos, depth);
}

 // ============================================================================
 // Position 1: Starting Position
 // ============================================================================
 // All known values from https://www.chessprogramming.org/Perft_Results

 TEST_CASE("perft: startpos depth 1", "[perft][startpos]") {
     auto r = run("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1);
     REQUIRE(r.nodes == 20);
 }

 TEST_CASE("perft: startpos depth 2", "[perft][startpos]") {
     auto r = run("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 2);
     REQUIRE(r.nodes == 400);
 }

 TEST_CASE("perft: startpos depth 3", "[perft][startpos]") {
     auto r = run("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 3);
     REQUIRE(r.nodes      == 8902);
     REQUIRE(r.captures   == 34);
     REQUIRE(r.ep         == 0);
     REQUIRE(r.castles    == 0);
     REQUIRE(r.promotions == 0);
     REQUIRE(r.checks     == 12);
 }

 TEST_CASE("perft: startpos depth 4", "[perft][startpos]") {
     auto r = run("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4);
     REQUIRE(r.nodes      == 197281);
     REQUIRE(r.captures   == 1576);
     REQUIRE(r.ep         == 0);
     REQUIRE(r.castles    == 0);
     REQUIRE(r.promotions == 0);
     REQUIRE(r.checks     == 469);
 }

 TEST_CASE("perft: startpos depth 5", "[perft][startpos]") {
     auto r = run("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5);
     REQUIRE(r.nodes      == 4865609);
     REQUIRE(r.captures   == 82719);
     REQUIRE(r.ep         == 258);
     REQUIRE(r.castles    == 0);
     REQUIRE(r.promotions == 0);
     REQUIRE(r.checks     == 27351);
 }

 // ============================================================================
 // Position 2: Kiwipete
 // Complex position with castling, en passant, and promotions available
 // ============================================================================

 TEST_CASE("perft: kiwipete depth 1", "[perft][kiwipete]") {
     auto r = run("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1);
     REQUIRE(r.nodes == 48);
 }

 TEST_CASE("perft: kiwipete depth 2", "[perft][kiwipete]") {
     auto r = run("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 2);
     REQUIRE(r.nodes      == 2039);
     REQUIRE(r.captures   == 351);
     REQUIRE(r.ep         == 1);
     REQUIRE(r.castles    == 91);
     REQUIRE(r.promotions == 0);
     REQUIRE(r.checks     == 3);
 }

 TEST_CASE("perft: kiwipete depth 3", "[perft][kiwipete]") {
     auto r = run("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3);
     REQUIRE(r.nodes      == 97862);
     REQUIRE(r.captures   == 17102);
     REQUIRE(r.ep         == 45);
     REQUIRE(r.castles    == 3162);
     REQUIRE(r.promotions == 0);
     REQUIRE(r.checks     == 993);
 }

 TEST_CASE("perft: kiwipete depth 4", "[perft][kiwipete]") {
     auto r = run("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4);
     REQUIRE(r.nodes      == 4085603);
     REQUIRE(r.captures   == 757163);
     REQUIRE(r.ep         == 1929);
     REQUIRE(r.castles    == 128013);
     REQUIRE(r.promotions == 15172);
     REQUIRE(r.checks     == 25523);
 }

 // ============================================================================
 // Position 3: Endgame-like with promotions
 // ============================================================================

 TEST_CASE("perft: position 3 depth 1", "[perft][pos3]") {
     auto r = run("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 1);
     REQUIRE(r.nodes == 14);
 }

 TEST_CASE("perft: position 3 depth 2", "[perft][pos3]") {
     auto r = run("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 2);
     REQUIRE(r.nodes == 191);
 }

 TEST_CASE("perft: position 3 depth 3", "[perft][pos3]") {
     auto r = run("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3);
     REQUIRE(r.nodes      == 2812);
     REQUIRE(r.ep         == 2);
     REQUIRE(r.promotions == 0);
 }

 TEST_CASE("perft: position 3 depth 4", "[perft][pos3]") {
     auto r = run("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4);
     REQUIRE(r.nodes      == 43238);
     REQUIRE(r.captures   == 3348);
     REQUIRE(r.ep         == 123);
     REQUIRE(r.promotions == 0);
     REQUIRE(r.checks     == 1680);
 }

 TEST_CASE("perft: position 3 depth 5", "[perft][pos3]") {
     auto r = run("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5);
     REQUIRE(r.nodes      == 674624);
     REQUIRE(r.captures   == 52051);
     REQUIRE(r.ep         == 1165);
     REQUIRE(r.promotions == 0);
     REQUIRE(r.checks     == 52950);
 }

 // ============================================================================
 // Position 4: Castling and promotion heavy
 // ============================================================================

 TEST_CASE("perft: position 4 depth 1", "[perft][pos4]") {
     auto r = run("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 1);
     REQUIRE(r.nodes == 6);
 }

 TEST_CASE("perft: position 4 depth 2", "[perft][pos4]") {
     auto r = run("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 2);
     REQUIRE(r.nodes == 264);
     REQUIRE(r.captures == 87);
     REQUIRE(r.promotions == 48);
     REQUIRE(r.checks == 10);
 }

 TEST_CASE("perft: position 4 depth 3", "[perft][pos4]") {
     auto r = run("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3);
     REQUIRE(r.nodes      == 9467);
     REQUIRE(r.ep    == 4);
     REQUIRE(r.promotions == 120);
     REQUIRE(r.checks == 38);
 }

 TEST_CASE("perft: position 4 depth 4", "[perft][pos4]") {
     auto r = run("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4);
     REQUIRE(r.nodes      == 422333);
     REQUIRE(r.captures   == 131393);
     REQUIRE(r.castles    == 7795);
     REQUIRE(r.promotions == 60032);
     REQUIRE(r.checks     == 15492);
 }

 // ============================================================================
 // Position 5: Tricky position (pins, discovered checks)
 // ============================================================================

 TEST_CASE("perft: position 5 depth 1", "[perft][pos5]") {
     auto r = run("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 1);
     REQUIRE(r.nodes == 44);
 }

 TEST_CASE("perft: position 5 depth 2", "[perft][pos5]") {
     auto r = run("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 2);
     REQUIRE(r.nodes == 1486);
 }

 TEST_CASE("perft: position 5 depth 3", "[perft][pos5]") {
     auto r = run("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3);
     REQUIRE(r.nodes == 62379);
 }

 TEST_CASE("perft: position 5 depth 4", "[perft][pos5]") {
     auto r = run("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4);
     REQUIRE(r.nodes == 2103487);
 }

 // ============================================================================
 // Position 6: Steven Edwards position
 // ============================================================================

 TEST_CASE("perft: position 6 depth 1", "[perft][pos6]") {
     auto r = run("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 1);
     REQUIRE(r.nodes == 46);
 }

 TEST_CASE("perft: position 6 depth 2", "[perft][pos6]") {
     auto r = run("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 2);
     REQUIRE(r.nodes == 2079);
 }

 TEST_CASE("perft: position 6 depth 3", "[perft][pos6]") {
     auto r = run("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3);
     REQUIRE(r.nodes == 89890);
 }

 TEST_CASE("perft: position 6 depth 4", "[perft][pos6]") {
     auto r = run("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 4);
     REQUIRE(r.nodes == 3894594);
 }
