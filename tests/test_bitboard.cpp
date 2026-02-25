#include "bitboard.h"
#include "types.h"
#include "catch_amalgamated.hpp"

// Helper to initialize bitboard tables before tests that need them
inline void init_bb() {
  static bool initialized = false;
  if (!initialized) {
    BB::init();
    assert(BB::bishop_attack(e4, 0ULL) & (1ULL << b1));
    initialized = true;
  }
}

// ============================================================================
// Basic Bit Utilities
// ============================================================================

TEST_CASE("BitUtilities / PopcountZero") {
  CHECK(popcount(0ULL) == 0);
}

TEST_CASE("BitUtilities / PopcountOne") {
  CHECK(popcount(1ULL) == 1);
  CHECK(popcount(0x8000000000000000ULL) == 1);
}

TEST_CASE("BitUtilities / PopcountAllOnes") {
  CHECK(popcount(0xFFFFFFFFFFFFFFFFULL) == 64);
}

TEST_CASE("BitUtilities / PopcountSparsePattern") {
  // Pattern: 0x5555... (every other bit set)
  CHECK(popcount(0x5555555555555555ULL) == 32);
}

TEST_CASE("BitUtilities / LSBSingleBit") {
  CHECK(lsb(0x1ULL) == 0);
  CHECK(lsb(0x2ULL) == 1);
  CHECK(lsb(0x4ULL) == 2);
  CHECK(lsb(0x8000000000000000ULL) == 63);
}

TEST_CASE("BitUtilities / LSBMultipleBits") {
  CHECK(lsb(0xFFULL) == 0);      // Lowest bit is 0
  CHECK(lsb(0xFF00ULL) == 8);    // Lowest bit is 8
}

TEST_CASE("BitUtilities / MSBSingleBit") {
  CHECK(msb(0x1ULL) == 0);
  CHECK(msb(0x2ULL) == 1);
  CHECK(msb(0x8000000000000000ULL) == 63);
}

TEST_CASE("BitUtilities / MSBMultipleBits") {
  CHECK(msb(0xFFULL) == 7);
  CHECK(msb(0xFF00ULL) == 15);
}

TEST_CASE("BitUtilities / PopLSB") {
  Bitboard b = 0b10101010ULL;
  ui8 first = pop_lsb(b);
  CHECK(first == 1);  // First set bit is at position 1
  CHECK(b == 0b10101000ULL);  // LSB cleared
}

TEST_CASE("BitUtilities / TestBit") {
  Bitboard b = 0b00001111ULL;
  CHECK(test_bit(b, (Square)0));
  CHECK(test_bit(b, (Square)3));
  CHECK(!test_bit(b, (Square)4));
  CHECK(!test_bit(b, (Square)7));
}

TEST_CASE("BitUtilities / SetBit") {
  Bitboard b = 0;
  set_bit(b, (Square)5);
  CHECK(b == 0b00100000ULL);
  set_bit(b, (Square)2);
  CHECK(b == 0b00100100ULL);
}

TEST_CASE("BitUtilities / ClearBit") {
  Bitboard b = 0b11111111ULL;
  clear_bit(b, (Square)3);
  CHECK(b == 0b11110111ULL);
  clear_bit(b, (Square)0);
  CHECK(b == 0b11110110ULL);
}

TEST_CASE("BitUtilities / ToggleBit") {
  Bitboard b = 0;
  toggle_bit(b, (Square)0);
  CHECK(b == 1);
  toggle_bit(b, (Square)0);
  CHECK(b == 0);
}

// ============================================================================
// Square & File/Rank Utilities
// ============================================================================

TEST_CASE("SquareUtilities / FileRankExtraction") {
  CHECK(file(a1) == 0);
  CHECK(rank(a1) == 0);
  
  CHECK(file(h1) == 7);
  CHECK(rank(h1) == 0);
  
  CHECK(file(a8) == 0);
  CHECK(rank(a8) == 7);
  
  CHECK(file(h8) == 7);
  CHECK(rank(h8) == 7);
}

TEST_CASE("SquareUtilities / MakeSquare") {
  CHECK(make_square(0, 0) == a1);
  CHECK(make_square(7, 0) == h1);
  CHECK(make_square(0, 7) == a8);
  CHECK(make_square(7, 7) == h8);
  CHECK(make_square(4, 4) == e5);
}

TEST_CASE("SquareUtilities / RoundTrip") {
  for (int sq = 0; sq < 64; sq++) {
    Square s = (Square)sq;
    ui8 f = file(s);
    ui8 r = rank(s);
    Square reconstructed = make_square(f, r);
    CHECK(reconstructed == s);
  }
}

// ============================================================================
// Bitboard Masks
// ============================================================================

TEST_CASE("BitboardMasks / RankMasks") {
  init_bb();
  CHECK(BB::RANK_1 == 0x00000000000000FFULL);
  CHECK(BB::RANK_8 == 0xFF00000000000000ULL);
  CHECK(popcount(BB::RANK_1) == 8);
  CHECK(popcount(BB::RANK_8) == 8);
}

TEST_CASE("BitboardMasks / FileMasks") {
  init_bb();
  CHECK(BB::FILE_A == 0x0101010101010101ULL);
  CHECK(BB::FILE_H == 0x8080808080808080ULL);
  CHECK(popcount(BB::FILE_A) == 8);
  CHECK(popcount(BB::FILE_H) == 8);
}

TEST_CASE("BitboardMasks / SquareBB") {
  init_bb();
  CHECK(BB::square_bb(a1) == 1ULL);
  CHECK(BB::square_bb(h1) == 0x80ULL);
  CHECK(BB::square_bb(a8) == 0x0100000000000000ULL);
  CHECK(popcount(BB::square_bb((Square)33)) == 1);
}

TEST_CASE("BitboardMasks / RankFileContainSquare") {
  init_bb();
  Bitboard r4 = BB::rank_bb(d4);
  CHECK(test_bit(r4, d4));
  CHECK(test_bit(r4, a4));
  CHECK(test_bit(r4, h4));
  CHECK(!test_bit(r4, d5));
  
  Bitboard fe = BB::file_bb(e4);
  CHECK(test_bit(fe, e1));
  CHECK(test_bit(fe, e4));
  CHECK(test_bit(fe, e8));
  CHECK(!test_bit(fe, d4));
}

// ============================================================================
// Pawn Attacks
// ============================================================================

TEST_CASE("PawnAttacks / WhitePawnPush") {
  init_bb();
  Bitboard pawns = BB::square_bb(e2);
  Bitboard pushed = BB::pawn_push(pawns, WHITE);
  CHECK(pushed == BB::square_bb(e3));
}

TEST_CASE("PawnAttacks / BlackPawnPush") {
  init_bb();
  Bitboard pawns = BB::square_bb(e7);
  Bitboard pushed = BB::pawn_push(pawns, BLACK);
  CHECK(pushed == BB::square_bb(e6));
}

TEST_CASE("PawnAttacks / WhitePawnAttacks") {
  init_bb();
  Bitboard pawns = BB::square_bb(e4);
  Bitboard attacks = BB::pawn_attacks(pawns, WHITE);
  
  // White pawn on e4 attacks d5 and f5
  CHECK(test_bit(attacks, d5));
  CHECK(test_bit(attacks, f5));
  CHECK(!test_bit(attacks, e5));
}

TEST_CASE("PawnAttacks / BlackPawnAttacks") {
  init_bb();
  Bitboard pawns = BB::square_bb(e4);
  Bitboard attacks = BB::pawn_attacks(pawns, BLACK);
  
  // Black pawn on e4 attacks d3 and f3
  CHECK(test_bit(attacks, d3));
  CHECK(test_bit(attacks, f3));
  CHECK(!test_bit(attacks, e3));
}

// ============================================================================
// Knight Attacks
// ============================================================================

TEST_CASE("KnightAttacks / CenterBoard") {
  init_bb();
  // Knight on e4 can move to 8 squares
  Bitboard attacks = BB::knight_attack(e4);
  CHECK(popcount(attacks) == 8);
  CHECK(test_bit(attacks, d6));
  CHECK(test_bit(attacks, f6));
  CHECK(test_bit(attacks, c5));
  CHECK(test_bit(attacks, g5));
  CHECK(test_bit(attacks, c3));
  CHECK(test_bit(attacks, g3));
  CHECK(test_bit(attacks, d2));
  CHECK(test_bit(attacks, f2));
}

TEST_CASE("KnightAttacks / Corner") {
  init_bb();
  // Knight on a1 can only move to 2 squares
  Bitboard attacks = BB::knight_attack(a1);
  CHECK(popcount(attacks) == 2);
  CHECK(test_bit(attacks, b3));
  CHECK(test_bit(attacks, c2));
}

TEST_CASE("KnightAttacks / Edge") {
  init_bb();
  // Knight on a4 can move to 4 squares
  Bitboard attacks = BB::knight_attack(a4);
  CHECK(popcount(attacks) == 4);
  CHECK(test_bit(attacks, b6));
  CHECK(test_bit(attacks, c5));
  CHECK(test_bit(attacks, c3));
  CHECK(test_bit(attacks, b2));
}

// ============================================================================
// King Attacks
// ============================================================================

TEST_CASE("KingAttacks / CenterBoard") {
  init_bb();
  // King on d4 can move to 8 squares
  Bitboard attacks = BB::king_attack(d4);
  CHECK(popcount(attacks) == 8);
}

TEST_CASE("KingAttacks / Corner") {
  init_bb();
  // King on a1 can move to 3 squares
  Bitboard attacks = BB::king_attack(a1);
  CHECK(popcount(attacks) == 3);
  CHECK(test_bit(attacks, a2));
  CHECK(test_bit(attacks, b1));
  CHECK(test_bit(attacks, b2));
}

// ============================================================================
// Bishop Attacks (Magic Bitboards)
// ============================================================================

TEST_CASE("BishopAttacks / EmptyBoard") {
  init_bb();
  // Bishop on e4, empty board
  Bitboard attacks = BB::bishop_attack(e4, 0);
  CHECK(test_bit(attacks, d5));
  CHECK(test_bit(attacks, f5));
  CHECK(test_bit(attacks, c6));
  CHECK(test_bit(attacks, a8));
  CHECK(test_bit(attacks, h1));
}

TEST_CASE("BishopAttacks / WithBlockers") {
  init_bb();
  // Bishop on e4, blocked by piece on g6
  Bitboard blockers = BB::square_bb(g6);
  Bitboard attacks = BB::bishop_attack(e4, blockers);
  CHECK(test_bit(attacks, f5));
  CHECK(test_bit(attacks, g6));  // Blocker square is attacked
  CHECK(!test_bit(attacks, h7));  // Beyond blocker not attacked
}

// ============================================================================
// Rook Attacks (Magic Bitboards)
// ============================================================================

TEST_CASE("RookAttacks / EmptyBoard") {
  init_bb();
  // Rook on e4, empty board
  Bitboard attacks = BB::rook_attack(e4, 0);
  CHECK(test_bit(attacks, e1));
  CHECK(test_bit(attacks, e8));
  CHECK(test_bit(attacks, a4));
  CHECK(test_bit(attacks, h4));
  CHECK(popcount(attacks) == 14);  // 7 on rank + 7 on file
}

TEST_CASE("RookAttacks / WithBlockers") {
  init_bb();
  // Rook on e4, blocked by piece on e6
  Bitboard blockers = BB::square_bb(e6);
  Bitboard attacks = BB::rook_attack(e4, blockers);
  CHECK(test_bit(attacks, e5));
  CHECK(test_bit(attacks, e6));
  CHECK(!test_bit(attacks, e7));
  CHECK(!test_bit(attacks, e8));
}

// ============================================================================
// Queen Attacks (Bishop + Rook)
// ============================================================================

TEST_CASE("QueenAttacks / Combined") {
  init_bb();
  // Queen should attack like bishop + rook combined
  Bitboard bishop_attacks = BB::bishop_attack(e4, 0);
  Bitboard rook_attacks = BB::rook_attack(e4, 0);
  Bitboard queen_attacks = BB::queen_attack(e4, 0);
  
  CHECK(queen_attacks == (bishop_attacks | rook_attacks));
}

// ============================================================================
// Distance Metrics
// ============================================================================

TEST_CASE("Distance / SameSquare") {
  init_bb();
  CHECK(BB::square_distance(e4, e4) == 0);
}

TEST_CASE("Distance / OneSquare") {
  init_bb();
  CHECK(BB::square_distance(e4, e5) == 1);
  CHECK(BB::square_distance(e4, d4) == 1);
  CHECK(BB::square_distance(e4, d5) == 1);
}

TEST_CASE("Distance / Maximum") {
  init_bb();
  CHECK(BB::square_distance(a1, h8) == 7);
  CHECK(BB::square_distance(a1, a8) == 7);
  CHECK(BB::square_distance(a1, h1) == 7);
}
