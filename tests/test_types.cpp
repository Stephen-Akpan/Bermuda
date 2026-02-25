#include "types.h"
#include "catch_amalgamated.hpp"

// ============================================================================
// Colour Tests
// ============================================================================

TEST_CASE("Colour / Values") {
  CHECK(WHITE == 0);
  CHECK(BLACK == 1);
}

TEST_CASE("Colour / Flip") {
  CHECK(flip(WHITE) == BLACK);
  CHECK(flip(BLACK) == WHITE);
  CHECK(flip(flip(WHITE)) == WHITE);
}

TEST_CASE("Colour / PawnPushDirection") {
  CHECK(pawn_push_dir(WHITE) == 8);   // North
  CHECK(pawn_push_dir(BLACK) == -8);  // South
}

// ============================================================================
// Square Tests
// ============================================================================

TEST_CASE("Square / EnumValues") {
  CHECK(a1 == 0);
  CHECK(b1 == 1);
  CHECK(h1 == 7);
  CHECK(a2 == 8);
  CHECK(h8 == 63);
  CHECK(NO_SQUARE == 64);
}

TEST_CASE("Square / FileExtraction") {
  CHECK(file(a1) == 0);
  CHECK(file(b1) == 1);
  CHECK(file(h1) == 7);
  CHECK(file(a8) == 0);
  CHECK(file(h8) == 7);
  CHECK(file(d4) == 3);
}

TEST_CASE("Square / RankExtraction") {
  CHECK(rank(a1) == 0);
  CHECK(rank(h1) == 0);
  CHECK(rank(a8) == 7);
  CHECK(rank(h8) == 7);
  CHECK(rank(d4) == 3);
}

TEST_CASE("Square / MakeSquareFromFileRank") {
  CHECK(make_square(0, 0) == a1);
  CHECK(make_square(1, 0) == b1);
  CHECK(make_square(7, 0) == h1);
  CHECK(make_square(0, 7) == a8);
  CHECK(make_square(7, 7) == h8);
  CHECK(make_square(3, 3) == d4);
}

TEST_CASE("Square / Consistency") {
  // Ensure all squares can be decomposed and reconstructed
  for (int i = 0; i < 64; i++) {
    Square sq = (Square)i;
    ui8 f = file(sq);
    ui8 r = rank(sq);
    CHECK(make_square(f, r) == sq);
    CHECK(f < 8);
    CHECK(r < 8);
  }
}

// ============================================================================
// PieceType Tests
// ============================================================================

TEST_CASE("PieceType / EnumValues") {
  CHECK(PAWN == 0);
  CHECK(KNIGHT == 1);
  CHECK(BISHOP == 2);
  CHECK(ROOK == 3);
  CHECK(QUEEN == 4);
  CHECK(KING == 5);
  CHECK(NO_PIECE_TYPE == 6);
}

// ============================================================================
// Piece Tests
// ============================================================================

TEST_CASE("Piece / EnumValues") {
  // White pieces
  CHECK(WHITE_PAWN == 0);
  CHECK(WHITE_KNIGHT == 1);
  CHECK(WHITE_BISHOP == 2);
  CHECK(WHITE_ROOK == 3);
  CHECK(WHITE_QUEEN == 4);
  CHECK(WHITE_KING == 5);
  
  // Black pieces
  CHECK(BLACK_PAWN == 8);
  CHECK(BLACK_KNIGHT == 9);
  CHECK(BLACK_BISHOP == 10);
  CHECK(BLACK_ROOK == 11);
  CHECK(BLACK_QUEEN == 12);
  CHECK(BLACK_KING == 13);
  
  CHECK(NO_PIECE == 14);
}

TEST_CASE("Piece / TypeOf") {
  CHECK(type_of(WHITE_PAWN) == PAWN);
  CHECK(type_of(WHITE_KNIGHT) == KNIGHT);
  CHECK(type_of(WHITE_BISHOP) == BISHOP);
  CHECK(type_of(WHITE_ROOK) == ROOK);
  CHECK(type_of(WHITE_QUEEN) == QUEEN);
  CHECK(type_of(WHITE_KING) == KING);
  
  // Black pieces have same type
  CHECK(type_of(BLACK_PAWN) == PAWN);
  CHECK(type_of(BLACK_KNIGHT) == KNIGHT);
  CHECK(type_of(BLACK_KING) == KING);
}

TEST_CASE("Piece / ColourOf") {
  CHECK(colour_of(WHITE_PAWN) == WHITE);
  CHECK(colour_of(WHITE_KNIGHT) == WHITE);
  CHECK(colour_of(WHITE_KING) == WHITE);
  
  CHECK(colour_of(BLACK_PAWN) == BLACK);
  CHECK(colour_of(BLACK_KNIGHT) == BLACK);
  CHECK(colour_of(BLACK_KING) == BLACK);
}

TEST_CASE("Piece / MakePiece") {
  CHECK(make_piece(WHITE, PAWN) == WHITE_PAWN);
  CHECK(make_piece(WHITE, KNIGHT) == WHITE_KNIGHT);
  CHECK(make_piece(WHITE, KING) == WHITE_KING);
  
  CHECK(make_piece(BLACK, PAWN) == BLACK_PAWN);
  CHECK(make_piece(BLACK, KNIGHT) == BLACK_KNIGHT);
  CHECK(make_piece(BLACK, KING) == BLACK_KING);
}

TEST_CASE("Piece / RoundTrip") {
  // Ensure piece can be decomposed and reconstructed
  for (Colour c : {WHITE, BLACK}) {
    for (PieceType pt : {PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING}) {
      Piece p = make_piece(c, pt);
      CHECK(type_of(p) == pt);
      CHECK(colour_of(p) == c);
      CHECK(make_piece(colour_of(p), type_of(p)) == p);
    }
  }
}

// ============================================================================
// Direction Tests
// ============================================================================

TEST_CASE("Direction / Values") {
  CHECK(NORTH == 8);
  CHECK(SOUTH == -8);
  CHECK(EAST == 1);
  CHECK(WEST == -1);
  CHECK(NORTH_EAST == 9);
  CHECK(NORTH_WEST == 7);
  CHECK(SOUTH_EAST == -7);
  CHECK(SOUTH_WEST == -9);
}

// ============================================================================
// Bit Utility Cross-Compilation Tests
// ============================================================================

TEST_CASE("BitUtils / Popcount") {
  // Cross-platform popcount tests
  CHECK(popcount(0ULL) == 0);
  CHECK(popcount(1ULL) == 1);
  CHECK(popcount(255ULL) == 8);
  CHECK(popcount(0x0F0F0F0F0F0F0F0FUL) == 32);
  CHECK(popcount(0xFFFFFFFFFFFFFFFFUL) == 64);
}

TEST_CASE("BitUtils / LSB") {
  CHECK(lsb(1ULL) == 0);
  CHECK(lsb(2ULL) == 1);
  CHECK(lsb(4ULL) == 2);
  CHECK(lsb(256ULL) == 8);
  CHECK(lsb(0x8000000000000000ULL) == 63);
}

TEST_CASE("BitUtils / MSB") {
  CHECK(msb(1ULL) == 0);
  CHECK(msb(2ULL) == 1);
  CHECK(msb(256ULL) == 8);
  CHECK(msb(0x0100000000000000ULL) == 56);
  CHECK(msb(0x8000000000000000ULL) == 63);
}
