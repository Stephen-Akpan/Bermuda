#pragma once

#include <cstdint>
#include <assert.h>

// Numeric types

// Signed integer types

// 8 bit integer
typedef int8_t i8;

// 16 bit integer
typedef int16_t i16;

// 32 bit integer
typedef int32_t i32;

// 64 bit integer
typedef int64_t i64;

// Unsigned integer types

// 8 bit unsigned integer
typedef uint8_t ui8;

// 16 bit unsigned integer
typedef uint16_t ui16;

// 32 bit unsigned integer
typedef uint32_t ui32;

// 64 bit unsigned integer
typedef uint64_t ui64;

// Precision types

// 32 bit precision type
typedef float f32;

// 64 bit precision type
typedef double f64;

using Score = i32;
using Bitboard = ui64;
// ============================================================================
// Squares
// ============================================================================
// Little-endian rank-file mapping: a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63

enum Square : ui8 {
  a1, b1, c1, d1, e1, f1, g1, h1,
  a2, b2, c2, d2, e2, f2, g2, h2,
  a3, b3, c3, d3, e3, f3, g3, h3,
  a4, b4, c4, d4, e4, f4, g4, h4,
  a5, b5, c5, d5, e5, f5, g5, h5,
  a6, b6, c6, d6, e6, f6, g6, h6,
  a7, b7, c7, d7, e7, f7, g7, h7,
  a8, b8, c8, d8, e8, f8, g8, h8,
  NO_SQUARE = 64
};

// Square utilities
inline ui8 file(Square sq) { return sq & 7; }
inline ui8 rank(Square sq) { return sq >> 3; }
inline Square make_square(ui8 f, ui8 r) { return (Square)(r * 8 + f); }
inline bool is_valid_square(Square sq) { return sq >= 0 && sq < 64; }

// ============================================================================
// Colours
// ============================================================================

enum Colour : ui8 {
  WHITE = 0,
  BLACK = 1,
  NO_COLOUR = 2
};

// Colour utilities
inline Colour flip(Colour c) { return (Colour)(c ^ 1); }
inline i32 pawn_push_dir(Colour c) { return c == WHITE ? 8 : -8; }

// ============================================================================
// Piece Types & Pieces
// ============================================================================

enum PieceType : ui8 {
  PAWN = 0,
  KNIGHT = 1,
  BISHOP = 2,
  ROOK = 3,
  QUEEN = 4,
  KING = 5,
  NO_PIECE_TYPE = 6
};

enum Piece : ui8 {
  WHITE_PAWN = 0,
  WHITE_KNIGHT = 1,
  WHITE_BISHOP = 2,
  WHITE_ROOK = 3,
  WHITE_QUEEN = 4,
  WHITE_KING = 5,
  BLACK_PAWN = 8,
  BLACK_KNIGHT = 9,
  BLACK_BISHOP = 10,
  BLACK_ROOK = 11,
  BLACK_QUEEN = 12,
  BLACK_KING = 13,
  NO_PIECE = 14
};

// Piece utilities
inline PieceType type_of(Piece p) { return (PieceType)(p & 7); }
inline Colour colour_of(Piece p) { return (Colour)(p >> 3); }
inline Piece make_piece(Colour c, PieceType pt) { return (Piece)((c << 3) | pt); }

// Piece values (for eval)
enum Value {
  PAWN_MG   = 100,   PAWN_EG   = 200,
  KNIGHT_MG = 700,   KNIGHT_EG = 800,
  BISHOP_MG = 800,   BISHOP_EG = 900,
  ROOK_MG   = 1000,  ROOK_EG   = 1100,
  QUEEN_MG  = 2000,  QUEEN_EG  = 2300,
  KING_MG   = 0,     KING_EG   = 0
};

constexpr f64 PIECE_VALUES[12] = {
  // Middle game
  PAWN_MG, KNIGHT_MG, BISHOP_MG, ROOK_MG, QUEEN_MG, KING_MG,
  // End game
  PAWN_EG, KNIGHT_EG, BISHOP_EG, ROOK_EG, QUEEN_EG, KING_EG
};

// ============================================================================
// Directions
// ============================================================================
// Directions for moving pieces, matching square indexing (little-endian rank-file)

enum Direction : i8 {
  NORTH = 8,
  SOUTH = -8,
  EAST = 1,
  WEST = -1,
  
  // Diagonal directions
  NORTH_EAST = 9,
  NORTH_WEST = 7,
  SOUTH_EAST = -7,
  SOUTH_WEST = -9
};

// ============================================================================
// Move Encoding (16-bit)
// ============================================================================
//
// Bit layout:
//   bits  0-5  : from square  (0-63)
//   bits  6-11 : to square    (0-63)
//   bits 12-13 : promotion piece type (see PromoPiece below), 0 if not a promo
//   bits 14-15 : move flag    (see MoveFlag below)
//
// A value of 0 is used as a null/empty move sentinel.
// ============================================================================

// ============================================================================
// Move Flags
// ============================================================================

enum MoveFlag : ui8 {
    NORMAL      = 0,
    PROMOTION   = 1,
    EN_PASSANT  = 2,
    CASTLING    = 3
};

// ============================================================================
// Promotion Piece
// ============================================================================
// Encoded in bits 12-13 when flag == PROMOTION.
// Only knight, bishop, rook, queen are valid promotion targets.

enum PromoPiece : ui8 {
    PROMO_KNIGHT = 0,
    PROMO_BISHOP = 1,
    PROMO_ROOK   = 2,
    PROMO_QUEEN  = 3
};

// ============================================================================
// Castling Rights
// ============================================================================
enum CastlingRight : ui8 {
  NO_CASTLING = 0,
  WHITE_OO = 1,
  WHITE_OOO = 2,
  BLACK_OO = 4,
  BLACK_OOO = 8,
  ALL_CASTLING = 15
};

// ============================================================================
// Bit Utilities
// ============================================================================

// Count set bits (population count)
inline i32 popcount(Bitboard b) {
#if defined(_MSC_VER)
  return __popcnt64(b);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(b);
#else
  // Fallback: count bits manually
  i32 count = 0;
  while (b) {
    count += b & 1;
    b >>= 1;
  }
  return count;
#endif
}

// Get least significant bit (LSB) position (0-63), undefined if b == 0
inline ui8 lsb(Bitboard b) {
#if defined(_MSC_VER)
  unsigned long idx;
  _BitScanForward64(&idx, b);
  return idx;
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_ctzll(b);  // Count trailing zeros
#else
  ui8 idx = 0;
  if ((b & 0xFFFFFFFFULL) == 0) { b >>= 32; idx = 32; }
  if ((b & 0xFFFFULL) == 0) { b >>= 16; idx += 16; }
  if ((b & 0xFFULL) == 0) { b >>= 8; idx += 8; }
  if ((b & 0xFULL) == 0) { b >>= 4; idx += 4; }
  if ((b & 0x3ULL) == 0) { b >>= 2; idx += 2; }
  if ((b & 0x1ULL) == 0) { idx += 1; }
  return idx;
#endif
}

// Get most significant bit (MSB) position (0-63), undefined if b == 0
inline ui8 msb(Bitboard b) {
#if defined(_MSC_VER)
  unsigned long idx;
  _BitScanReverse64(&idx, b);
  return idx;
#elif defined(__GNUC__) || defined(__clang__)
  return 63 - __builtin_clzll(b);  // 63 - count leading zeros
#else
  ui8 idx = 63;
  if ((b & 0xFFFFFFFF00000000ULL) == 0) { b <<= 32; idx -= 32; }
  if ((b & 0xFFFF000000000000ULL) == 0) { b <<= 16; idx -= 16; }
  if ((b & 0xFF00000000000000ULL) == 0) { b <<= 8; idx -= 8; }
  if ((b & 0xF000000000000000ULL) == 0) { b <<= 4; idx -= 4; }
  if ((b & 0xC000000000000000ULL) == 0) { b <<= 2; idx -= 2; }
  if ((b & 0x8000000000000000ULL) == 0) { idx -= 1; }
  return idx;
#endif
}

// Get and clear least significant bit
inline ui8 pop_lsb(Bitboard& b) {
  ui8 idx = lsb(b);
  b &= b - 1;  // Remove LSB
  return idx;
}

// Check if square is on bitboard
inline bool test_bit(Bitboard b, Square sq) {
  return (b >> sq) & 1;
}

// Set bit at square
inline void set_bit(Bitboard& b, Square sq) {
  b |= (1ULL << sq);
}

// Clear bit at square
inline void clear_bit(Bitboard& b, Square sq) {
  b &= ~(1ULL << sq);
}

// Toggle bit at square
inline void toggle_bit(Bitboard& b, Square sq) {
  b ^= (1ULL << sq);
}