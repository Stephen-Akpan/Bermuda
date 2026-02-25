#pragma once

#include "types.h"
#include <string>

// ============================================================================
// Bitboard Utilities
// ============================================================================

namespace BB {

// ============================================================================
// Bitboard Constants & Masks
// ============================================================================

// Rank and file masks
constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
constexpr Bitboard RANK_3 = 0x0000000000FF0000ULL;
constexpr Bitboard RANK_4 = 0x00000000FF000000ULL;
constexpr Bitboard RANK_5 = 0x000000FF00000000ULL;
constexpr Bitboard RANK_6 = 0x0000FF0000000000ULL;
constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;

constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_B = 0x0202020202020202ULL;
constexpr Bitboard FILE_C = 0x0404040404040404ULL;
constexpr Bitboard FILE_D = 0x0808080808080808ULL;
constexpr Bitboard FILE_E = 0x1010101010101010ULL;
constexpr Bitboard FILE_F = 0x2020202020202020ULL;
constexpr Bitboard FILE_G = 0x4040404040404040ULL;
constexpr Bitboard FILE_H = 0x8080808080808080ULL;

constexpr Bitboard FILE_AB = FILE_A | FILE_B;
constexpr Bitboard FILE_GH = FILE_G | FILE_H;

// Diagonal masks
constexpr Bitboard LIGHT_SQUARES = 0x55AA55AA55AA55AAULL;
constexpr Bitboard DARK_SQUARES = 0xAA55AA55AA55AA55ULL;

// Edge masks (used for piece movement constraints)
constexpr Bitboard EDGE = FILE_A | FILE_H | RANK_1 | RANK_8;

// Convert square to single-bit bitboard
inline Bitboard square_bb(Square sq) {
  return 1ULL << sq;
}

// Get rank mask containing the square
inline Bitboard rank_bb(Square sq) {
  return RANK_1 << (rank(sq) * 8);
}

// Get file mask containing the square
inline Bitboard file_bb(Square sq) {
  return FILE_A << file(sq);
}

// ============================================================================
// Shift Operations (utility for move generation)
// ============================================================================

inline Bitboard shift(Direction d, Bitboard b) {
  switch (d) {
    case NORTH:      return (b << 8);
    case SOUTH:      return (b >> 8);
    case EAST:       return (b << 1) & ~FILE_A;
    case WEST:       return (b >> 1) & ~FILE_H;
    case NORTH_EAST: return (b << 9) & ~FILE_A;
    case NORTH_WEST: return (b << 7) & ~FILE_H;
    case SOUTH_EAST: return (b >> 7) & ~FILE_A;
    case SOUTH_WEST: return (b >> 9) & ~FILE_H;
    default:         return 0;
  }
}

// ============================================================================
// Pawn Attacks & Push
// ============================================================================

// Single pawn push (no blocking squares considered)
inline Bitboard pawn_push(Bitboard pawns, Colour c) {
  return c == WHITE ? (pawns << 8) : (pawns >> 8);
}

// Pawn attacks (captures) to the left and right
// Returns bitboard of squares where pawns can capture
inline Bitboard pawn_attacks(Bitboard pawns, Colour c) {
    return c == WHITE ? shift(NORTH_WEST, pawns) | shift(NORTH_EAST, pawns)
                    : shift(SOUTH_WEST, pawns) | shift(SOUTH_EAST, pawns);
}

// ============================================================================
// Knight Attacks
// ============================================================================

// Pre-computed knight attack masks (one per square)
extern Bitboard knight_attacks[64];

// Get knight attacks from a square
inline Bitboard knight_attack(Square sq) {
  return knight_attacks[sq];
}

// ============================================================================
// Bishop Attacks (Sliding Pieces)
// ============================================================================

// Pre-computed bishop attack masks (one per square, all blockers)
extern Bitboard bishop_masks[64];

// Magic number structure for bishop moves
struct Magic {
  Bitboard* attacks;  // Pointer to attack table for this square
  Bitboard mask;      // Blocker mask (relevant bits for sliding attacks)
  Bitboard magic;     // Magic number for hashing
  ui8 shift;          // Bit shift for indexing (64 - popcount(mask))
};

extern Magic bishop_magics[64];
extern Magic rook_magics[64];

// Attack tables (large pre-computed arrays)
extern Bitboard bishop_attacks_table[64][512];
extern Bitboard rook_attacks_table[64][4096];

// Get bishop attacks from a square given occupied squares
inline Bitboard bishop_attack(Square sq, Bitboard occupied) {
  const Magic& m = bishop_magics[sq];
  return m.attacks[((occupied & m.mask) * m.magic) >> m.shift];
}

// Get rook attacks from a square given occupied squares
inline Bitboard rook_attack(Square sq, Bitboard occupied) {
  const Magic& m = rook_magics[sq];
  return m.attacks[((occupied & m.mask) * m.magic) >> m.shift];
}

// Get queen attacks (bishop + rook)
inline Bitboard queen_attack(Square sq, Bitboard occupied) {
  return bishop_attack(sq, occupied) | rook_attack(sq, occupied);
}

// ============================================================================
// King Attacks
// ============================================================================

// Pre-computed king attack masks (one per square)
extern Bitboard king_attacks[64];

// Get king attacks from a square
inline Bitboard king_attack(Square sq) {
  return king_attacks[sq];
}

// ============================================================================
// Distance & Connectivity
// ============================================================================

// Pre-computed distances between squares
extern ui8 distance[64][64];

// Get squared Euclidean distance: returns 0-128 (max 8*8 + 8*8)
inline ui8 square_distance(Square s1, Square s2) {
  return distance[s1][s2];
}

// Get squares between two squares (excluding the endpoints, for blocking detection)
extern Bitboard between[64][64];

// Get bitboard of squares between s1 and s2
inline Bitboard squares_between(Square s1, Square s2) {
  return between[s1][s2];
}

// ============================================================================
// Debugging & Utilities
// ============================================================================

// Print bitboard to stdout in human-readable format
void print(Bitboard b);

// Convert bitboard to string representation
std::string to_string(Bitboard b);

// Initialize all bitboard lookup tables (call once at startup)
void init();

}  // namespace BB
