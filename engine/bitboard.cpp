#include "bitboard.h"
#include "magics.h"
#include <iostream>
#include <sstream>
#include <cstring>

namespace BB {

// ============================================================================
// Lookup Table Storage
// ============================================================================

Bitboard knight_attacks[64];
Bitboard king_attacks[64];
Bitboard bishop_masks[64];

Magic bishop_magics[64];
Magic rook_magics[64];

Bitboard bishop_attacks_table[64][512];
Bitboard rook_attacks_table[64][4096];

ui8 distance[64][64];
Bitboard between[64][64];

// ============================================================================
// Attack Pattern Generation
// ============================================================================

static Bitboard knight_moves(Square sq) {
  Bitboard b = 0;
  i32 f = file(sq);
  i32 r = rank(sq);

  constexpr i32 knight_offsets[8] = {-17, -15, -10, -6, 6, 10, 15, 17};

  for (i32 offset : knight_offsets) {
    i32 target = sq + offset;
    if (target < 0 || target > 63) continue;

    i32 tf = file(Square(target));
    i32 tr = rank(Square(target));

    if ((abs(tf - f) == 1 && abs(tr - r) == 2) ||
        (abs(tf - f) == 2 && abs(tr - r) == 1)) {
      b |= (1ULL << target);
    }
  }

  return b;
}

static Bitboard king_moves(Square sq) {
  Bitboard b    = 0;
  Bitboard sq_bb = 1ULL << sq;

  if (sq_bb & ~FILE_A) b |= sq_bb >> 1;
  if (sq_bb & ~FILE_H) b |= sq_bb << 1;
  if (sq_bb & ~RANK_1) b |= sq_bb >> 8;
  if (sq_bb & ~RANK_8) b |= sq_bb << 8;
  if ((sq_bb & ~FILE_A) & ~RANK_1) b |= sq_bb >> 9;
  if ((sq_bb & ~FILE_H) & ~RANK_1) b |= sq_bb >> 7;
  if ((sq_bb & ~FILE_A) & ~RANK_8) b |= sq_bb << 7;
  if ((sq_bb & ~FILE_H) & ~RANK_8) b |= sq_bb << 9;

  return b;
}

static Bitboard bishop_blocker_mask(Square sq) {
  Bitboard b = 0;
  i32 f = file(sq);
  i32 r = rank(sq);

  for (i32 nf = f+1, nr = r+1; nf <= 6 && nr <= 6; nf++, nr++) b |= 1ULL << make_square(nf, nr);
  for (i32 nf = f-1, nr = r+1; nf >= 1 && nr <= 6; nf--, nr++) b |= 1ULL << make_square(nf, nr);
  for (i32 nf = f+1, nr = r-1; nf <= 6 && nr >= 1; nf++, nr--) b |= 1ULL << make_square(nf, nr);
  for (i32 nf = f-1, nr = r-1; nf >= 1 && nr >= 1; nf--, nr--) b |= 1ULL << make_square(nf, nr);

  return b;
}

static Bitboard rook_blocker_mask(Square sq) {
  Bitboard b = 0;
  i32 f = file(sq);
  i32 r = rank(sq);

  for (i32 i = r + 1; i < 7; i++) b |= (1ULL << make_square(f, i));
  for (i32 i = r - 1; i > 0; i--) b |= (1ULL << make_square(f, i));
  for (i32 i = f + 1; i < 7; i++) b |= (1ULL << make_square(i, r));
  for (i32 i = f - 1; i > 0; i--) b |= (1ULL << make_square(i, r));

  return b;
}

static Bitboard generate_bishop_attacks(Square sq, Bitboard blockers) {
  Bitboard b = 0;
  i32 f = file(sq);
  i32 r = rank(sq);

  for (i32 i = 1; i < 8; i++) { i32 nf=f+i, nr=r+i; if(nf>=8||nr>=8) break; Square t=make_square(nf,nr); b|=(1ULL<<t); if(blockers&(1ULL<<t)) break; }
  for (i32 i = 1; i < 8; i++) { i32 nf=f-i, nr=r+i; if(nf< 0||nr>=8) break; Square t=make_square(nf,nr); b|=(1ULL<<t); if(blockers&(1ULL<<t)) break; }
  for (i32 i = 1; i < 8; i++) { i32 nf=f+i, nr=r-i; if(nf>=8||nr< 0) break; Square t=make_square(nf,nr); b|=(1ULL<<t); if(blockers&(1ULL<<t)) break; }
  for (i32 i = 1; i < 8; i++) { i32 nf=f-i, nr=r-i; if(nf< 0||nr< 0) break; Square t=make_square(nf,nr); b|=(1ULL<<t); if(blockers&(1ULL<<t)) break; }

  return b;
}

static Bitboard generate_rook_attacks(Square sq, Bitboard blockers) {
  Bitboard b = 0;
  i32 f = file(sq);
  i32 r = rank(sq);

  for (i32 i=r+1; i<8; i++) { Square t=make_square(f,i); b|=(1ULL<<t); if(blockers&(1ULL<<t)) break; }
  for (i32 i=r-1; i>=0;i--) { Square t=make_square(f,i); b|=(1ULL<<t); if(blockers&(1ULL<<t)) break; }
  for (i32 i=f+1; i<8; i++) { Square t=make_square(i,r); b|=(1ULL<<t); if(blockers&(1ULL<<t)) break; }
  for (i32 i=f-1; i>=0;i--) { Square t=make_square(i,r); b|=(1ULL<<t); if(blockers&(1ULL<<t)) break; }

  return b;
}

// ============================================================================
// Initialization
// ============================================================================

void init() {
  // ── Knight & king (non-sliding, no magic needed) ──────────────────────────
  for (i32 sq = 0; sq < 64; sq++) {
    knight_attacks[sq] = knight_moves(Square(sq));
    king_attacks[sq]   = king_moves(Square(sq));
  }

  // ── Bishop magic attack tables ─────────────────────────────────────────────
  for (i32 sq = 0; sq < 64; sq++) {
    Bitboard mask  = bishop_blocker_mask(Square(sq));
    Bitboard magic = bishop_magics_data[sq];   // pre-computed value from magics.h
    ui8      shift = 64 - popcount(mask);      // index width = popcount(mask) bits

    // Store mask in the separate legacy array (used elsewhere in the codebase)
    bishop_masks[sq] = mask;

    // Wire up the Magic struct — attacks pointer aims at this square's slice
    bishop_magics[sq].mask    = mask;
    bishop_magics[sq].magic   = magic;
    bishop_magics[sq].shift   = shift;
    bishop_magics[sq].attacks = bishop_attacks_table[sq];

    // Carry-rippler: enumerate every subset of `mask`
    // `blockers` visits all 2^n occupancy configurations, starting and
    // ending at 0. The do-while ensures the zero (empty board) case is
    // also written, which is necessary for positions with no blockers.
    Bitboard blockers = 0ULL;
    do {
      // KEY STEP: mask must be applied before multiplying.
      // `blockers` IS already a subset of mask here (the rippler guarantees
      // it), but being explicit makes the intent match the runtime lookup:
      //   index = (occupied & mask) * magic >> shift
      ui64 index = ((blockers & mask) * magic) >> shift;

      // Write the attack set for this exact blocker configuration.
      // Blockers on the boundary squares ARE included in the attack set
      // (they represent capturable pieces), so we pass `blockers` raw.
      bishop_attacks_table[sq][index] = generate_bishop_attacks(Square(sq), blockers);

      // Advance to next subset (carry-rippler trick)
      blockers = (blockers - mask) & mask;
    } while (blockers != 0ULL);
  }

  // ── Rook magic attack tables ───────────────────────────────────────────────
  for (i32 sq = 0; sq < 64; sq++) {
    Bitboard mask  = rook_blocker_mask(Square(sq));
    Bitboard magic = rook_magics_data[sq];     // pre-computed value from magics.h
    ui8      shift = 64 - popcount(mask);

    rook_magics[sq].mask    = mask;
    rook_magics[sq].magic   = magic;
    rook_magics[sq].shift   = shift;
    rook_magics[sq].attacks = rook_attacks_table[sq];

    Bitboard blockers = 0ULL;
    do {
      ui64 index = ((blockers & mask) * magic) >> shift;
      rook_attacks_table[sq][index] = generate_rook_attacks(Square(sq), blockers);
      blockers = (blockers - mask) & mask;
    } while (blockers != 0ULL);
  }

  // ── Chebyshev distance ─────────────────────────────────────────────────────
  for (i32 i = 0; i < 64; i++) {
    for (i32 j = 0; j < 64; j++) {
      i32 df = abs(file(Square(i)) - file(Square(j)));
      i32 dr = abs(rank(Square(i)) - rank(Square(j)));
      distance[i][j] = (df > dr) ? df : dr;
    }
  }

  // ── Between table ──────────────────────────────────────────────────────────
  for (i32 s1 = 0; s1 < 64; s1++) {
    for (i32 s2 = 0; s2 < 64; s2++) {
      between[s1][s2] = 0;
      if (s1 == s2) continue;

      i32 df = file(Square(s2)) - file(Square(s1));
      i32 dr = rank(Square(s2)) - rank(Square(s1));

      bool same_rank = (dr == 0);
      bool same_file = (df == 0);
      bool same_diag = (abs(df) == abs(dr));

      if (!same_rank && !same_file && !same_diag) continue;

      if (df != 0) df /= abs(df);
      if (dr != 0) dr /= abs(dr);

      i32 f = file(Square(s1)) + df;
      i32 r = rank(Square(s1)) + dr;

      while (f != file(Square(s2)) || r != rank(Square(s2))) {
        between[s1][s2] |= (1ULL << make_square(f, r));
        f += df;
        r += dr;
      }
    }
  }
}

// ============================================================================
// Debugging & Utilities
// ============================================================================

void print(Bitboard b) { std::cout << to_string(b); }

std::string to_string(Bitboard b) {
  std::ostringstream oss;
  for (i32 r = 7; r >= 0; r--) {
    oss << (r + 1) << "  ";
    for (i32 f = 0; f < 8; f++) {
      Square sq = make_square(f, r);
      oss << ((b & (1ULL << sq)) ? "1 " : ". ");
    }
    oss << "\n";
  }
  oss << "   a b c d e f g h\n";
  return oss.str();
}

}  // namespace BB
