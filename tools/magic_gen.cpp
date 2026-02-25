/**
 * Magic Bitboard Generator
 * Emits magics.h for use with BB::Magic (pointer-based) in bitboard.cpp.
 *
 * Compile:  g++ -O3 -o generate_magics generate_magics.cpp
 * Run:      ./generate_magics > magics.h
 */

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <iomanip>

using Bitboard = uint64_t;
using ui64     = uint64_t;
using ui8      = uint8_t;

inline int popcount(Bitboard b) {
#if defined(_MSC_VER)
  return __popcnt64(b);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(b);
#else
  // Fallback: count bits manually
  int count = 0;
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

static inline int file_of(int sq)      { return sq & 7; }
static inline int rank_of(int sq)      { return sq >> 3; }
static inline int make_sq(int f, int r){ return r * 8 + f; }

// ─── Blocker masks ────────────────────────────────────────────────────────────
// Must exactly match bishop_blocker_mask / rook_blocker_mask in bitboard.cpp.
// Edges excluded: a blocker on the board edge can never stop a ray earlier.

static Bitboard bishop_mask(int sq) {
    Bitboard b = 0;
    int f = file_of(sq), r = rank_of(sq);
    for (int nf=f+1,nr=r+1; nf<=6&&nr<=6; nf++,nr++) b|=1ULL<<make_sq(nf,nr);
    for (int nf=f-1,nr=r+1; nf>=1&&nr<=6; nf--,nr++) b|=1ULL<<make_sq(nf,nr);
    for (int nf=f+1,nr=r-1; nf<=6&&nr>=1; nf++,nr--) b|=1ULL<<make_sq(nf,nr);
    for (int nf=f-1,nr=r-1; nf>=1&&nr>=1; nf--,nr--) b|=1ULL<<make_sq(nf,nr);
    return b;
}

static Bitboard rook_mask(int sq) {
    Bitboard b = 0;
    int f = file_of(sq), r = rank_of(sq);
    for (int i=r+1; i<7; i++) b|=1ULL<<make_sq(f,i);
    for (int i=r-1; i>0; i--) b|=1ULL<<make_sq(f,i);
    for (int i=f+1; i<7; i++) b|=1ULL<<make_sq(i,r);
    for (int i=f-1; i>0; i--) b|=1ULL<<make_sq(i,r);
    return b;
}

// ─── Reference attack generators ─────────────────────────────────────────────

static Bitboard bishop_attacks(int sq, Bitboard occ) {
    Bitboard b = 0; int f=file_of(sq),r=rank_of(sq);
    for(int i=1;i<8;i++){int nf=f+i,nr=r+i;if(nf>7||nr>7)break;b|=1ULL<<make_sq(nf,nr);if(occ>>make_sq(nf,nr)&1)break;}
    for(int i=1;i<8;i++){int nf=f-i,nr=r+i;if(nf<0||nr>7)break;b|=1ULL<<make_sq(nf,nr);if(occ>>make_sq(nf,nr)&1)break;}
    for(int i=1;i<8;i++){int nf=f+i,nr=r-i;if(nf>7||nr<0)break;b|=1ULL<<make_sq(nf,nr);if(occ>>make_sq(nf,nr)&1)break;}
    for(int i=1;i<8;i++){int nf=f-i,nr=r-i;if(nf<0||nr<0)break;b|=1ULL<<make_sq(nf,nr);if(occ>>make_sq(nf,nr)&1)break;}
    return b;
}

static Bitboard rook_attacks(int sq, Bitboard occ) {
    Bitboard b = 0; int f=file_of(sq),r=rank_of(sq);
    for(int i=r+1;i<8;i++){b|=1ULL<<make_sq(f,i);if(occ>>make_sq(f,i)&1)break;}
    for(int i=r-1;i>=0;i--){b|=1ULL<<make_sq(f,i);if(occ>>make_sq(f,i)&1)break;}
    for(int i=f+1;i<8;i++){b|=1ULL<<make_sq(i,r);if(occ>>make_sq(i,r)&1)break;}
    for(int i=f-1;i>=0;i--){b|=1ULL<<make_sq(i,r);if(occ>>make_sq(i,r)&1)break;}
    return b;
}

// ─── PRNG (xorshift64, sparse candidates) ────────────────────────────────────

static ui64 seed = 0x123456789ABCDEF0ULL;
static ui64 rand64() {
    seed ^= seed>>12; seed ^= seed<<25; seed ^= seed>>27;
    return seed * 0x2545F4914F6CDD1DULL;
}
static ui64 sparse_rand() { return rand64() & rand64() & rand64(); }

// ─── Magic search ─────────────────────────────────────────────────────────────

static ui64 find_magic(int sq, Bitboard mask, bool is_rook) {
    static Bitboard occs[4096], atks[4096], used[4096];

    int n = popcount(mask), size = 1 << n;

    Bitboard sub = 0;
    for (int i = 0; i < size; i++) {
        occs[i] = sub;
        atks[i] = is_rook ? rook_attacks(sq, sub) : bishop_attacks(sq, sub);
        sub = (sub - mask) & mask;
    }

    for (int attempt = 0; attempt < 100'000'000; attempt++) {
        ui64 magic = sparse_rand();
        if (popcount((mask * magic) >> 56) < 6) continue;

        memset(used, 0, sizeof(Bitboard) * size);
        bool ok = true;
        for (int i = 0; i < size && ok; i++) {
            ui64 idx = (occs[i] * magic) >> (64 - n);
            if      (!used[idx])           used[idx] = atks[i];
            else if (used[idx] != atks[i]) ok = false;
        }
        if (ok) return magic;
    }
    std::cerr << "Failed for sq " << sq << "\n";
    return 0ULL;
}

// ─── Emit ─────────────────────────────────────────────────────────────────────

static void emit_array(const char* name, const ui64 vals[64]) {
    std::cout << "constexpr ui64 " << name << "[64] = {\n";
    for (int sq = 0; sq < 64; sq++) {
        std::cout << "    0x" << std::hex << std::setw(16) << std::setfill('0')
                  << vals[sq] << "ULL" << (sq<63 ? "," : "") << "\n";
    }
    std::cout << "};\n\n";
}

int main() {
    seed = (ui64)time(nullptr) ^ 0xDEADBEEFCAFEBABEULL;

    ui64 bm[64], rm[64];

    std::cerr << "Generating bishop magics...\n";
    for (int sq = 0; sq < 64; sq++) bm[sq] = find_magic(sq, bishop_mask(sq), false);

    std::cerr << "Generating rook magics...\n";
    for (int sq = 0; sq < 64; sq++) rm[sq] = find_magic(sq, rook_mask(sq), true);

    std::cerr << "Done.\n";

    std::cout << "#pragma once\n";
    std::cout << "// Auto-generated by generate_magics.cpp — do not edit.\n\n";
    std::cout << "#include \"types.h\"\n\n";

    // bitboard.cpp's init() reads these two arrays and derives shift
    // locally as (64 - popcount(mask)), so we only need the magic numbers.
    emit_array("bishop_magics_data", bm);
    emit_array("rook_magics_data",   rm);

    return 0;
}
