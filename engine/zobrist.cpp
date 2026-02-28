#include "zobrist.h"
#include "types.h"
#include "bitboard.h"
#include <cstdint>
#include <iostream>

namespace Zobrist {

// ============================================================================
// Key Storage
// ============================================================================

ui64 piece_keys[14][64];
ui64 side_key;
ui64 castling_keys[16];
ui64 ep_keys[8];

// ============================================================================
// PRNG (xorshift64)
// ============================================================================
// Simple, fast, high-quality pseudo-random number generator.
// Used only during init() to fill the key tables.

static ui64 prng_state = 0x246C6B3F6CF0A75DULL;  // arbitrary non-zero seed

static ui64 prng() {
    prng_state ^= prng_state << 13;
    prng_state ^= prng_state >> 7;
    prng_state ^= prng_state << 17;
    return prng_state;
}

// ============================================================================
// Initialisation
// ============================================================================

void init() {
    // Piece keys — for every (piece, square) combination
    for (i32 p = 0; p < 14; p++)
        for (i32 sq = 0; sq < 64; sq++)
            piece_keys[p][sq] = prng();

    // Side to move key — XORed in when black is to move
    side_key = prng();

    // Castling keys — one per combination of rights (16 possible values)
    for (i32 i = 0; i < 16; i++)
        castling_keys[i] = prng();

    // En passant keys — one per file
    for (i32 f = 0; f < 8; f++)
        ep_keys[f] = prng();
}

// ============================================================================
// Full Hash Computation
// ============================================================================

ui64 compute(
    const ui64  piece_bb[2][6],
    ui64        side_to_move,
    ui8         castling_rights,
    Square      ep_sq
) {
    ui64 hash = 0;

    // XOR in all pieces
    for (i32 c = 0; c < 2; c++) {
        for (i32 pt = 0; pt < 6; pt++) {
            ui64 bb = piece_bb[c][pt];
            while (bb) {
                i32 sq = lsb(bb);
                bb &= bb - 1;

                // Reconstruct the Piece enum value: white = pt, black = pt + 8
                i32 piece = (c == 1) ? pt + 8 : pt;
                hash ^= piece_keys[piece][sq];
            }
        }
    }

    // XOR in side to move
    if (side_to_move == 1)  // BLACK
        hash ^= side_key;

    // XOR in castling rights
    hash ^= castling_keys[castling_rights & 0xF];

    // XOR in en passant file
    if (ep_sq != NO_SQUARE) {
        Colour stm = (Colour)side_to_move;
        Bitboard ep_bb = BB::square_bb(ep_sq);
        Bitboard attackers = BB::pawn_attacks(ep_bb, flip(stm)) & piece_bb[stm][PAWN];

        if (attackers) {
            hash ^= ep_keys[file(ep_sq)];
        }
    }

    return hash;
}

} // namespace Zobrist
