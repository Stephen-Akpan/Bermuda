#include "zobrist.h"
#include <random>

ui64 Zobrist::piece[12][64];
ui64 Zobrist::castle[16];
ui64 Zobrist::ep[64];
ui64 Zobrist::side;

void Zobrist::init() {
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL);
    for (i32 p = 0; p < 12; ++p) {
        for (i32 s = 0; s < 64; ++s) {
            piece[p][s] = rng();
        }
    }

    for (i32 i = 0; i < 16; ++i) {
        castle[i] = rng();
    }
}
