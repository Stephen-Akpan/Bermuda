#pragma once

#include "types.h"

namespace Zobrist {
    extern ui64 piece[12][64];
    extern ui64 castle[16];
    extern ui64 ep[64];
    extern ui64 side;
    void init();
}
