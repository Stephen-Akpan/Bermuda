#include "tt.h"
#include <cstring>
#include <algorithm>

// ============================================================================
// Global TT Instance
// ============================================================================

TranspositionTable TT;

// ============================================================================
// Construction & Resize
// ============================================================================

TranspositionTable::TranspositionTable(size_t mb) {
    resize(mb);
}

void TranspositionTable::resize(size_t mb) {
    // Compute number of entries that fit in the requested size
    size_t bytes   = mb * 1024 * 1024;
    size_t count   = bytes / sizeof(TTEntry);

    // Round down to the nearest power of two for fast indexing
    // Actually we use the __uint128_t multiply trick which works with any size,
    // so just use the exact count
    entries.assign(count, TTEntry{});
    current_age = 0;
}

void TranspositionTable::clear() {
    std::fill(entries.begin(), entries.end(), TTEntry{});
}

// ============================================================================
// Probe
// ============================================================================

const TTEntry* TranspositionTable::probe(ui64 hash) const {
    const TTEntry& entry = entries[index(hash)];

    // Verify the full key to rule out hash collisions
    if (entry.key == hash && entry.is_valid())
        return &entry;

    return nullptr;
}

// ============================================================================
// Store
// ============================================================================

void TranspositionTable::store(
    ui64  hash,
    Move  move,
    i16   score,
    ui8   depth,
    Bound bound)
{
    TTEntry& entry = entries[index(hash)];

    if (move.is_null() && entry.key == hash)
        move = entry.move;

    bool same_pos = (entry.key == hash);
    bool replace;

    if (!entry.is_valid()) {
        replace = true;
    }
    else if (entry.age != current_age) {
        replace = true;
    }
    else if (same_pos) {
        replace = depth >= entry.depth || entry.bound != BOUND_EXACT;
    }
    else {
        replace = entry.age != current_age || depth >= entry.depth;
    }

    if (!replace)
        return;

    entry.key = hash;
    entry.move = move;
    entry.score = score;
    entry.depth = depth;
    entry.bound = bound;
    entry.age = current_age;
}

// ============================================================================
// Utilities
// ============================================================================

i32 TranspositionTable::hashfull() const {
    // Sample the first 1000 entries to estimate fill rate
    i32 filled = 0;
    i32 sample = (i32)std::min(entries.size(), (size_t)1000);

    for (i32 i = 0; i < sample; i++)
        if (entries[i].is_valid() && entries[i].age == current_age)
            filled++;

    return (sample > 0) ? (filled * 1000 / sample) : 0;
}
