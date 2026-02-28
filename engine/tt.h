#pragma once

#include "types.h"
#include "move.h"
#include <cstddef>
#include <vector>

// ============================================================================
// Transposition Table
// ============================================================================
// The TT stores previously searched positions so we don't have to re-search
// them. Each entry maps a Zobrist hash to a search result.
//
// Entry layout uses a single 64-bit value plus the hash for verification,
// keeping each entry small to maximise cache efficiency.
//
// Bound types:
//   EXACT  — the score is exact (we searched all moves and found the best)
//   LOWER  — score is a lower bound (we failed high, this is a beta cutoff)
//   UPPER  — score is an upper bound (we failed low, no move improved alpha)

// ============================================================================
// Bound Type
// ============================================================================

enum Bound : ui8 {
    BOUND_NONE  = 0,
    BOUND_UPPER = 1,   // Alpha node  — score <= alpha
    BOUND_LOWER = 2,   // Beta node   — score >= beta (cut node)
    BOUND_EXACT = 3,   // PV node     — exact score
};

// ============================================================================
// TT Entry
// ============================================================================
// 16 bytes per entry. Aligned for cache efficiency.

struct alignas(16) TTEntry {
    ui64  key;        // Full Zobrist hash for verification
    Move  move;       // Best move found (NULL_MOVE if none)
    i16   score;      // Score at this node
    ui8   depth;      // Depth searched
    Bound bound;      // Type of bound on the score
    ui8   age;        // Search generation (for replacement policy)

    bool is_valid()  const { return bound != BOUND_NONE; }
    bool is_exact()  const { return bound == BOUND_EXACT; }
    bool is_lower()  const { return bound == BOUND_LOWER; }
    bool is_upper()  const { return bound == BOUND_UPPER; }
};

// ============================================================================
// Transposition Table
// ============================================================================

class TranspositionTable {
public:
    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    // Construct with a size in megabytes (default 64MB)
    explicit TranspositionTable(size_t mb = 64);

    // -------------------------------------------------------------------------
    // Resize
    // -------------------------------------------------------------------------

    // Resize the table to the given number of megabytes.
    // Clears all existing entries.
    void resize(size_t mb);

    // Clear all entries
    void clear();

    // -------------------------------------------------------------------------
    // Age management
    // -------------------------------------------------------------------------

    // Call at the start of each new search to advance the generation counter.
    // This makes old entries from previous searches preferentially replaced.
    void new_search() { current_age++; }

    ui8 age() const { return current_age; }

    // -------------------------------------------------------------------------
    // Probe
    // -------------------------------------------------------------------------

    // Look up a position by its Zobrist hash.
    // Returns a pointer to the entry if found and the key matches,
    // or nullptr if the slot is empty or belongs to a different position.
    const TTEntry* probe(ui64 hash) const;

    // -------------------------------------------------------------------------
    // Store
    // -------------------------------------------------------------------------

    // Store a search result for a position.
    // Replacement policy: prefer replacing entries from older searches,
    // or entries searched to a lower depth.
    void store(ui64   hash,
               Move   move,
               i16    score,
               ui8    depth,
               Bound  bound);

    // -------------------------------------------------------------------------
    // Utilities
    // -------------------------------------------------------------------------

    // Approximate hash fill rate as a per-mille value (0-1000)
    i32 hashfull() const;

    // Total number of entries the table can hold
    size_t capacity() const { return entries.size(); }

private:
    std::vector<TTEntry> entries;
    ui8                  current_age = 0;

    // Map a hash to a table index
    size_t index(ui64 hash) const {
        ui64 hi = hash >> 32;
        ui64 lo = hash & 0XFFFFFFFFULL;
        ui64 n = (ui64)entries.size();
        return (size_t)((hi * n +((lo * n) >> 32)) >> 32);
    }
};

// ============================================================================
// Global TT instance
// ============================================================================
// Declared here, defined in tt.cpp.
// Search and other components access it via this global.

extern TranspositionTable TT;
