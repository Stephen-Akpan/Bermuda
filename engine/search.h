#pragma once

#include "position.h"
#include "move.h"
#include "tt.h"
#include "eval.h"
#include "repetition.h"
#include <atomic>
#include <chrono>

// ============================================================================
// Search Constants
// ============================================================================

constexpr i32 INF       = 32000;
constexpr i32 MATE      = 30000;
constexpr i32 MAX_PLY   = 128;

// Score thresholds for mate detection
constexpr i32 MATE_BOUND = MATE - MAX_PLY;

// Helpers for encoding mate distance into score
inline i32 mate_in(i32 ply)  { return  MATE - ply; }
inline i32 mated_in(i32 ply) { return -MATE + ply; }

// Returns true if the score represents a forced mate
inline bool is_mate_score(i32 score) {
    return score > MATE_BOUND || score < -MATE_BOUND;
}

// ============================================================================
// Per-Ply Search Stack
// ============================================================================
// One entry per ply in the search tree. Extended as search goes deeper.

struct Stack {
    Move current_move;      // Move played to reach this node
    Move killers[2];        // Killer moves at this ply
    i32  ply;               // Distance from root
    i32  static_eval;       // Cached static evaluation (set on first visit)
    bool null_move_allowed;
};

// ============================================================================
// Search Limits
// ============================================================================

struct SearchLimits {
    i32  depth      = MAX_PLY;  // Maximum search depth
    i32  movetime   = 0;        // Fixed time per move in ms (0 = none)
    i32  wtime      = 0;        // White time remaining in ms
    i32  btime      = 0;        // Black time remaining in ms
    i32  winc       = 0;        // White increment in ms
    i32  binc       = 0;        // Black increment in ms
    i32  movestogo  = 0;        // Moves until next time control (0 = sudden death)
    bool infinite   = false;    // Search indefinitely (ignore time)
    bool ponder     = false;    // Pondering mode
};

// ============================================================================
// Search Info
// ============================================================================
// Owns all state for a single search. One instance per thread.

struct SearchInfo {
    // -------------------------------------------------------------------------
    // Position and game history
    // -------------------------------------------------------------------------
    Position        pos;
    RepetitionTable rep_table;

    // -------------------------------------------------------------------------
    // Search control
    // -------------------------------------------------------------------------
    SearchLimits    limits;
    std::atomic<bool> stop { false };  // Set externally to abort search

    // -------------------------------------------------------------------------
    // Time management
    // -------------------------------------------------------------------------
    std::chrono::steady_clock::time_point start_time;
    i32 allocated_time_ms = 0;  // How many ms we're allowed for this move

    // -------------------------------------------------------------------------
    // Statistics (reset each search)
    // -------------------------------------------------------------------------
    ui64  nodes      = 0;     // Total nodes visited
    ui64  tb_hits    = 0;     // TT hits (for hashfull reporting)
    i32   root_depth = 0;     // Current iterative deepening depth

    // -------------------------------------------------------------------------
    // Results
    // -------------------------------------------------------------------------
    Move  best_move;                  // Best move found so far
    i32   best_score = -INF;          // Score for best move

    // Per-ply stack (+4 for safety margin from check extensions)
    Stack stack[MAX_PLY + 4];

    // -------------------------------------------------------------------------
    // Heuristic tables (reset each search)
    // -------------------------------------------------------------------------

    // Killer moves: quiet moves that caused beta cutoffs indexed by [ply][slot]
    Move  killers[MAX_PLY][2];

    // History heuristic: bonus for quiet moves that caused cutoffs
    // Indexed [colour][from][to]
    i32   history[2][64][64];

    // -------------------------------------------------------------------------
    // Setup
    // -------------------------------------------------------------------------

    // Reset all search state for a new search
    void reset() {
        nodes      = 0;
        tb_hits    = 0;
        root_depth = 0;
        best_move  = NULL_MOVE;
        best_score = -INF;
        stop.store(false);

        for (i32 i = 0; i < MAX_PLY + 4; i++) {
            stack[i].current_move = NULL_MOVE;
            stack[i].killers[0]   = NULL_MOVE;
            stack[i].killers[1]   = NULL_MOVE;
            stack[i].ply          = i;
            stack[i].static_eval  = 0;
            stack[i].null_move_allowed = true;
        }

        for (i32 i = 0; i < MAX_PLY; i++) {
            killers[i][0] = NULL_MOVE;
            killers[i][1] = NULL_MOVE;
        }

        for (i32 c = 0; c < 2; c++)
            for (i32 f = 0; f < 64; f++)
                for (i32 t = 0; t < 64; t++)
                    history[c][f][t] = 0;
    }

    // -------------------------------------------------------------------------
    // Time management helpers
    // -------------------------------------------------------------------------

    // Compute allocated time based on limits and side to move
    void compute_allocated_time() {
        if (limits.infinite || limits.movetime == 0 && limits.wtime == 0) {
            allocated_time_ms = 0;  // no limit
            return;
        }

        if (limits.movetime > 0) {
            // Fixed time per move with small overhead buffer
            allocated_time_ms = limits.movetime - 10;
            return;
        }

        // Time control — allocate a fraction of remaining time
        i32 time_remaining = (pos.side_to_move() == WHITE)
            ? limits.wtime : limits.btime;
        i32 increment = (pos.side_to_move() == WHITE)
            ? limits.winc : limits.binc;

        i32 moves_to_go = (limits.movestogo > 0)
            ? limits.movestogo : 30;  // assume 30 moves to go in sudden death

        // Allocate time: remaining / moves_to_go + fraction of increment
        allocated_time_ms = (time_remaining / moves_to_go)
                          + (increment * 3 / 4);

        // Safety cap: never use more than half remaining time
        allocated_time_ms = std::min(allocated_time_ms, time_remaining / 2);

        // Minimum: at least 10ms to avoid immediate timeout
        allocated_time_ms = std::max(allocated_time_ms, 10);
    }

    i32 elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        return (i32)std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start_time).count();
    }

    bool out_of_time() const {
        if (allocated_time_ms == 0) return false;  // no limit
        return elapsed_ms() >= allocated_time_ms;
    }
};

// ============================================================================
// Search Entry Points
// ============================================================================

namespace Search {

// Run a full iterative deepening search.
// Populates info.best_move and info.best_score.
// Returns the best move found.
Move start(SearchInfo& info);

// Score adjustments for TT mate scores
// Scores stored in TT are relative to the node, not the root.
// These convert between the two representations.
inline i32 score_to_tt(i32 score, i32 ply) {
    if (score >  MATE_BOUND) return score + ply;
    if (score < -MATE_BOUND) return score - ply;
    return score;
}

inline i32 score_from_tt(i32 score, i32 ply) {
    if (score >  MATE_BOUND) return score - ply;
    if (score < -MATE_BOUND) return score + ply;
    return score;
}

} // namespace Search
