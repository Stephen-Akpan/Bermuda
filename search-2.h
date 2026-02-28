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

constexpr int INF       = 32000;
constexpr int MATE      = 30000;
constexpr int MAX_PLY   = 128;

// Score thresholds for mate detection
constexpr int MATE_BOUND = MATE - MAX_PLY;

// Helpers for encoding mate distance into score
inline int mate_in(int ply)  { return  MATE - ply; }
inline int mated_in(int ply) { return -MATE + ply; }

// Returns true if the score represents a forced mate
inline bool is_mate_score(int score) {
    return score > MATE_BOUND || score < -MATE_BOUND;
}

// ============================================================================
// Per-Ply Search Stack
// ============================================================================

struct Stack {
    Move current_move;          // Move played to reach this node
    Move killers[2];            // Killer moves at this ply
    int  ply;                   // Distance from root
    int  static_eval;           // Cached static evaluation (set on first visit)
    bool null_move_allowed;     // False after a null move to prevent consecutive nulls
};

// ============================================================================
// Search Limits
// ============================================================================

struct SearchLimits {
    int  depth      = MAX_PLY;  // Maximum search depth
    int  movetime   = 0;        // Fixed time per move in ms (0 = none)
    int  wtime      = 0;        // White time remaining in ms
    int  btime      = 0;        // Black time remaining in ms
    int  winc       = 0;        // White increment in ms
    int  binc       = 0;        // Black increment in ms
    int  movestogo  = 0;        // Moves until next time control (0 = sudden death)
    bool infinite   = false;    // Search indefinitely (ignore time)
    bool ponder     = false;    // Pondering mode
};

// ============================================================================
// Search Info
// ============================================================================

struct SearchInfo {
    // -------------------------------------------------------------------------
    // Position and game history
    // -------------------------------------------------------------------------
    Position        pos;
    RepetitionTable rep_table;

    // -------------------------------------------------------------------------
    // Search control
    // -------------------------------------------------------------------------
    SearchLimits      limits;
    std::atomic<bool> stop { false };

    // -------------------------------------------------------------------------
    // Time management
    // -------------------------------------------------------------------------
    std::chrono::steady_clock::time_point start_time;

    // Soft limit: don't start a new iteration if exceeded after one completes.
    // Hard limit: abort immediately mid-search if exceeded.
    // Both are 0 when there is no time constraint.
    int soft_limit_ms = 0;
    int hard_limit_ms = 0;

    // -------------------------------------------------------------------------
    // Statistics (reset each search)
    // -------------------------------------------------------------------------
    ui64  nodes      = 0;
    ui64  tb_hits    = 0;
    int   root_depth = 0;

    // -------------------------------------------------------------------------
    // Results
    // -------------------------------------------------------------------------
    Move  best_move;
    int   best_score = -INF;

    // Per-ply stack (+4 for safety margin from check extensions)
    Stack stack[MAX_PLY + 4];

    // -------------------------------------------------------------------------
    // Heuristic tables (reset each search)
    // -------------------------------------------------------------------------

    // Killer moves: quiet moves that caused beta cutoffs, indexed [ply][slot]
    Move  killers[MAX_PLY][2];

    // History heuristic: bonus for quiet moves that caused cutoffs
    // Indexed [colour][from][to]
    int   history[2][64][64];

    // -------------------------------------------------------------------------
    // Setup
    // -------------------------------------------------------------------------

    void reset() {
        nodes      = 0;
        tb_hits    = 0;
        root_depth = 0;
        best_move  = NULL_MOVE;
        best_score = -INF;
        stop.store(false);

        for (int i = 0; i < MAX_PLY + 4; i++) {
            stack[i].current_move      = NULL_MOVE;
            stack[i].killers[0]        = NULL_MOVE;
            stack[i].killers[1]        = NULL_MOVE;
            stack[i].ply               = i;
            stack[i].static_eval       = 0;
            stack[i].null_move_allowed = true;
        }

        for (int i = 0; i < MAX_PLY; i++) {
            killers[i][0] = NULL_MOVE;
            killers[i][1] = NULL_MOVE;
        }

        for (int c = 0; c < 2; c++)
            for (int f = 0; f < 64; f++)
                for (int t = 0; t < 64; t++)
                    history[c][f][t] = 0;
    }

    // -------------------------------------------------------------------------
    // Time management
    // -------------------------------------------------------------------------

    void compute_allocated_time() {
        // No time limit for infinite or ponder searches
        if (limits.infinite || limits.ponder) {
            soft_limit_ms = 0;
            hard_limit_ms = 0;
            return;
        }

        // Fixed movetime — soft and hard are identical
        if (limits.movetime > 0) {
            soft_limit_ms = limits.movetime;
            hard_limit_ms = limits.movetime;
            return;
        }

        // No time info provided at all — search indefinitely
        if (limits.wtime == 0 && limits.btime == 0) {
            soft_limit_ms = 0;
            hard_limit_ms = 0;
            return;
        }

        // Time control — allocate a fraction of remaining time
        int time_remaining = (pos.side_to_move() == WHITE)
            ? limits.wtime : limits.btime;
        int increment = (pos.side_to_move() == WHITE)
            ? limits.winc : limits.binc;

        // Assume 25 moves to go in sudden death (more realistic than 30)
        int moves_to_go = (limits.movestogo > 0) ? limits.movestogo : 25;

        // Base allocation: time slice + most of the increment
        int base = (time_remaining / moves_to_go) + (increment * 3 / 4);

        // Hard cap: never commit more than 80% of remaining time to one move
        int hard_cap = time_remaining * 8 / 10;

        // Soft limit: target time per move
        // Hard limit: 1.5x soft — allows finishing a nearly-complete iteration
        //             before aborting, but won't let us run away
        soft_limit_ms = std::min(base,         hard_cap);
        hard_limit_ms = std::min(base * 3 / 2, hard_cap);

        // Minimums — avoid degenerate near-zero allocations
        soft_limit_ms = std::max(soft_limit_ms, 5);
        hard_limit_ms = std::max(hard_limit_ms, 10);
    }

    int elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        return (int)std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start_time).count();
    }

    // Exceeded soft limit: stop after the current iteration completes
    bool past_soft_limit() const {
        if (soft_limit_ms == 0) return false;
        return elapsed_ms() >= soft_limit_ms;
    }

    // Exceeded hard limit: abort immediately mid-search
    bool past_hard_limit() const {
        if (hard_limit_ms == 0) return false;
        return elapsed_ms() >= hard_limit_ms;
    }
};

// ============================================================================
// Search Entry Points
// ============================================================================

namespace Search {

Move start(SearchInfo& info);

// Score adjustments for TT mate scores.
// Scores in the TT are relative to the storing node; these convert to/from root.
inline int score_to_tt(int score, int ply) {
    if (score >  MATE_BOUND) return score + ply;
    if (score < -MATE_BOUND) return score - ply;
    return score;
}

inline int score_from_tt(int score, int ply) {
    if (score >  MATE_BOUND) return score - ply;
    if (score < -MATE_BOUND) return score + ply;
    return score;
}

} // namespace Search
