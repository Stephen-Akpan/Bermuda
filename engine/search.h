#pragma once

#include "position.h"
#include "move.h"
#include "tt.h"
#include "eval.h"
#include "repetition.h"
#include "nnue.h"
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

struct Stack {
    Move current_move;          // Move played to reach this node
    Move killers[2];            // Killer moves at this ply
    i32  ply;                   // Distance from root
    i32  static_eval;           // Cached static evaluation (set on first visit)
    bool null_move_allowed;     // False after a null move to prevent consecutive nulls
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
    i32 soft_limit_ms = 0;
    i32 hard_limit_ms = 0;

    // -------------------------------------------------------------------------
    // Statistics (reset each search)
    // -------------------------------------------------------------------------
    ui64  nodes      = 0;
    ui64  tb_hits    = 0;
    i32   root_depth = 0;

    // -------------------------------------------------------------------------
    // Results
    // -------------------------------------------------------------------------
    Move  best_move;
    i32   best_score = -INF;

    // Per-ply stack (+4 for safety margin from check extensions)
    Stack stack[MAX_PLY + 4];

    // -------------------------------------------------------------------------
    // Principal Variation table
    // -------------------------------------------------------------------------
    // Triangular array: pv[ply] holds the PV from that ply to the leaf.
    // pv_length[ply] is the index one past the last valid move in pv[ply].
    // At the root, pv[0][0..pv_length[0]-1] is the full PV line to print.
    Move pv[MAX_PLY][MAX_PLY];
    i32  pv_length[MAX_PLY];

    // -------------------------------------------------------------------------
    // Heuristic tables (reset each search)
    // -------------------------------------------------------------------------

    // Killer moves: quiet moves that caused beta cutoffs, indexed [ply][slot]
    Move  killers[MAX_PLY][2];

    // History heuristic: bonus for quiet moves that caused cutoffs
    // Indexed [colour][from][to]
    i32   history[2][64][64];

    // ------------------------------------------------------------------------
    // NNUE info
    // ------------------------------------------------------------------------
    NNUE_Data nnue_stack[MAX_PLY + 4];

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

        for (i32 i = 0; i < MAX_PLY + 4; i++) {
            stack[i].current_move      = NULL_MOVE;
            stack[i].killers[0]        = NULL_MOVE;
            stack[i].killers[1]        = NULL_MOVE;
            stack[i].ply               = i;
            stack[i].static_eval       = 0;
            stack[i].null_move_allowed = true;
        }

        for (i32 i = 0; i < MAX_PLY; i++) {
            killers[i][0] = NULL_MOVE;
            killers[i][1] = NULL_MOVE;
            pv_length[i]  = 0;
        }

        for (i32 c = 0; c < 2; c++)
            for (i32 f = 0; f < 64; f++)
                for (i32 t = 0; t < 64; t++)
                    history[c][f][t] = 0;
        
        for (i32 i = 0; i < MAX_PLY + 4; i++) {
            nnue_stack[i].accu.computed_accumulation = 0;
            nnue_stack[i].dirty_piece.dirty_num = 0;
        }
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
        i32 time_remaining = (pos.side_to_move() == WHITE)
            ? limits.wtime : limits.btime;
        i32 increment = (pos.side_to_move() == WHITE)
            ? limits.winc : limits.binc;

        // Assume 25 moves to go in sudden death (more realistic than 30)
        i32 moves_to_go = (limits.movestogo > 0) ? limits.movestogo : 25;

        // Base allocation: time slice + most of the increment
        i32 base = (time_remaining / moves_to_go) + (increment * 3 / 4);

        // Hard cap: never commit more than 80% of remaining time to one move
        i32 hard_cap = time_remaining * 7 / 10;

        // Soft limit: target time per move
        // Hard limit: 1.5x soft — allows finishing a nearly-complete iteration
        //             before aborting, but won't let us run away
        soft_limit_ms = min(base,         hard_cap);
        hard_limit_ms = min(base * 3 / 2, hard_cap);

        // Minimums — avoid degenerate near-zero allocations
        soft_limit_ms = max(soft_limit_ms, 5);
        hard_limit_ms = max(hard_limit_ms, 10);
    }

    i32 elapsed_ms() const {
        auto now = std::chrono::steady_clock::now();
        return (i32)std::chrono::duration_cast<std::chrono::milliseconds>(
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
