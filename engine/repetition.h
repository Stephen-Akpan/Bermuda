#pragma once

#include "types.h"
#include <algorithm>

// ============================================================================
// Repetition Table
// ============================================================================
// Tracks position hashes seen during the game and current search to detect
// draw by repetition (threefold repetition) and the fifty-move rule.
//
// Design:
//   - A flat array of hashes indexed by ply/move number
//   - The game history (moves before search started) is loaded at startup
//   - The search extends this array as it makes/unmakes moves
//   - Repetition is detected by scanning backward through the array
//
// We only need to scan back as far as the last irreversible move (capture
// or pawn move) since no position before that can repeat. The halfmove
// clock tells us exactly how far back to look.

struct RepetitionTable {
    static constexpr i32 MAX_GAME_LENGTH = 1024;

    ui64 hashes[MAX_GAME_LENGTH];
    i32  count = 0;   // number of entries currently stored

    // -------------------------------------------------------------------------
    // Setup
    // -------------------------------------------------------------------------

    void clear() { count = 0; }

    // Push a hash when entering a position (make_move or game history load)
    void push(ui64 hash) {
        hashes[count++] = hash;
    }

    // Pop a hash when leaving a position (unmake_move)
    void pop() {
        if (count > 0) count--;
    }

    // -------------------------------------------------------------------------
    // Repetition check
    // -------------------------------------------------------------------------

    // Returns true if the given hash has appeared before in the current line.
    // halfmove_clock tells us how many plies back we need to check —
    // positions before the last irreversible move cannot repeat.
    //
    // We check for a single repetition (not threefold) during search —
    // treating any repetition as a draw is conservative but correct and
    // avoids the engine walking into repetitions thinking it can escape.
    bool is_repetition(ui64 hash, i32 halfmove_clock) const {
        // Repetitions can only occur at same-colour positions (every 2 plies)
        // and only within the halfmove clock window
        i32 lookback = std::min(halfmove_clock, count - 1);

        // Walk back in steps of 2 (same side to move) within the window
        for (i32 i = count - 2; i >= count - lookback; i -= 2) {
            if (hashes[i] == hash)
                return true;
        }

        return false;
    }

    // Returns true if the fifty-move rule applies
    bool is_fifty_move(i32 halfmove_clock) const {
        return halfmove_clock >= 100;
    }

    // Combined draw check used in search
    bool is_draw(ui64 hash, i32 halfmove_clock) const {
        return is_fifty_move(halfmove_clock)
            || is_repetition(hash, halfmove_clock);
    }
};
