#include "search.h"
#include "movegen.h"
#include <algorithm>
#include <iostream>
#include <cmath>

namespace Search {

static constexpr i32 PIECE_TO_NNUE[2][6] = {
    //  PAWN        KNIGHT      BISHOP      ROOK        QUEEN       KING
    {   6,     5,   4,   3,     2,    1  },  // WHITE
    {   12,     11,   10,   9,     8,    7  },  // BLACK
};

static void fill_dirty_piece(const Position& pos, Move m, DirtyPiece& dp) {
    const Square from = m.from();
    const Square to = m.to();
    const Piece mover = pos.piece_on(from);
    const Colour us = colour_of(mover);

    dp.dirty_num = 1;

    if (m.flag() == CASTLING) {
        // King move
        dp.pc[0] = PIECE_TO_NNUE[us][KING];
        dp.from[0] = (i32)from;
        dp.to[0] = (i32)to;
        // Rook move
        Square rook_from, rook_to;
        if (to == g1) { rook_from = h1; rook_to = f1; } // White kingside
        else if (to == c1) { rook_from = a1; rook_to = d1; } // White kingside
        else if (to == g8) { rook_from = h8; rook_to = f8; } // Black kingside
        else { rook_from = a8; rook_to = d8; } // Black queenside
        dp.pc[1] = PIECE_TO_NNUE[us][ROOK];
        dp.from[1] = (i32)rook_from;
        dp.to[1] = (i32)rook_to;
        dp.dirty_num = 2;
    }
    else if (m.flag() == EN_PASSANT) {
        Square cap_sq = (us == WHITE) ? Square(to - 8) : Square(to + 8);
        // Pawn move
        dp.pc[0] = PIECE_TO_NNUE[us][PAWN];
        dp.from[0] = (i32)from;
        dp.to[0] = (i32)to;
        // Captured pawn removed
        dp.pc[1] = PIECE_TO_NNUE[flip(us)][PAWN];
        dp.from[1] = (i32)cap_sq;
        dp.to[1] = (i32)64;
        dp.dirty_num = 2;
    }
    else if (m.flag() == PROMOTION) {
        PieceType promo_pt;

        switch (m.promo()) {
        case PROMO_KNIGHT: promo_pt = KNIGHT; break;
        case PROMO_BISHOP: promo_pt = BISHOP; break;
        case PROMO_ROOK: promo_pt = ROOK; break;
        case PROMO_QUEEN: promo_pt = QUEEN; break;
        default: promo_pt = QUEEN; break;
        }
        dp.pc[0] = PIECE_TO_NNUE[us][PAWN];
        dp.from[1] = (i32)from;
        dp.to[1] = 64;
        
        dp.pc[1] = PIECE_TO_NNUE[us][promo_pt];
        dp.from[1] = 64;
        dp.to[1] = (i32)to;

        dp.dirty_num = 2;

        Piece captured = pos.piece_on(to);
        if (captured != NO_PIECE) {
            dp.pc[2] = PIECE_TO_NNUE[us][type_of(captured)];
            dp.from[2] = (i32)to;
            dp.to[2] = 64;
            dp.dirty_num = 3;
        }
    }
    else { // NORMAL
        dp.pc[0] = PIECE_TO_NNUE[us][type_of(mover)];
        dp.from[0] = (i32)from;
        dp.to[0] = (i32)to;

        Piece captured = pos.piece_on(to);

        if (captured != NO_PIECE) {
            dp.pc[1] = PIECE_TO_NNUE[colour_of(captured)][type_of(captured)];
            dp.from[1] = (i32)to;
            dp.to[1] = 64;
            dp.dirty_num = 2;
        }
    }
}

// --------------------------------------------------------------------------
// Helper for ply-bounds logic
// --------------------------------------------------------------------------
#define NNUE_PTRS(info, ply) \
    NNUE_Data* nd0 = &(info).nnue_stack[(ply)]; \
    NNUE_Data* nd1 = ((ply) > 0) ? &(info).nnue_stack[(ply) - 1] : nullptr; \
    NNUE_Data* nd2 = ((ply) > 1) ? &(info).nnue_stack[(ply) - 2] : nullptr; \

// ============================================================================
// LMR Reduction Table
// ============================================================================
// Precomputed at startup: reductions[depth][move_count]
// Formula: 1 + ln(depth) * ln(move_count) / 2
// Clamped to [1, depth-1] at call site.

static i32 LMR_TABLE[MAX_PLY][64];

static void init_lmr_table() {
    for (i32 depth = 1; depth < MAX_PLY; depth++) {
        for (i32 mc = 1; mc < 64; mc++) {
            LMR_TABLE[depth][mc] = static_cast<i32>(
                1.0 + std::log(depth) * std::log(mc) / 2.0);
        }
    }
}

// ============================================================================
// Futility Pruning Margins
// ============================================================================
// Indexed by depth (1-3). Represents the maximum positional gain we expect
// from a single quiet move. If static_eval + margin <= alpha, the move is
// unlikely to raise alpha and can be skipped.
//
// depth 1: ~one pawn      — fine-grained positional adjustments
// depth 2: ~minor piece   — a knight/bishop swing
// depth 3: ~a rook        — large positional swings at this depth are rare

static constexpr i32 FUTILITY_MARGIN[4] = { 0, 100, 300, 500 };
static constexpr i32 FUTILITY_MAX_DEPTH = 3;

// ============================================================================
// Piece values for delta pruning in quiescence search
// ============================================================================
// Indexed by PieceType (0=Pawn ... 5=King).
// King is given a large value so king captures are never pruned.

static constexpr i32 DELTA_PIECE_VALUE[6] = {
    100,   // Pawn
    300,   // Knight
    300,   // Bishop
    500,   // Rook
    900,   // Queen
    20000, // King — should never be captured, but guard anyway
};

// Safety margin for delta pruning — accounts for positional swings
// on top of the raw material gain from a capture.
static constexpr i32 DELTA_MARGIN = 200;


// ============================================================================
// Move Ordering
// ============================================================================

// MVV-LVA table: [attacker][victim] — higher = better capture
// Indexed by PieceType (0=Pawn ... 5=King)
static constexpr i32 MVV_LVA[6][6] = {
    // Victim:  P    N    B    R    Q    K
    /* P */  { 105, 205, 305, 405, 505, 605 },
    /* N */  { 104, 204, 304, 404, 504, 604 },
    /* B */  { 103, 203, 303, 403, 503, 603 },
    /* R */  { 102, 202, 302, 402, 502, 602 },
    /* Q */  { 101, 201, 301, 401, 501, 601 },
    /* K */  { 100, 200, 300, 400, 500, 600 },
};

// Score a single move for ordering purposes
static i32 score_move(const Position& pos, Move m, Move tt_move,
                      const Move killers[2], const i32 history[64][64]) {
    if (m == tt_move)
        return 1'000'000;

    if (m.is_promotion())
        return 900'000 + (i32)m.promo() * 1000;

    Square to     = m.to();
    Piece  victim = pos.piece_on(to);

    if (m.is_en_passant())
        return 800'000;

    if (victim != NO_PIECE) {
        Piece attacker = pos.piece_on(m.from());
        return 800'000 + MVV_LVA[type_of(attacker)][type_of(victim)];
    }

    if (m == killers[0]) return 700'000;
    if (m == killers[1]) return 699'000;

    return history[m.from()][m.to()];
}

// Sort moves in-place using insertion sort (fast for small lists)
static void order_moves(const Position& pos, MoveList& list, Move tt_move,
                        const Move killers[2], const i32 history[64][64]) {
    i32 scores[256];
    for (i32 i = 0; i < list.size(); i++)
        scores[i] = score_move(pos, list[i], tt_move, killers, history);

    for (i32 i = 1; i < list.size(); i++) {
        Move m = list[i];
        i32  s = scores[i];
        i32  j = i - 1;

        while (j >= 0 && scores[j] < s) {
            list[j + 1]  = list[j];
            scores[j + 1] = scores[j];
            j--;
        }

        list[j + 1]  = m;
        scores[j + 1] = s;
    }
}

// Update killer moves — keep two killers per ply, no duplicates
static void update_killers(SearchInfo& info, i32 ply, Move m) {
    if (info.killers[ply][0] != m) {
        info.killers[ply][1] = info.killers[ply][0];
        info.killers[ply][0] = m;
    }
}

// Update history heuristic — bonus for quiet moves that caused cutoffs
static void update_history(SearchInfo& info, Move m, i32 depth) {
    i32 colour = (i32)info.pos.side_to_move();
    i32 bonus  = depth * depth;

    i32& entry = info.history[colour][m.from()][m.to()];
    entry = min(entry + bonus, 10000);
}

// ============================================================================
// Quiescence Search
// ============================================================================

static i32 quiescence(SearchInfo& info, Stack* ss, i32 alpha, i32 beta) {
    info.nodes++;

    if ((info.nodes & 2047) == 0) {
        if (info.past_hard_limit()) {
            info.stop.store(true);
        }
    }

    NNUE_PTRS(info, ss->ply);
    i32 stand_pat  = Eval::evaluate(info.pos, nd0, nd1, nd2);

    if (info.stop.load(std::memory_order_relaxed))
        return stand_pat;

    if (ss->ply >= MAX_PLY - 1)
        return stand_pat;

    
    i32 best_score = stand_pat;

    if (stand_pat >= beta)
        return stand_pat;

    if (stand_pat > alpha)
        alpha = stand_pat;

    MoveList captures;
    MoveGen::generate_captures(info.pos, captures);

    order_moves(info.pos, captures, NULL_MOVE,
                info.killers[ss->ply],
                info.history[(i32)info.pos.side_to_move()]);

    for (i32 i = 0; i < captures.size(); i++) {
        Move m = captures[i];

        // ---------------------------------------------------------------------
        // Delta pruning
        // ---------------------------------------------------------------------
        // If the stand-pat score plus the captured piece's material value plus
        // a safety margin still can't raise alpha, this capture has no realistic
        // chance of improving our position. Skip it.
        //
        // Guards:
        //   - Not in check: eval is unreliable under check, don't prune
        //   - Not a promotion: a pawn capture leading to promotion is worth
        //     ~900cp+, far more than DELTA_PIECE_VALUE[PAWN] would suggest
        //   - Not a near-mate score: material margins are meaningless near mate
        if (!info.pos.in_check() && !m.is_promotion() && !is_mate_score(alpha)) {
            Piece victim = m.is_en_passant()
                ? make_piece(flip(info.pos.side_to_move()), PAWN)
                : info.pos.piece_on(m.to());

            if (victim != NO_PIECE) {
                i32 gain = DELTA_PIECE_VALUE[type_of(victim)];
                if (stand_pat + gain + DELTA_MARGIN <= alpha)
                    continue;
            }
        }

        NNUE_Data* nd_child = &info.nnue_stack[ss->ply + 1];
        nd_child->accu.computed_accumulation = 0;
        fill_dirty_piece(info.pos, m, nd_child->dirty_piece);

        UndoInfo undo = info.pos.make_move(m);
        info.rep_table.push(info.pos.hash());

        if (info.pos.is_attacked_by(
                info.pos.king_square(flip(info.pos.side_to_move())),
                info.pos.side_to_move())) {
            info.rep_table.pop();
            info.pos.unmake_move(m, undo);
            continue;
        }

        (ss + 1)->ply = ss->ply + 1;
        i32 score = -quiescence(info, ss + 1, -beta, -alpha);

        info.rep_table.pop();
        info.pos.unmake_move(m, undo);

        if (info.stop.load(std::memory_order_relaxed))
            return 0;

        if (score > best_score)
            best_score = score;

        if (score >= beta)
            return best_score;

        if (score > alpha)
            alpha = score;
    }

    return best_score;
}

// ============================================================================
// Main Search
// ============================================================================

static i32 search(SearchInfo& info, Stack* ss, i32 depth, i32 alpha, i32 beta) {
    bool is_pv   = (beta - alpha) > 1;
    bool is_root = (ss->ply == 0);

    info.nodes++;

    // Initialise PV length for this ply — prevents stale moves from a previous
    // search iteration bleeding into the PV if this node fails to raise alpha.
    info.pv_length[ss->ply] = ss->ply;

    // -------------------------------------------------------------------------
    // Early exits
    // -------------------------------------------------------------------------

    if (ss->ply >= MAX_PLY - 1) {
        NNUE_PTRS(info, ss->ply);
        return Eval::evaluate(info.pos, nd0, nd1, nd2);
    }

    if (depth <= 0)
        return quiescence(info, ss, alpha, beta);

    if (!is_root) {
        if (info.rep_table.is_draw(info.pos.hash(), info.pos.halfmove()))
            return 0;
    }

    // Periodic hard-limit check — abort mid-search if we've overrun
    if ((info.nodes & 2047) == 0) {
        if (info.past_hard_limit())
            info.stop.store(true);
    }

    if (info.stop.load(std::memory_order_relaxed))
        return 0;

    // -------------------------------------------------------------------------
    // Check extension
    // -------------------------------------------------------------------------
    bool in_check = info.pos.in_check();
    if (in_check && ss->ply < MAX_PLY / 2)
        depth++;

    // -------------------------------------------------------------------------
    // Transposition Table probe
    // -------------------------------------------------------------------------
    i32  original_alpha = alpha;
    Move tt_move        = NULL_MOVE;

    const TTEntry* tt_entry = TT.probe(info.pos.hash());
    if (tt_entry) {
        info.tb_hits++;
        tt_move = tt_entry->move;

        if (!is_pv && tt_entry->depth >= depth) {
            i32 tt_score = score_from_tt(tt_entry->score, ss->ply);

            if (tt_entry->bound == BOUND_EXACT)
                return tt_score;
            if (tt_entry->bound == BOUND_LOWER && tt_score >= beta)
                return tt_score;
            if (tt_entry->bound == BOUND_UPPER && tt_score <= alpha)
                return tt_score;
        }
    }

    // -------------------------------------------------------------------------
    // Null Move Pruning
    // -------------------------------------------------------------------------
    if (!is_pv
        && !in_check
        && ss->null_move_allowed
        && depth >= 3
        && ss->ply >= 1)
    {
        Colour us = info.pos.side_to_move();
        bool has_pieces = (info.pos.pieces(us, KNIGHT)
                         | info.pos.pieces(us, BISHOP)
                         | info.pos.pieces(us, ROOK)
                         | info.pos.pieces(us, QUEEN)) != 0;

        if (has_pieces) {
            i32 R = 3 + depth / 6;

            NNUE_Data* nd_null = &info.nnue_stack[ss->ply + 1];
            nd_null->accu.computed_accumulation = 0;
            nd_null->dirty_piece.dirty_num = 0;

            NullMoveUndo null_undo = info.pos.make_null_move();
            info.rep_table.push(info.pos.hash());

            (ss + 1)->ply               = ss->ply + 1;
            (ss + 1)->null_move_allowed = false;

            i32 null_score = -search(info, ss + 1, depth - 1 - R, -beta, -beta + 1);

            info.rep_table.pop();
            info.pos.unmake_null_move(null_undo);

            (ss + 1)->null_move_allowed = true;

            if (info.stop.load(std::memory_order_relaxed))
                return 0;

            if (null_score >= beta) {
                // Don't trust null move for mate scores — cap at beta
                if (null_score >= MATE_BOUND)
                    null_score = beta;

                return null_score;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Static evaluation — computed once per node, used for pruning decisions.
    // Skipped when in check since eval is unreliable under check.
    // -------------------------------------------------------------------------
    if (!in_check) {
        NNUE_PTRS(info, ss->ply);
        ss->static_eval = Eval::evaluate(info.pos, nd0, nd1, nd2);
    }

    // -------------------------------------------------------------------------
    // Generate and order moves
    // -------------------------------------------------------------------------
    MoveList moves;
    MoveGen::generate_moves(info.pos, moves);

    order_moves(info.pos, moves, tt_move,
                info.killers[ss->ply],
                info.history[(i32)info.pos.side_to_move()]);

    i32  legal_moves = 0;
    Move best_move   = NULL_MOVE;
    i32  best_score  = -INF;

    for (i32 i = 0; i < moves.size(); i++) {
        Move m = moves[i];

        // ---------------------------------------------------------------------
        // Futility pruning
        // ---------------------------------------------------------------------
        // At low depths, if the static eval plus a generous margin still can't
        // reach alpha, quiet moves have no realistic chance of improving alpha.
        // Skip them. Captures and promotions are always searched — they can
        // swing the score by large amounts regardless of the margin.
        //
        // Guards:
        //   - Not in check: we must search all moves when in check
        //   - Not a PV node: futility is only safe in zero-window nodes
        //   - Not a mate score: margins are meaningless near forced mates
        //   - At least one legal move found: never prune before we have a move
        //     (we need at least one to detect checkmate vs stalemate correctly)
        bool is_quiet_move = !m.is_en_passant()
                           && !m.is_promotion()
                           && info.pos.piece_on(m.to()) == NO_PIECE;

        if (!in_check
            && !is_pv
            && is_quiet_move
            && legal_moves > 0
            && depth <= FUTILITY_MAX_DEPTH
            && !is_mate_score(alpha)
            && ss->static_eval + FUTILITY_MARGIN[depth] <= alpha)
        {
            continue;
        }

        NNUE_Data* nd_child = &info.nnue_stack[ss->ply + 1];
        nd_child->accu.computed_accumulation = 0;
        fill_dirty_piece(info.pos, m, nd_child->dirty_piece);

        UndoInfo undo = info.pos.make_move(m);

        if (info.pos.is_attacked_by(
                info.pos.king_square(flip(info.pos.side_to_move())),
                info.pos.side_to_move())) {
            info.pos.unmake_move(m, undo);
            continue;
        }

        legal_moves++;
        info.rep_table.push(info.pos.hash());
        (ss + 1)->ply               = ss->ply + 1;
        (ss + 1)->current_move      = m;
        (ss + 1)->null_move_allowed = true;

        // -------------------------------------------------------------------------
        // PVS + LMR
        // -------------------------------------------------------------------------
        i32 score;

        if (legal_moves == 1) {
            // First move: always search at full depth and full window
            score = -search(info, ss + 1, depth - 1, -beta, -alpha);
        } else {
            // LMR conditions:
            //   - depth must be high enough to make reduction worthwhile
            //   - enough moves already searched (we've seen the good ones)
            //   - quiet move only — captures and promotions are always full depth
            //   - not in check — we don't reduce when king safety is at stake
            bool do_lmr = depth >= 3
                       && legal_moves >= 3
                       && is_quiet_move
                       && !in_check;

            i32 reduction = 0;

            if (do_lmr) {
                // Look up reduction from precomputed table.
                // Clamp move_count index to table bounds.
                i32 mc_idx = min(legal_moves, 63);
                reduction  = LMR_TABLE[depth][mc_idx];

                // Never reduce to zero or below — that would be a quiescence
                // call, which is handled separately. Minimum reduced depth is 1.
                reduction = min(reduction, depth - 1);
            }

            // Null-window search at reduced depth (or full depth if no reduction)
            score = -search(info, ss + 1, depth - 1 - reduction, -alpha - 1, -alpha);

            // If the reduced search raised alpha, we need to verify with a
            // full-depth null-window re-search. This catches cases where the
            // reduction was too aggressive.
            if (score > alpha && reduction > 0)
                score = -search(info, ss + 1, depth - 1, -alpha - 1, -alpha);

            // If the move looks like it could be best on a PV node, re-search
            // with the full window to get an exact score.
            if (score > alpha && score < beta)
                score = -search(info, ss + 1, depth - 1, -beta, -alpha);
        }

        info.rep_table.pop();
        info.pos.unmake_move(m, undo);

        if (info.stop.load(std::memory_order_relaxed))
            return 0;

        if (score > best_score) {
            best_score = score;
            best_move  = m;

            if (is_root) {
                info.best_move  = m;
                info.best_score = score;
            }
        }

        if (score > alpha) {
            alpha = score;

            // Update the PV at PV nodes only.
            // At zero-window nodes (is_pv == false) we don't maintain the PV —
            // those searches are probes, not exact results.
            if (is_pv) {
                // This move is the new PV move at this ply
                info.pv[ss->ply][ss->ply] = m;

                // Copy the child's PV into this ply's PV after the current move
                for (i32 next = ss->ply + 1; next < info.pv_length[ss->ply + 1]; next++)
                    info.pv[ss->ply][next] = info.pv[ss->ply + 1][next];

                // Our PV length is now the child's PV length
                info.pv_length[ss->ply] = info.pv_length[ss->ply + 1];
            }
        }

        if (alpha >= beta) {
            if (!m.is_en_passant()
                && !m.is_promotion()
                && info.pos.piece_on(m.to()) == NO_PIECE) {
                update_killers(info, ss->ply, m);
                update_history(info, m, depth);
            }
            break;
        }
    }

    // -------------------------------------------------------------------------
    // No legal moves — checkmate or stalemate
    // -------------------------------------------------------------------------
    if (legal_moves == 0) {
        if (in_check)
            return mated_in(ss->ply);
        return 0;
    }

    // -------------------------------------------------------------------------
    // Transposition Table store
    // -------------------------------------------------------------------------
    Bound bound;
    if (best_score <= original_alpha)
        bound = BOUND_UPPER;
    else if (best_score >= beta)
        bound = BOUND_LOWER;
    else
        bound = BOUND_EXACT;

    TT.store(info.pos.hash(),
             best_move,
             (i16)score_to_tt(best_score, ss->ply),
             (ui8)depth,
             bound);

    return best_score;
}

// ============================================================================
// Iterative Deepening
// ============================================================================

Move start(SearchInfo& info) {
    static bool lmr_initialised = false;
    if (!lmr_initialised) {
        init_lmr_table();
        lmr_initialised = true;
    }

    info.reset();
    info.start_time = std::chrono::steady_clock::now();
    info.compute_allocated_time();

    info.rep_table.push(info.pos.hash());

    Move best_move  = NULL_MOVE;
    i32  best_score = -INF;

    // Initial aspiration window half-width.
    // Only used from ASPIRATION_MIN_DEPTH onwards — below that scores
    // are too volatile for a narrow window to be useful.
    constexpr i32 ASPIRATION_DELTA     = 25;
    constexpr i32 ASPIRATION_MIN_DEPTH = 5;

    info.nnue_stack[0].accu.computed_accumulation = 0;
    info.nnue_stack[0].dirty_piece.dirty_num = 0;
    Eval::evaluate(info.pos, &info.nnue_stack[0], nullptr, nullptr); // computed_accumulation = 1

    for (i32 depth = 1; depth <= info.limits.depth; depth++) {
        info.root_depth = depth;

        info.stack[0].ply               = 0;
        info.stack[0].current_move      = NULL_MOVE;
        info.stack[0].null_move_allowed = true;

        i32 score;

        if (depth < ASPIRATION_MIN_DEPTH) {
            // Full window at low depths — scores are too volatile for aspiration
            score = search(info, &info.stack[0], depth, -INF, INF);
        } else {
            // Narrow window centred on previous iteration's score.
            // If the true score falls outside, widen and re-search.
            i32 delta = ASPIRATION_DELTA;
            i32 alpha = best_score - delta;
            i32 beta  = best_score + delta;

            while (true) {
                score = search(info, &info.stack[0], depth, alpha, beta);

                if (info.stop.load(std::memory_order_relaxed))
                    break;

                if (score <= alpha) {
                    // Fail-low: true score is below our window.
                    // Widen downward and keep beta where it is —
                    // we know the score is below beta so no need to move it.
                    alpha -= delta;
                    delta *= 2;

                    // Safety: don't go below -INF
                    if (alpha < -INF) alpha = -INF;

                } else if (score >= beta) {
                    // Fail-high: true score is above our window.
                    // Widen upward and keep alpha where it is.
                    // Note: on fail-high we do NOT update best_move yet —
                    // the search was cut short and info.best_move may be
                    // a lower bound, not the true best.
                    beta += delta;
                    delta *= 2;

                    // Safety: don't go above +INF
                    if (beta > INF) beta = INF;

                } else {
                    // Score is inside the window — exact result, we're done
                    break;
                }
            }
        }

        // Don't update best move from an aborted search
        if (info.stop.load(std::memory_order_relaxed))
            break;

        best_move  = info.best_move;
        best_score = score;

        // Print UCI info line
        i32 elapsed = info.elapsed_ms();
        i32 nps     = elapsed > 0
            ? (i32)(info.nodes * 1000 / elapsed)
            : (i32)info.nodes;

        std::cout << "info depth " << depth
                  << " score ";

        if (is_mate_score(best_score)) {
            i32 moves_to_mate = (MATE - std::abs(best_score) + 1) / 2;
            std::cout << "mate "
                      << (best_score > 0 ? moves_to_mate : -moves_to_mate);
        } else {
            std::cout << "cp " << best_score;
        }

        std::cout << " nodes " << info.nodes
                  << " nps "   << nps
                  << " time "  << elapsed
                  << " pv";

        for (i32 i = 0; i < info.pv_length[0]; i++)
            std::cout << " " << info.pv[0][i].to_string();

        std::cout << "\n";

        // Soft limit: don't start a new iteration if we're already over time
        if (info.past_soft_limit())
            break;
    }

    info.rep_table.pop();

    // Fallback — return first legal move if search produced nothing
    if (best_move.is_null()) {
        MoveList moves;
        MoveGen::generate_moves(info.pos, moves);

        for (i32 i = 0; i < moves.size(); i++) {
            UndoInfo undo = info.pos.make_move(moves[i]);
            bool legal = !info.pos.is_attacked_by(
                info.pos.king_square(flip(info.pos.side_to_move())),
                info.pos.side_to_move());
            info.pos.unmake_move(moves[i], undo);

            if (legal) {
                best_move = moves[i];
                break;
            }
        }
    }

    info.best_move = best_move;

    std::cout << "bestmove " << best_move.to_string() << "\n";

    return best_move;
}

} // namespace Search
