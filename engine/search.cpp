#include "search.h"
#include "movegen.h"
#include <algorithm>
#include <iostream>

namespace Search {

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
    // TT move — search first
    if (m == tt_move)
        return 1'000'000;

    if (m.is_promotion())
        return 900'000 + (i32)m.promo() * 1000;

    // Captures — scored by MVV-LVA
    Square to = m.to();
    Piece victim = pos.piece_on(to);

    if (m.is_en_passant())
        return 800'000;  // EP is always a pawn capture

    if (victim != NO_PIECE) {
        Piece attacker = pos.piece_on(m.from());
        return 800'000 + MVV_LVA[type_of(attacker)][type_of(victim)];
    }

    // Killer moves — quiet moves that caused cutoffs at this ply
    if (m == killers[0]) return 700'000;
    if (m == killers[1]) return 699'000;

    // History heuristic — quiet moves ordered by past cutoff frequency
    return history[m.from()][m.to()];
}

// Sort moves in-place using insertion sort (fast for small lists)
static void order_moves(const Position& pos, MoveList& list, Move tt_move,
                        const Move killers[2], const i32 history[64][64]) {
    // Score all moves first
    i32 scores[256];
    for (i32 i = 0; i < list.size(); i++)
        scores[i] = score_move(pos, list[i], tt_move, killers, history);

    // Insertion sort — stable, fast for ~30-50 moves
    for (i32 i = 1; i < list.size(); i++) {
        Move m    = list[i];
        i32  s    = scores[i];
        i32  j    = i - 1;

        while (j >= 0 && scores[j] < s) {
            list[j + 1] = list[j];
            scores[j + 1] = scores[j];
            j--;
        }

        list[j + 1] = m;
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
    i32 bonus  = depth * depth;  // quadratic bonus — deeper cutoffs matter more

    // Clamp to prevent overflow
    i32& entry = info.history[colour][m.from()][m.to()];
    entry = std::min(entry + bonus, 10000);
}

// ============================================================================
// Quiescence Search
// ============================================================================

static i32 quiescence(SearchInfo& info, Stack* ss, i32 alpha, i32 beta) {
    info.nodes++;

    // Ply limit safety
    if (ss->ply >= MAX_PLY - 1)
        return Eval::evaluate(info.pos);

    // Stand-pat: the score we get by doing nothing (lower bound)
    i32 stand_pat = Eval::evaluate(info.pos);
    i32 best_score = stand_pat;

    if (stand_pat >= beta)
        return stand_pat;   // fail-soft: return actual score not beta

    if (stand_pat > alpha)
        alpha = stand_pat;

    // Generate and order captures only
    MoveList captures;
    MoveGen::generate_captures(info.pos, captures);

    // Order captures by MVV-LVA (no TT move or killers in qsearch)
    order_moves(info.pos, captures, NULL_MOVE,
                info.killers[ss->ply], info.history[(i32)info.pos.side_to_move()]);

    for (i32 i = 0; i < captures.size(); i++) {
        Move m = captures[i];

        UndoInfo undo = info.pos.make_move(m);
        info.rep_table.push(info.pos.hash());

        // Skip illegal moves
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

        if (score > best_score) {
            best_score = score;
        }

        if (score >= beta)
            return best_score;  // fail-soft

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

    // -------------------------------------------------------------------------
    // Early exits
    // -------------------------------------------------------------------------

    // Ply limit
    if (ss->ply >= MAX_PLY - 1)
        return Eval::evaluate(info.pos);

    // Drop into quiescence at depth 0
    if (depth <= 0)
        return quiescence(info, ss, alpha, beta);

    // Draw detection — repetition and fifty-move rule
    // Don't return draw at root (we need a best move)
    if (!is_root) {
        if (info.rep_table.is_draw(info.pos.hash(), info.pos.halfmove()))
            return 0;
    }

    // Check periodically if we're out of time
    if ((info.nodes & 2047) == 0) {
        if (info.out_of_time())
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
    i32    original_alpha = alpha;
    Move   tt_move        = NULL_MOVE;

    const TTEntry* tt_entry = TT.probe(info.pos.hash());
    if (tt_entry) {
        info.tb_hits++;
        tt_move = tt_entry->move;

        // Only use TT score for non-PV nodes and sufficient depth
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
    // Generate and order moves
    // -------------------------------------------------------------------------
    MoveList moves;
    MoveGen::generate_moves(info.pos, moves);

    order_moves(info.pos, moves, tt_move,
                info.killers[ss->ply],
                info.history[(i32)info.pos.side_to_move()]);

    // -------------------------------------------------------------------------
    // No legal moves — checkmate or stalemate
    // -------------------------------------------------------------------------
    i32 legal_moves = 0;
    Move best_move  = NULL_MOVE;
    i32  best_score = -INF;

    for (i32 i = 0; i < moves.size(); i++) {
        Move m = moves[i];

        UndoInfo undo = info.pos.make_move(m);

        // Skip illegal moves (leaves king in check)
        if (info.pos.is_attacked_by(
                info.pos.king_square(flip(info.pos.side_to_move())),
                info.pos.side_to_move())) {
            info.pos.unmake_move(m, undo);
            continue;
        }

        legal_moves++;
        info.rep_table.push(info.pos.hash());
        (ss + 1)->ply         = ss->ply + 1;
        (ss + 1)->current_move = m;

        // -------------------------------------------------------------------------
        // PVS — search first move with full window, rest with null window
        // -------------------------------------------------------------------------
        i32 score;

        if (legal_moves == 1) {
            // Full window search for first (best) move
            score = -search(info, ss + 1, depth - 1, -beta, -alpha);
        } else {
            // Null window search
            score = -search(info, ss + 1, depth - 1, -alpha - 1, -alpha);

            // Re-search with full window if the move might improve alpha on a PV node
            if (score > alpha && score < beta)
                score = -search(info, ss + 1, depth - 1, -beta, -alpha);
        }

        info.rep_table.pop();
        info.pos.unmake_move(m, undo);

        if (info.stop.load(std::memory_order_relaxed))
            return 0;

        // -------------------------------------------------------------------------
        // Update best score
        // -------------------------------------------------------------------------
        if (score > best_score) {
            best_score = score;
            best_move  = m;

            if (is_root) {
                info.best_move  = m;
                info.best_score = score;
            }
        }

        if (score > alpha)
            alpha = score;

        if (alpha >= beta) {
            // Beta cutoff — update heuristics for quiet moves
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
    // No legal moves found
    // -------------------------------------------------------------------------
    if (legal_moves == 0) {
        if (in_check)
            return mated_in(ss->ply);  // Checkmate
        return 0;                       // Stalemate
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
    info.reset();
    info.start_time = std::chrono::steady_clock::now();
    info.compute_allocated_time();

    // Push the current position hash into the repetition table
    info.rep_table.push(info.pos.hash());

    Move  best_move  = NULL_MOVE;
    i32   best_score = -INF;

    // Iterative deepening loop
    for (i32 depth = 1; depth <= info.limits.depth; depth++) {
        info.root_depth = depth;

        // Reset root stack entry
        info.stack[0].ply          = 0;
        info.stack[0].current_move = NULL_MOVE;

        i32 score = search(info, &info.stack[0], depth, -INF, INF);

        // If search was aborted, don't update best move
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
                  << " pv "    << best_move.to_string()
                  << "\n";

        // Check time after each completed iteration
        if (info.out_of_time())
            break;
    }

    // Pop the root position hash
    info.rep_table.pop();

    // Always return a move — fall back to first legal move if search failed
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
