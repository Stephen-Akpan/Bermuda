#include "search.h"
#include "movegen.h"
#include <algorithm>
#include <iostream>
#include <cmath>

namespace Search {
// ============================================================================
// LMR Reduction Table
// ============================================================================
// Precomputed at startup: reductions[depth][move_count]
// Formula: ln(depth) * ln(move_count) / 2
// Clamped to [1, depth-1] at call site.

static i32 LMR_TABLE[MAX_PLY][64];

static void init_lmr_table() {
    for (i32 depth = 1; depth < MAX_PLY; depth++) {
        for (i32 mc = 1; mc < 64; mc++) {
            LMR_TABLE[depth][mc] = static_cast<i32>(
                std::log(depth) * std::log(mc) / 2.0);
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

static constexpr i32 FUTILITY_MARGIN[4] = { 0, 90, 180, 280 };
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

static constexpr i32 SEE_VALUE[6] = { 100, 320, 335, 500, 930, 20000 };

static Bitboard attackers_to(Square target, Colour colour,
                             const Bitboard pieces[2][6], Bitboard occupied) {
    Bitboard attackers = BB::pawn_attacks(BB::square_bb(target), flip(colour))
                       & pieces[colour][PAWN];
    attackers |= BB::knight_attack(target) & pieces[colour][KNIGHT];
    attackers |= BB::bishop_attack(target, occupied)
               & (pieces[colour][BISHOP] | pieces[colour][QUEEN]);
    attackers |= BB::rook_attack(target, occupied)
               & (pieces[colour][ROOK] | pieces[colour][QUEEN]);
    attackers |= BB::king_attack(target) & pieces[colour][KING];
    return attackers;
}

static i32 static_exchange(const Position& pos, Move move) {
    const Colour us = pos.side_to_move();
    const Square from = move.from();
    const Square target = move.to();
    const Piece moving = pos.piece_on(from);
    const Piece victim = move.is_en_passant()
        ? make_piece(flip(us), PAWN) : pos.piece_on(target);
    const Square capturedSquare = move.is_en_passant()
        ? static_cast<Square>(static_cast<i32>(target) - pawn_push_dir(us)) : target;

    Bitboard pieces[2][6];
    for (Colour colour : { WHITE, BLACK })
        for (PieceType type : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
            pieces[colour][type] = pos.pieces(colour, type);

    Bitboard occupied = pos.all_pieces();
    pieces[us][type_of(moving)] &= ~BB::square_bb(from);
    occupied &= ~BB::square_bb(from);
    if (victim != NO_PIECE) {
        pieces[flip(us)][type_of(victim)] &= ~BB::square_bb(capturedSquare);
        occupied &= ~BB::square_bb(capturedSquare);
    }

    PieceType occupant = type_of(moving);
    i32 initialGain = victim == NO_PIECE ? 0 : SEE_VALUE[type_of(victim)];
    if (move.is_promotion()) {
        const PieceType promoted = move.promo() == PROMO_QUEEN ? QUEEN
            : move.promo() == PROMO_ROOK ? ROOK
            : move.promo() == PROMO_BISHOP ? BISHOP : KNIGHT;
        initialGain += SEE_VALUE[promoted] - SEE_VALUE[PAWN];
        occupant = promoted;
    }
    pieces[us][occupant] |= BB::square_bb(target);
    occupied |= BB::square_bb(target);

    i32 gains[32] = {};
    gains[0] = initialGain;
    Colour side = flip(us);
    i32 depth = 0;

    while (depth < 30) {
        Bitboard attackers = attackers_to(target, side, pieces, occupied);
        if (!attackers)
            break;

        PieceType attackerType = PAWN;
        Square attackerSquare = NO_SQUARE;
        bool found = false;
        for (PieceType type : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING }) {
            const Bitboard candidates = attackers & pieces[side][type];
            if (candidates) {
                attackerType = type;
                attackerSquare = static_cast<Square>(lsb(candidates));
                found = true;
                break;
            }
        }
        if (!found)
            break;

        ++depth;
        gains[depth] = SEE_VALUE[occupant] - gains[depth - 1];
        if (std::max(-gains[depth - 1], gains[depth]) < 0)
            break;

        pieces[flip(side)][occupant] &= ~BB::square_bb(target);
        pieces[side][attackerType] &= ~BB::square_bb(attackerSquare);
        occupied &= ~BB::square_bb(attackerSquare);
        pieces[side][attackerType] |= BB::square_bb(target);
        occupant = attackerType;
        side = flip(side);
    }

    while (depth > 0) {
        gains[depth - 1] = -std::max(-gains[depth - 1], gains[depth]);
        --depth;
    }
    return gains[0];
}

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
        const i32 exchange = static_exchange(pos, m);
        const i32 value = MVV_LVA[type_of(attacker)][type_of(victim)];
        return exchange >= 0
            ? 800'000 + value * 100 + exchange
            : -100'000 + exchange;
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
static void update_history(SearchInfo& info, Move m, i32 depth, i32 direction) {
    i32 colour = (i32)info.pos.side_to_move();
    i32 bonus  = std::min(depth * depth * 16, 2048) * direction;

    i32& entry = info.history[colour][m.from()][m.to()];
    entry += bonus - entry * std::abs(bonus) / 16384;
    entry = std::max(-16384, std::min(entry, 16384));
}

// ============================================================================
// Quiescence Search
// ============================================================================

static i32 quiescence(SearchInfo& info, Stack* ss, i32 alpha, i32 beta) {
    info.nodes++;

    if ((info.nodes & 127) == 0) {
        if (info.past_hard_limit()) {
            info.stop.store(true);
        }
    }

    const bool in_check = info.pos.in_check();

    if (info.stop.load(std::memory_order_relaxed))
        return 0;

    if (info.rep_table.is_draw(info.pos.hash(), info.pos.halfmove()))
        return 0;

    if (ss->ply >= MAX_PLY - 1) {
        if (!in_check)
            return Eval::evaluate(info.pos);

        MoveList evasions;
        MoveGen::generate_moves(info.pos, evasions);
        i32 best_evasion = -INF;
        for (i32 i = 0; i < evasions.size(); ++i) {
            const Move move = evasions[i];
            const UndoInfo undo = info.pos.make_move(move);
            const bool legal = !info.pos.is_attacked_by(
                info.pos.king_square(flip(info.pos.side_to_move())),
                info.pos.side_to_move());
            if (legal)
                best_evasion = std::max(best_evasion, -Eval::evaluate(info.pos));
            info.pos.unmake_move(move, undo);
        }
        return best_evasion == -INF ? mated_in(ss->ply) : best_evasion;
    }

    i32 best_score = in_check ? -INF : Eval::evaluate(info.pos);

    if (!in_check) {
        if (best_score >= beta)
            return best_score;

        if (best_score > alpha)
            alpha = best_score;
    }

    MoveList moves;
    if (in_check)
        MoveGen::generate_moves(info.pos, moves);
    else
        MoveGen::generate_captures(info.pos, moves);

    order_moves(info.pos, moves, NULL_MOVE,
                info.killers[ss->ply],
                info.history[(i32)info.pos.side_to_move()]);

    i32 legal_moves = 0;
    for (i32 i = 0; i < moves.size(); i++) {
        Move m = moves[i];

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
        if (!in_check && !m.is_promotion() && !is_mate_score(alpha)) {
            Piece victim = m.is_en_passant()
                ? make_piece(flip(info.pos.side_to_move()), PAWN)
                : info.pos.piece_on(m.to());

            if (victim != NO_PIECE) {
                i32 gain = DELTA_PIECE_VALUE[type_of(victim)];
                if (best_score + gain + DELTA_MARGIN <= alpha)
                    continue;
            }
        }

        if (!in_check && !m.is_promotion() && !m.is_en_passant()
            && static_exchange(info.pos, m) < -80)
            continue;

        UndoInfo undo = info.pos.make_move(m);

        if (info.pos.is_attacked_by(
                info.pos.king_square(flip(info.pos.side_to_move())),
                info.pos.side_to_move())) {
            info.pos.unmake_move(m, undo);
            continue;
        }

        ++legal_moves;
        info.rep_table.push(info.pos.hash());
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

    if (in_check && legal_moves == 0)
        return mated_in(ss->ply);

    return best_score;
}

// ============================================================================
// Main Search
// ============================================================================

static i32 search(SearchInfo& info, Stack* ss, i32 depth, i32 alpha, i32 beta) {
    bool is_pv   = (beta - alpha) > 1;
    bool is_root = (ss->ply == 0);

    info.nodes++;

    if (ss->ply >= MAX_PLY - 1)
        return quiescence(info, ss, alpha, beta);

    // Initialise PV length for this ply — prevents stale moves from a previous
    // search iteration bleeding into the PV if this node fails to raise alpha.
    info.pv_length[ss->ply] = ss->ply;

    // -------------------------------------------------------------------------
    // Early exits
    // -------------------------------------------------------------------------

    if (!is_root) {
        if (info.rep_table.is_draw(info.pos.hash(), info.pos.halfmove()))
            return 0;
    }

    if (depth <= 0)
        return quiescence(info, ss, alpha, beta);

    // Periodic hard-limit check — abort mid-search if we've overrun
    if ((info.nodes & 127) == 0) {
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

    if (!in_check)
        ss->static_eval = Eval::evaluate(info.pos);

    const Colour us = info.pos.side_to_move();
    const bool has_non_pawn_material =
        (info.pos.pieces(us, KNIGHT) | info.pos.pieces(us, BISHOP)
       | info.pos.pieces(us, ROOK) | info.pos.pieces(us, QUEEN)) != 0;

    if (!is_pv && !is_root && !in_check && !is_mate_score(beta)
        && has_non_pawn_material && depth <= FUTILITY_MAX_DEPTH
        && ss->static_eval - FUTILITY_MARGIN[depth] >= beta)
        return ss->static_eval;

    if (!is_pv && !in_check && tt_move.is_null() && depth <= 3
        && !is_mate_score(alpha)) {
        static constexpr i32 RAZOR_MARGIN[4] = { 0, 420, 520, 620 };
        const i32 razorAlpha = alpha - RAZOR_MARGIN[depth];
        const i32 razorScore = quiescence(info, ss, razorAlpha, razorAlpha + 1);
        if (razorScore <= razorAlpha)
            return razorScore;
    }

    // -------------------------------------------------------------------------
    // Null Move Pruning
    // -------------------------------------------------------------------------
    if (!is_pv
        && !in_check
        && ss->null_move_allowed
        && depth >= 3
        && ss->ply >= 1
        && has_non_pawn_material
        && ss->static_eval >= beta)
    {
        const i32 surplus = std::max(0, ss->static_eval - beta);
        const i32 reduction = 2 + depth / 4 + std::min(surplus / 200, 3);

        const NullMoveUndo null_undo = info.pos.make_null_move();
        (ss + 1)->ply = ss->ply + 1;
        (ss + 1)->null_move_allowed = false;
        const i32 null_score = -search(
            info, ss + 1, depth - 1 - reduction, -beta, -beta + 1);
        info.pos.unmake_null_move(null_undo);
        (ss + 1)->null_move_allowed = true;

        if (info.stop.load(std::memory_order_relaxed))
            return 0;

        if (null_score >= beta) {
            if (null_score >= MATE_BOUND)
                return beta;

            if (depth < 8)
                return null_score;

            const bool nullMoveAllowed = ss->null_move_allowed;
            ss->null_move_allowed = false;
            const i32 verification = search(
                info, ss, depth - 1 - reduction, beta - 1, beta);
            ss->null_move_allowed = nullMoveAllowed;
            if (info.stop.load(std::memory_order_relaxed))
                return 0;
            if (verification >= beta)
                return null_score;
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

    i32  legal_moves = 0;
    Move searched_quiets[MoveList::MAX_MOVES];
    i32  searched_quiet_count = 0;
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

        const bool is_capture = m.is_en_passant()
            || info.pos.piece_on(m.to()) != NO_PIECE;
        if (!is_root && !in_check && legal_moves > 0
            && is_capture && !m.is_promotion()
            && depth <= 6 && !is_mate_score(alpha)
            && static_exchange(info.pos, m) < -35 * depth * depth)
            continue;

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
        if (is_quiet_move)
            searched_quiets[searched_quiet_count++] = m;

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
                       && !in_check
                       && !info.pos.in_check();

            i32 reduction = 0;

            if (do_lmr) {
                // Look up reduction from precomputed table.
                // Clamp move_count index to table bounds.
                i32 mc_idx = std::min(legal_moves, 63);
                i32 depth_idx = std::min(depth, MAX_PLY - 1);
                reduction  = LMR_TABLE[depth_idx][mc_idx];

                if (is_pv)
                    reduction = std::max(0, reduction - 1);

                const i32 move_history =
                    info.history[static_cast<i32>(flip(info.pos.side_to_move()))]
                                [m.from()][m.to()];
                if (move_history > 2048)
                    reduction = std::max(0, reduction - 1);
                else if (move_history < -1024)
                    ++reduction;

                // Never reduce to zero or below — that would be a quiescence
                // call, which is handled separately. Minimum reduced depth is 1.
                reduction = std::min(reduction, depth - 1);
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
                update_history(info, m, depth, 1);
            }
            for (i32 quiet = 0; quiet < searched_quiet_count; ++quiet)
                if (searched_quiets[quiet] != m)
                    update_history(info, searched_quiets[quiet], depth, -1);
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

    Eval::evaluate(info.pos); // computed_accumulation = 1

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
