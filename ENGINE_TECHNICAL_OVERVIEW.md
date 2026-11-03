# Bermuda Evaluation and Search: Technical Overview

This document describes the handcrafted evaluation and alpha-beta search
implemented in Bermuda's current source tree. It is intended as a guide to the
code, its score conventions, its heuristics, and the places where further
testing or tuning is appropriate. It describes Bermuda's own implementation;
it is not a transcription of another engine's source.

## Source map

| Area | Source |
|---|---|
| Evaluation parameters and public evaluation API | [`engine/eval.h`](./engine/eval.h) |
| Evaluation implementation | [`engine/eval.cpp`](./engine/eval.cpp) |
| Piece-square tables | [`engine/psqt.h`](./engine/psqt.h) |
| Search limits, state, time manager, mate scores | [`engine/search.h`](./engine/search.h) |
| Search, move ordering, SEE, pruning, iterative deepening | [`engine/search.cpp`](./engine/search.cpp) |
| Pseudo-legal move generation | [`engine/movegen.cpp`](./engine/movegen.cpp) |
| Position state and make/unmake operations | [`engine/position.h`](./engine/position.h) |
| Repetition and fifty-move draw checks | [`engine/repetition.h`](./engine/repetition.h) |
| UCI options and clock parsing | [`engine/uci.cpp`](./engine/uci.cpp), [`engine/uci.h`](./engine/uci.h) |
| Texel/SPSA evaluation tuner | [`tools/texel_tuner/`](./tools/texel_tuner/) |

## 1. Evaluation

### 1.1 Interface and score perspective

`Eval::evaluate(position)` returns an integer score in centipawns from the
**side-to-move's perspective**. A positive value favors the player to move; a
negative value favors the opponent. Search uses this convention with negamax:
after making a move, it negates the child score.

Internally, evaluation calculates separate middlegame (MG) and endgame (EG)
scores for White and Black. Each side's component scores are combined, White's
totals are subtracted from Black's, and the result is tapered according to the
remaining non-pawn material. The sign is then flipped when Black is to move.
A tempo bonus is added from the current side's perspective.

### 1.2 Tapered phase

The phase sums material-dependent weights for both players and is clamped to
the range 0–24:

| Piece | Phase weight |
|---|---:|
| Pawn | 0 |
| Knight | 1 |
| Bishop | 1 |
| Rook | 2 |
| Queen | 4 |
| King | 0 |

The result is mapped to an integer range of 0–128. The starting position
therefore has phase 128; a position without queens, rooks, or minor pieces
trends toward phase 0. The final score is blended as:

```text
blended = (middlegame_score * phase
         + endgame_score * (128 - phase)) / 128
```

The phase is based on the number of pieces remaining, not on the tuned piece
values. Material, piece-square, pawn, mobility, activity, threat, and king
terms contribute to MG and/or EG before blending.

### 1.3 Material

The built-in MG piece values are:

| Piece | MG value |
|---|---:|
| Pawn | 100 |
| Knight | 320 |
| Bishop | 335 |
| Rook | 500 |
| Queen | 930 |
| King | 0 |

The built-in EG values are 120, 310, 335, 510, 900, and 0 respectively.
Material for each piece type is counted by bitboard population. In addition
to the general `material` multiplier, each non-king piece can have separate
MG and EG value multipliers, such as `rook_mg` and `rook_eg`. Thus, for
example, a rook's MG material contribution is its count multiplied by its
built-in MG value, its `rook_mg` scale, and the global `material` scale.

The King is not assigned a material value; it is handled through piece-square,
activity, and king-safety terms.

Material also includes count-based interaction bonuses. The bishop pair is
valued according to the number of friendly pawns; two knights gain a smaller
pawn-dependent bonus; and two rooks are discounted in pawn-poor positions.
These terms are assigned to `material`, alongside a small bishop-versus-enemy-
minor-piece interaction.

### 1.4 Piece-square tables

[`engine/psqt.h`](./engine/psqt.h) contains separate MG and EG piece-square
tables for every piece type. Black's square index is vertically mirrored with
`square ^ 56`, so the same table can be used for both colors. The current
`piece_square` parameter scales the whole PSQT contribution; the individual
piece-value parameters do **not** change PSQT bonuses.

These are static positional bonuses. The evaluator does not currently learn a
separate multiplier for every square or every piece's PSQT.

### 1.5 Pawn structure

Pawn terms are evaluated independently for White and Black and then scaled by
`pawn_structure` or `passed_pawns`.

#### Structural terms

* **Isolated pawns:** a pawn with no friendly pawn on either adjacent file is
  penalized by 13 MG and 10 EG points. If an opposing pawn is ahead on its
  file, the penalties are reduced by 3 MG and 2 EG.
* **Supported or phalanx pawns:** a pawn supported by a friendly pawn, or
  adjacent to a friendly pawn on the same rank, receives a small advancement-
  dependent bonus. The phalanx condition adds an additional bonus.
* **Unsupported pawns:** a pawn with a neighboring friendly pawn but no pawn
  support or same-rank neighbor receives a modest penalty. The penalty grows
  when its immediately forward square is empty but controlled by an opposing
  pawn.
* **Pawn support:** a pawn attacked by another friendly pawn receives a small
  additional MG/EG bonus.
* **Doubled pawns:** every pawn beyond the first on a file is penalized by
  12 MG and 20 EG points.

The implementation uses bitboard file masks, adjacent-file masks, pawn attack
masks, and relative rank; these are geometric heuristics, not a full pawn
structure classifier.

#### Passed pawns

A pawn is treated as passed when there are no enemy pawns on its own or
adjacent files ahead of it. Passed-pawn bonuses increase with relative rank.
An unobstructed pawn also gets an advancement bonus and a king-distance term
that compares the kings' distance to the promotion and blocking squares. A
further bonus is awarded when the immediate block square is not attacked by
an enemy pawn. The full path to promotion also contributes according to enemy
blockers and pawn attacks, friendly pawn support, and friendly/enemy rook or
queen alignment behind the passer.

The passer arrays define the base MG bonuses by relative rank as
`0, 0, 8, 18, 35, 65, 110, 0` and the EG bonuses as
`0, 0, 12, 28, 55, 100, 180, 0`. Advancement, king distance, and block-square
bonuses are added separately.

### 1.6 Mobility

Knights, bishops, rooks, and queens receive mobility scores. Mobility counts
their attack squares after excluding:

* Squares controlled by enemy pawns.
* Certain blocked friendly-pawn squares.
* The friendly king square.
* Friendly occupied squares.

The piece-specific score is selected from a bounded table, with separate
curves for knights, bishops, rooks, and queens. MG and EG values differ slightly.
The entire mobility contribution is scaled by `mobility`.

### 1.7 Piece activity and positional features

The `activity` component currently includes:

* **Minor-piece outposts:** knights and bishops on relative ranks 3–5 that are
  outside the enemy-pawn span receive a bonus. Friendly pawn support increases
  the bonus; knights receive larger bonuses than bishops.
* **Minor piece behind a friendly pawn:** a small MG bonus is applied when a
  knight or bishop occupies the coded square behind a friendly pawn, subject
  to the relative-rank condition in `evaluate_pieces`.
* **Bishop color-complex congestion:** friendly pawns on the bishop's square
  color impose MG and EG penalties.
* **Reachable outposts:** a minor piece receives a smaller bonus if it can
  move to a safe outpost square outside the enemy pawn span; pawn-supported
  destination squares are preferred.
* **Rooks on open files:** a rook on a file without friendly pawns receives a
  bonus, larger if the file has no enemy pawns either.
* **Rooks on the seventh rank:** the bonus is larger when the enemy king or a
  pawn remains on its back rank.
* **Rooks attacking enemy pawns:** enemy pawns on the rook's attack ray
  contribute a small MG bonus.
* **Rook behind a passed pawn:** a rook on the same file behind a friendly
  passed pawn receives an MG/EG bonus.
* **Trapped-rook pattern:** a low-mobility rook beside its king on the home
  rank is penalized. The penalty depends on whether castling rights remain.

These bonuses are grouped under the `activity` scale, except mobility, which
has its own parameter.

### 1.8 Piece threats

For each side, the evaluator finds enemy knights, bishops, rooks, and queens
that are attacked by at least one friendly piece. It awards a type-dependent
threat bonus. The bonus is scaled based on the target's defense:

* No attack by any enemy piece on the target: threat scale 3.
* Target defended by an enemy pawn: threat scale 1.
* Otherwise: threat scale 2.

If the target is attacked at least twice and is not pawn-defended, an additional
bonus is applied. These are static threat incentives; they do not prove that a
piece can safely be won, because the evaluation does not search a capture
sequence itself. Search's separate SEE heuristic handles some tactical
exchange sequences. Safe pawn attacks on enemy pieces and king attacks on
undefended targets contribute additional threat bonuses.

The evaluator also penalizes friendly knights, bishops, rooks, and queens
whose squares are attacked by an enemy pawn, and more generally penalizes a
piece attacked by the opponent when it is not defended by any friendly
attack. The penalties are greater for rooks and queens. A queen attacked along
an enemy knight, bishop, or rook attack is also given a small MG penalty.

### 1.9 King safety and space

The king-safety component includes several approximations:

* **King-zone attack pressure:** the king zone is the king square, its adjacent
  squares, and the next rank in front. Enemy knights, bishops, rooks, and
  queens whose attacks intersect the zone count as attackers. With at least
  two attackers and a sufficiently high phase, a squared attacker-weight danger
  penalty is combined with undefended and multiply-attacked ring squares and
  available safe checking squares.
* **Pawn shelter:** in high-phase positions, friendly pawns on the next two
  ranks in front of the king earn bonuses. Missing shield pawns are penalized;
  an enemy pawn on a shield square adds another penalty.
* **Enemy attacks on the king flank:** when the side has a rook or queen and
  the phase is high, enemy attacked squares on the king's and adjacent files
  produce a capped MG penalty.
* **Pawnless king flank:** if no pawns remain on the king's file or adjacent
  files, the king receives a modest MG/EG safety penalty.
* **Space:** in higher-phase positions, safe central squares on the side's
  three homeward ranks contribute a bonus, with squares behind friendly pawns
  weighted again. The bonus is scaled by remaining friendly piece activity and
  reduced when the position has open files.

These terms are assigned to `king_safety` and tapered with the other MG/EG
components. Some are MG-only; therefore their final effect fades as the game
moves toward the endgame.

### 1.10 Insufficient material

Before calculating the other terms, evaluation returns zero if neither side
has pawns, rooks, or queens and the combined number of knights and bishops is
at most one. This handles bare kings and king-plus-single-minor versus bare
king. It is a deliberately small insufficient-material recognizer; it is not a
complete tablebase or draw adjudicator.

### 1.11 Tunable parameters

The parameter structure currently exposes 19 scales:

| Parameter group | Keys | Built-in value | Supported file range |
|---|---|---:|---:|
| Overall material | `material` | 1.0 | 0–3 |
| Piece-square tables | `piece_square` | 1.0 | 0–3 |
| Pawn structure | `pawn_structure` | 1.0 | 0–3 |
| Passed pawns | `passed_pawns` | 1.0 | 0–3 |
| Mobility | `mobility` | 1.0 | 0–3 |
| Piece activity | `activity` | 1.0 | 0–3 |
| Threats | `threats` | 1.0 | 0–3 |
| King safety and space | `king_safety` | 1.0 | 0–3 |
| Tempo | `tempo` | 1.0 | 0–3 |
| MG piece values | `pawn_mg` … `queen_mg` | 1.0 | 0.7–1.3 |
| EG piece values | `pawn_eg` … `queen_eg` | 1.0 | 0.7–1.3 |

The tuner writes text key/value parameter files. Bermuda loads them at runtime
through UCI:

```text
setoption name EvalParams value path-to-parameters.txt
```

An empty value resets the in-process evaluator to built-in parameters.
Malformed or out-of-range values are rejected and reported. The first nine
feature scales are required by the loader; per-piece MG/EG keys are optional
and default to 1.0, which preserves compatibility with earlier nine-key
parameter files.

The tuner library in `tools/texel_tuner/` uses SPSA updates to minimize
the mean squared error between the game's final result and a logistic
probability derived from Bermuda's static score. It reads PGNs and
`GAME_ID RESULT FEN` text datasets, or reusable versioned `.btd` datasets.
PGN tags and import statistics are retained; repeat `--metadata-filter
KEY=VALUE` to select exact tag-based source cohorts. Game-level split manifests
record membership and fingerprints. The objective supports the original
position weighting and equal-per-game weighting, while parameter families can
select positional terms, MG/EG piece values, or individual parameters.

The phase blend is 128 in the initial position (middlegame) and approaches 0
as non-pawn material disappears (endgame). In low-phase opposite-colored
bishop endings, the endgame score is reduced to reflect their drawish tendency.
A modest initiative correction uses pawn-file asymmetry, pawn-count difference,
and king proximity, capped to avoid changing the endgame score's sign.
In pawn-only endings, a passed pawn with an unobstructed route and a promotion
race favorable against the opposing king receives an additional EG bonus.

Runs write a human-readable JSON config and final report plus a resumable CSV
progress log. Binary datasets use explicit little-endian fields and FEN
positions rather than serializing C++ object memory; their FNV-1a checksum
detects accidental corruption but is not a cryptographic integrity guarantee.
The validation set is a small internal holdout; it does not replace independent
matches or a separate untouched test set. Tune focused parameter families
when possible and compare candidates in paired games before adopting them.

## 2. Search

### 2.1 Search model and scores

Bermuda uses iterative-deepening negamax alpha-beta search with principal
variation search (PVS). The maximum configured depth is 128 plies. A score
window `[alpha, beta)` represents the range in which an exact result is
needed. Fail-low and fail-high results are bounds, which the transposition
table records.

Mate scores are near `+30000` or `-30000` and include ply distance. The
`score_to_tt` and `score_from_tt` helpers adjust mate scores when storing and
retrieving them so that the distance remains meaningful at a different node.

Move generation returns pseudo-legal moves. Search makes each candidate move,
tests whether the mover's king is attacked, and unmakes illegal moves. This
legality check is used in the main search and quiescence search.

### 2.2 Iterative deepening and aspiration windows

`Search::start` searches depths 1, 2, 3, and onward, retaining the best move
from the last completed iteration. Depths 1–4 use a full window. Starting at
depth 5, the engine searches with a 25-centipawn aspiration window around the
previous iteration's score. If the score fails low or high, the relevant side
of the window is widened exponentially and the iteration is repeated.

Only a completed iteration is accepted as the current result. If time expires
mid-iteration, search aborts and returns the best move from the most recent
completed iteration. If no iteration completed, the engine scans generated
moves and returns the first legal one as a fallback.

At PV nodes, the principal variation table records the move that improved
alpha and copies the child's line upward. Search outputs completed depth,
score, node count, elapsed time, nodes per second, PV, and the final `bestmove`
in UCI format.

### 2.3 Principal variation search

The first legal move at a node is searched at full depth and with the full
current alpha-beta window. Later moves are generally searched with a null
window (`[-alpha-1, -alpha]`) to cheaply test whether they can improve the
current best. A move that beats alpha is re-searched at full depth if it was
reduced, and then with a full window when needed to establish an exact PV
score.

### 2.4 Move ordering

Moves are scored and sorted with insertion sort:

1. The transposition-table move is first.
2. Promotions are ordered immediately after it.
3. En-passant and captures with non-negative static exchange evaluations
   receive high scores; capture ordering uses MVV-LVA and SEE.
4. Captures with negative SEE receive a deliberately low ordering score.
5. Two killer quiet moves for the current ply follow.
6. Remaining quiet moves are ordered by history score.

The MVV-LVA table favors captures of more valuable victims and, within a
victim class, captures by less valuable attackers. SEE can lower the priority
of a capture that appears to lose material after a sequence of recaptures.
The SEE implementation builds local piece bitboards and occupancy, repeatedly
selects the least valuable attacker to the target, and backs up the resulting
gain sequence.

The history table is indexed by color, source square, and destination square.
Quiet moves causing beta cutoffs receive a depth-squared bonus, capped at
2048 before gravity adjustment; previously searched quiet moves that did not
cause the cutoff receive a malus. Entries are clamped to ±16384. Two killer
moves are retained per ply.

### 2.5 Transposition table

The global transposition table is keyed and verified with the position's
Zobrist hash. Entries contain the best move, score, search depth, bound type
(exact, lower, or upper), and search generation. A sufficiently deep entry
can cut off a non-PV node when its bound proves the result is outside the
current window. The table move is also used for move ordering. `Hash` changes
the table allocation; `Clear Hash` clears it.

### 2.6 Check extension and draw handling

Nodes where the side to move is in check receive a one-ply extension while the
search ply is below half of `MAX_PLY`. The extension allows the engine to
search the forced evasion before dropping to the nominal horizon.

The repetition table tracks position hashes and uses the halfmove clock to
limit backward scanning. Search treats a repeated position as a draw on the
first repeat encountered in the current line, which is more conservative than
waiting for an official threefold claim. The fifty-move condition is detected
at 100 halfmoves. These checks are applied in both main search and
quiescence; the root's current position is seeded before iterative deepening.

### 2.7 Pruning and reductions

Search uses several selective techniques to reduce the tree. Each can save
time, but each can also miss a line if its assumptions are wrong; tactical
regressions and match testing are important.

#### Static exchange evaluation

SEE orders captures and is also used for pruning:

* In quiescence, a non-promotion, non-en-passant capture with SEE below -80 is
  skipped when not in check.
* In the main search, a non-root, non-check capture (excluding promotions) is
  skipped at depths up to 6 if SEE is below `-35 * depth * depth`. Pruning is
  guarded so that at least one legal move is searched before such captures
  can be omitted.

SEE is a static exchange approximation, not a legal recursive search. It uses
least-valuable attackers and does not model every strategic consequence.

#### Futility pruning

At non-PV, non-check nodes up to depth 3, quiet moves can be omitted after at
least one legal move has been searched if the static evaluation plus the
depth-indexed margin cannot reach alpha. The margins are 90, 180, and 280
centipawns at depths 1, 2, and 3. Captures and promotions are excluded.

At depths up to 3, a separate reverse-futility-style check returns the static
evaluation at a non-root, non-PV node if `static_eval - margin >= beta`. This
requires the side to move to have non-pawn material and excludes check and
mate windows.

#### Razoring

At non-PV, non-check nodes of depth at most 3 without a TT move, quiescence is
searched against alpha reduced by a depth-specific margin (420, 520, or 620
centipawns). If that result still fails low, the node returns the quiescence
score instead of expanding the full move tree.

#### Null-move pruning

Null-move pruning is considered at non-PV, non-check nodes from depth 3 onward
when:

* Null moves are allowed at this ply.
* The side to move has at least one knight, bishop, rook, or queen.
* Static evaluation is at least beta.
* The node is not the root.

The reduction is `2 + depth / 4 + min((static_eval - beta) / 200, 3)`, with
non-negative surplus. After a null move and a null-window search, a score at
or above beta can produce a cutoff. Scores near mate are returned conservatively.
At depth 8 or greater, the engine performs a verification search with null
moves disabled at the current node before accepting the cutoff. Null-move
pruning is disabled immediately below the null move to prevent consecutive
null moves.

#### Late move reductions

Late quiet moves can be searched at reduced depth when the node is at least
depth 3, at least two legal moves have already been searched, neither the
current nor resulting position is in check, and the move is quiet. The
reduction table is initialized from:

```text
floor(ln(depth) * ln(move_count) / 2)
```

PV nodes reduce the value by one. A favorable history score reduces the
reduction; a poor one increases it. If the reduced search raises alpha, the
move is re-searched at full depth, and a possible PV improvement is then
searched with a full window.

### 2.8 Quiescence search

When the main search reaches depth zero, it enters quiescence search to extend
tactically unstable positions:

* Outside check, the evaluator supplies the stand-pat score. If it already
  reaches beta, quiescence returns; otherwise it raises alpha when appropriate.
* Outside check, it generates captures and queen promotions, orders them, and
  searches legal captures recursively.
* In check, stand pat is disabled and all pseudo-legal evasions are generated.
  Every evasion is legality-checked. If none are legal, the node is checkmate.
* Delta pruning skips a non-promotion capture when stand pat plus the victim's
  nominal value and a 200-centipawn margin cannot reach alpha. It is disabled
  in check and near mate scores.
* SEE pruning, repetition, fifty-move handling, and the maximum-ply guard are
  also applied.

The capture generator includes quiet queen promotions because they can be
decisive tactical continuations even though they are not captures. It does not
include quiet underpromotions in quiescence.

### 2.9 Maximum ply

`MAX_PLY` is 128. At the maximum ply, quiescence returns static evaluation
when not in check; when in check, it tries one legal evasion and evaluates the
resulting position. This bounds recursive stack use and prevents check
extensions or tactical sequences from searching indefinitely.

## 3. Time management and UCI controls

### 3.1 Limits accepted

The UCI `go` command reads depth, fixed `movetime`, each side's remaining
clock, increment, and `movestogo`. `infinite` and `ponder` disable the
internal time limits. With no time fields, search continues until externally
stopped or the depth limit is reached.

`MoveOverhead` defaults to 50 ms. Before allocation, UCI subtracts that
overhead from the reported clock and clamps positive clock values to at least
1 ms. For `movetime`, the overhead is subtracted and the result is likewise
clamped to at least 1 ms.

### 3.2 Allocation

For a clock-controlled search, Bermuda selects the remaining clock and
increment for the side to move. If `movestogo` is supplied, it uses that
number; otherwise it assumes 50 moves remain. It reserves the larger of 20 ms
or 2% of the remaining clock, capped at the entire clock. The base target is:

```text
base = time_remaining / (moves_to_go + 2)
     + 3 * max(increment, 0) / 4
```

The soft limit is the target, capped by the remaining time after reserve. The
hard limit is 1.5 times the target, capped by the same available time. If the
side's reported remaining clock is non-positive while a clock control is
active, both limits are set to 1 ms.

The soft limit is checked after an iteration completes, preventing the engine
from starting another iteration when it is already over budget. The hard
limit is checked every 128 nodes in both the main search and quiescence; if
exceeded, search marks itself stopped and unwinds to the last completed
iteration's move.

These are target budgets, not hard real-time guarantees. Operating-system
scheduling, concurrent games, output and protocol overhead, expensive nodes,
and the GUI's clock accounting can all affect actual elapsed time. The
`MoveOverhead` value should be calibrated on the same machine and concurrency
used for the tournament. A larger value makes the engine search less and
leaves more time for external delays; it does not make the evaluator or search
stronger.

## 4. Practical interpretation and limitations

1. **Evaluation is handcrafted and approximate.** Threat, king safety,
   outpost, pawn, rook, and space terms are heuristics. They do not replace
   search or prove a tactical win.
2. **The search is selective.** SEE pruning, futility pruning, razoring,
   null-move pruning, and LMR trade exhaustive coverage for speed. Validate
   changes with tactical positions, perft/legal-move checks, and games.
3. **Parameter fitting is not playing-strength proof.** Texel loss measures
   how well static scores predict supplied game results. Those labels are
   noisy and positions within a game are correlated. Use held-out games and
   paired match tests before accepting tuned weights.
4. **Ratings depend on the test conditions.** Opening set, time control,
   hardware, concurrency, hash size, engine versions, and opponent pool all
   affect measured rating. Ratings anchored to a fixed-depth engine should not
   be treated as directly interchangeable with a CCRL rating.
5. **Time management requires tournament-specific testing.** A UCI smoke test
   confirms that the engine returns moves; it cannot establish a low flagging
   rate in a parallel tournament. Collect PGNs and GUI logs, and compare
   remaining clock near timeouts before adjusting overhead or budgets.

## 5. Useful validation sequence

For evaluation or search changes, a repeatable validation loop is:

1. Build the engine and run the complete unit-test suite.
2. Run perft and tactical/search regression positions, including check,
   checkmate, promotions, repetition, and hanging-piece examples.
3. Use a fixed opening set and paired colors for candidate-versus-baseline
   matches. Keep hardware, concurrency, hash, and time control unchanged.
4. Record the executable version, parameter file, opening file, seed, match
   PGN, timeouts, and SPRT configuration.
5. Treat inconclusive match results as inconclusive. Keep the candidate
   separate until stronger evidence supports replacing the baseline.
