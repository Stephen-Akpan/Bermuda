#include "eval.h"

#include "bitboard.h"
#include "psqt.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace Eval {
namespace {

constexpr i32 MAX_PHASE = 24;
constexpr i32 PHASE_WEIGHT[6] = { 0, 1, 1, 2, 4, 0 };
constexpr Score PIECE_MG[6] = { 100, 320, 335, 500, 930, 0 };
constexpr Score PIECE_EG[6] = { 120, 310, 335, 510, 900, 0 };
constexpr Score PASSED_MG[8] = { 0, 0, 8, 18, 35, 65, 110, 0 };
constexpr Score PASSED_EG[8] = { 0, 0, 12, 28, 55, 100, 180, 0 };
constexpr Score ISOLATED_MG = 13;
constexpr Score ISOLATED_EG = 10;
constexpr Score DOUBLED_MG = 12;
constexpr Score DOUBLED_EG = 20;

double piece_material_scale(const Parameters& parameters, PieceType type,
                           bool endgame) {
    if (endgame) {
        switch (type) {
        case PAWN: return parameters.pawn_eg;
        case KNIGHT: return parameters.knight_eg;
        case BISHOP: return parameters.bishop_eg;
        case ROOK: return parameters.rook_eg;
        case QUEEN: return parameters.queen_eg;
        default: return 1.0;
        }
    }
    switch (type) {
    case PAWN: return parameters.pawn_mg;
    case KNIGHT: return parameters.knight_mg;
    case BISHOP: return parameters.bishop_mg;
    case ROOK: return parameters.rook_mg;
    case QUEEN: return parameters.queen_mg;
    default: return 1.0;
    }
}

struct ScorePair {
    double mg = 0;
    double eg = 0;

    ScorePair& operator+=(ScorePair other) {
        mg += other.mg;
        eg += other.eg;
        return *this;
    }

    ScorePair& operator-=(ScorePair other) {
        mg -= other.mg;
        eg -= other.eg;
        return *this;
    }
};

ScorePair scaled(ScorePair score, double factor) {
    score.mg *= factor;
    score.eg *= factor;
    return score;
}

Eval::Parameters current_parameters;

struct AttackInfo {
    Bitboard by_type[6] = {};
    Bitboard all = 0;
    Bitboard twice = 0;
};

i32 relative_rank(Colour colour, Square sq) {
    return colour == WHITE ? rank(sq) : 7 - rank(sq);
}

i32 king_distance(Square a, Square b) {
    return std::max(std::abs(static_cast<i32>(file(a)) - static_cast<i32>(file(b))),
                    std::abs(static_cast<i32>(rank(a)) - static_cast<i32>(rank(b))));
}

Bitboard adjacent_files(Square sq) {
    Bitboard result = 0;
    if (file(sq) > 0)
        result |= BB::file_bb(make_square(file(sq) - 1, 0));
    if (file(sq) < 7)
        result |= BB::file_bb(make_square(file(sq) + 1, 0));
    return result;
}

Bitboard pawn_span(Colour colour, Square sq) {
    const Bitboard files = BB::file_bb(sq) | adjacent_files(sq);
    const i32 step = colour == WHITE ? 1 : -1;
    Bitboard span = 0;
    for (i32 r = static_cast<i32>(rank(sq)) + step; r >= 0 && r < 8; r += step)
        span |= (BB::RANK_1 << (r * 8)) & files;
    return span;
}

Bitboard pawn_path(Colour colour, Square sq) {
    Bitboard path = 0;
    const i32 step = pawn_push_dir(colour);
    for (i32 index = static_cast<i32>(sq) + step;
         index >= 0 && index < 64;
         index += step) {
        const Square target = static_cast<Square>(index);
        path |= BB::square_bb(target);
        if (relative_rank(colour, target) == 7)
            break;
    }
    return path;
}

ScorePair piece_square_score(PieceType type, Colour colour, Square square) {
    const i32 index = colour == WHITE ? square : (square ^ 56);
    const Score* mg = nullptr;
    const Score* eg = nullptr;

    switch (type) {
    case PAWN:   mg = PSQT::pawn_mg;   eg = PSQT::pawn_eg;   break;
    case KNIGHT: mg = PSQT::knight_mg; eg = PSQT::knight_eg; break;
    case BISHOP: mg = PSQT::bishop_mg; eg = PSQT::bishop_eg; break;
    case ROOK:   mg = PSQT::rook_mg;   eg = PSQT::rook_eg;   break;
    case QUEEN:  mg = PSQT::queen_mg;  eg = PSQT::queen_eg;  break;
    case KING:   mg = PSQT::king_mg;   eg = PSQT::king_eg;   break;
    default: return {};
    }
    return { static_cast<double>(mg[index]), static_cast<double>(eg[index]) };
}

Bitboard attacks_from(PieceType type, Square square, Colour colour, Bitboard occupied) {
    switch (type) {
    case PAWN:   return BB::pawn_attacks(BB::square_bb(square), colour);
    case KNIGHT: return BB::knight_attack(square);
    case BISHOP: return BB::bishop_attack(square, occupied);
    case ROOK:   return BB::rook_attack(square, occupied);
    case QUEEN:  return BB::queen_attack(square, occupied);
    case KING:   return BB::king_attack(square);
    default:     return 0;
    }
}

AttackInfo build_attacks(const Position& pos, Colour colour) {
    AttackInfo info;
    const Bitboard occupied = pos.all_pieces();
    const Bitboard pawns = pos.pieces(colour, PAWN);
    info.by_type[PAWN] = BB::pawn_attacks(pawns, colour);
    info.all = info.by_type[PAWN];

    for (PieceType type : { KNIGHT, BISHOP, ROOK, QUEEN, KING }) {
        Bitboard pieces = pos.pieces(colour, type);
        while (pieces) {
            const Square square = static_cast<Square>(pop_lsb(pieces));
            const Bitboard attacks = attacks_from(type, square, colour, occupied);
            info.twice |= info.all & attacks;
            info.by_type[type] |= attacks;
            info.all |= attacks;
        }
    }
    return info;
}

ScorePair mobility_score(PieceType type, i32 mobility) {
    static constexpr Score KNIGHT_MOBILITY[] = {
        -20, -10, 0, 9, 17, 24, 30, 35, 39
    };
    static constexpr Score BISHOP_MOBILITY[] = {
        -18, -10, -2, 6, 14, 21, 28, 34, 40, 45, 49, 53, 56, 59
    };
    static constexpr Score ROOK_MOBILITY[] = {
        -12, -5, 2, 9, 16, 22, 28, 34, 39, 44, 48, 52, 55, 58, 60
    };
    static constexpr Score QUEEN_MOBILITY[] = {
        -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44,
        48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96, 100
    };
    const Score* values = nullptr;
    i32 count = 0;
    switch (type) {
    case KNIGHT:
        values = KNIGHT_MOBILITY;
        count = static_cast<i32>(sizeof(KNIGHT_MOBILITY) / sizeof(*KNIGHT_MOBILITY));
        break;
    case BISHOP:
        values = BISHOP_MOBILITY;
        count = static_cast<i32>(sizeof(BISHOP_MOBILITY) / sizeof(*BISHOP_MOBILITY));
        break;
    case ROOK:
        values = ROOK_MOBILITY;
        count = static_cast<i32>(sizeof(ROOK_MOBILITY) / sizeof(*ROOK_MOBILITY));
        break;
    case QUEEN:
        values = QUEEN_MOBILITY;
        count = static_cast<i32>(sizeof(QUEEN_MOBILITY) / sizeof(*QUEEN_MOBILITY));
        break;
    default: return {};
    }

    const Score value = values[std::min(mobility, count - 1)];
    const Score endgameAdjustment = type == KNIGHT ? -2
        : type == BISHOP ? 2 : type == ROOK ? 4 : 6;
    return { static_cast<double>(value),
             static_cast<double>(value + endgameAdjustment) };
}

ScorePair material_imbalance(const Position& pos, Colour colour) {
    const Colour enemy = flip(colour);
    const i32 bishops = popcount(pos.pieces(colour, BISHOP));
    const i32 knights = popcount(pos.pieces(colour, KNIGHT));
    const i32 rooks = popcount(pos.pieces(colour, ROOK));
    const i32 pawns = popcount(pos.pieces(colour, PAWN));
    const i32 enemyMinors = popcount(pos.pieces(enemy, KNIGHT)
                                   | pos.pieces(enemy, BISHOP));

    ScorePair score;
    if (bishops >= 2) {
        score.mg += 22 + std::min(pawns, 8) * 2;
        score.eg += 28 + std::min(pawns, 8) * 3;
    }
    if (knights >= 2) {
        score.mg += std::min(pawns, 8) * 2;
        score.eg += std::min(pawns, 8) * 3;
    }
    if (rooks >= 2) {
        const i32 lackOfPawns = std::max(0, 6 - pawns);
        score.mg -= lackOfPawns * 2;
        score.eg -= lackOfPawns * 3;
    }
    score.mg += std::min(bishops, 2) * enemyMinors * 2;
    score.eg += std::min(bishops, 2) * enemyMinors;
    return score;
}

void evaluate_pawns(const Position& pos, Colour colour, ScorePair& structure,
                    ScorePair& passed) {
    const Colour enemy = flip(colour);
    const Bitboard own = pos.pieces(colour, PAWN);
    const Bitboard opposing = pos.pieces(enemy, PAWN);
    const Bitboard enemyAttacks = BB::pawn_attacks(opposing, enemy);
    const bool pawnOnlyEnding = !(pos.pieces(KNIGHT) | pos.pieces(BISHOP)
                                | pos.pieces(ROOK) | pos.pieces(QUEEN));
    Bitboard pawns = own;

    while (pawns) {
        const Square square = static_cast<Square>(pop_lsb(pawns));
        const i32 advance = relative_rank(colour, square);
        const Bitboard fileMask = BB::file_bb(square);
        const Bitboard neighbours = own & adjacent_files(square);
        const bool opposed = (opposing & fileMask & pawn_span(colour, square)) != 0;

        if (!neighbours) {
            structure.mg -= ISOLATED_MG - (opposed ? 3 : 0);
            structure.eg -= ISOLATED_EG - (opposed ? 2 : 0);
        } else {
            const Bitboard supportedByPawn = BB::pawn_attacks(own, colour);
            const bool supported = (supportedByPawn & BB::square_bb(square)) != 0;
            const bool phalanx = (neighbours & BB::rank_bb(square)) != 0;
            if (supported || phalanx) {
                const Score bonus = 4 + advance * 2 + (phalanx ? 3 : 0);
                structure.mg += bonus;
                structure.eg += bonus + advance;
            } else {
                const i32 push = static_cast<i32>(square) + pawn_push_dir(colour);
                if (push >= 0 && push < 64
                    && pos.empty(static_cast<Square>(push))
                    && (enemyAttacks & BB::square_bb(static_cast<Square>(push)))) {
                    structure.mg -= opposed ? 9 : 13;
                    structure.eg -= opposed ? 5 : 8;
                } else {
                    structure.mg -= 5;
                    structure.eg -= 3;
                }
            }
        }

        const Bitboard forward = pawn_span(colour, square) | fileMask;
        if (!(opposing & forward)) {
            const Score rankBonus = PASSED_MG[advance];
            const Score endBonus = PASSED_EG[advance];
            passed.mg += rankBonus;
            passed.eg += endBonus;

            const i32 blockIndex = static_cast<i32>(square) + pawn_push_dir(colour);
            if (blockIndex >= 0 && blockIndex < 64) {
                const Square block = static_cast<Square>(blockIndex);
                const Bitboard path = pawn_path(colour, square);
                const Bitboard enemyBlockers = path & pos.pieces(enemy);
                const Bitboard ownBlockers = path & pos.pieces(colour);
                const Bitboard enemyControlled = path & enemyAttacks;
                const Bitboard ownControlled = path
                    & BB::pawn_attacks(own, colour);

                passed.mg -= popcount(enemyBlockers) * (3 + advance * 2);
                passed.eg -= popcount(enemyBlockers) * (5 + advance * 3);

                if (pos.empty(block)) {
                    passed.mg += advance * 2;
                    passed.eg += advance * 5;

                    const Square enemyKing = pos.king_square(enemy);
                    const Square ownKing = pos.king_square(colour);
                    const i32 promoIndex = static_cast<i32>(square)
                        + pawn_push_dir(colour) * (7 - advance);
                    const Square promotion = static_cast<Square>(promoIndex);
                    passed.eg += (king_distance(enemyKing, promotion)
                                - king_distance(ownKing, block)) * 3;
                    passed.mg += std::max(0, popcount(path & ~enemyControlled)
                                             - popcount(enemyControlled)) * (1 + advance / 2);
                    passed.eg += std::max(0, popcount(path & ~enemyControlled)
                                             - popcount(enemyControlled)) * (2 + advance);
                    passed.mg += popcount(ownControlled) * 2;
                    passed.eg += popcount(ownControlled) * 4;

                    const Bitboard ownMajors = pos.pieces(colour, ROOK)
                                             | pos.pieces(colour, QUEEN);
                    const Bitboard enemyMajors = pos.pieces(enemy, ROOK)
                                               | pos.pieces(enemy, QUEEN);
                    const Bitboard rookRay = BB::rook_attack(square, pos.all_pieces());
                    if (ownMajors & rookRay) {
                        passed.mg += 8;
                        passed.eg += 18;
                    }
                    if (enemyMajors & rookRay) {
                        passed.mg -= 6;
                        passed.eg -= 14;
                    }

                    if (pawnOnlyEnding && !(path & pos.all_pieces())) {
                        const i32 movesToPromote = 7 - advance
                            + (pos.side_to_move() == colour ? 0 : 1);
                        if (king_distance(enemyKing, promotion) > movesToPromote) {
                            passed.eg += 35 + advance * 10;
                            if (king_distance(ownKing, promotion)
                                < king_distance(enemyKing, promotion))
                                passed.eg += 15;
                        }
                    }
                }
                else if (ownBlockers & BB::square_bb(block)) {
                    passed.mg += advance;
                    passed.eg += advance * 2;
                }
            }
        }

        const Bitboard behind = BB::pawn_attacks(own, colour) & BB::square_bb(square);
        if (behind) {
            structure.mg += 3;
            structure.eg += 5;
        }
    }

    for (ui8 f = 0; f < 8; ++f) {
        Bitboard pawnsOnFile = own & (BB::FILE_A << f);
        if (popcount(pawnsOnFile) > 1) {
            const i32 extras = popcount(pawnsOnFile) - 1;
            structure.mg -= extras * DOUBLED_MG;
            structure.eg -= extras * DOUBLED_EG;
        }
    }
}

ScorePair evaluate_pieces(const Position& pos, Colour colour,
                          const AttackInfo& ownAttacks,
                          const AttackInfo& enemyAttacks,
                          i32 phase, ScorePair& mobilityScore,
                          ScorePair& threatScore, ScorePair& kingSafety) {
    ScorePair score;
    const Colour enemy = flip(colour);
    const Bitboard occupied = pos.all_pieces();
    const Bitboard own = pos.pieces(colour);
    const Bitboard enemyPawns = pos.pieces(enemy, PAWN);
    Bitboard enemyPawnSpan = 0;
    Bitboard enemyPawnsForSpan = enemyPawns;
    while (enemyPawnsForSpan) {
        const Square pawn = static_cast<Square>(pop_lsb(enemyPawnsForSpan));
        enemyPawnSpan |= pawn_span(enemy, pawn);
    }
    const Bitboard blockedPawns = colour == WHITE
        ? pos.pieces(colour, PAWN) & (BB::shift(SOUTH, occupied) | BB::RANK_2 | BB::RANK_3)
        : pos.pieces(colour, PAWN) & (BB::shift(NORTH, occupied) | BB::RANK_7 | BB::RANK_6);
    const Bitboard mobilityArea = ~(enemyAttacks.by_type[PAWN] | blockedPawns
                                  | BB::square_bb(pos.king_square(colour)));
    const Bitboard outpostRanks = colour == WHITE
        ? BB::RANK_4 | BB::RANK_5 | BB::RANK_6
        : BB::RANK_3 | BB::RANK_4 | BB::RANK_5;

    for (PieceType type : { KNIGHT, BISHOP, ROOK, QUEEN }) {
        Bitboard pieces = pos.pieces(colour, type);
        while (pieces) {
            const Square square = static_cast<Square>(pop_lsb(pieces));
            Bitboard attacks = attacks_from(type, square, colour, occupied);
            const i32 mobility = popcount(attacks & mobilityArea & ~own);
            mobilityScore += mobility_score(type, mobility);

            if (type == KNIGHT || type == BISHOP) {
                const i32 rr = relative_rank(colour, square);
                const bool supported = (ownAttacks.by_type[PAWN]
                                      & BB::square_bb(square)) != 0;
                const bool onOutpost = (outpostRanks & BB::square_bb(square))
                    && !(enemyPawnSpan & BB::square_bb(square));
                if (onOutpost) {
                    const Score bonus = type == KNIGHT ? (supported ? 38 : 25)
                                                       : (supported ? 20 : 12);
                    score.mg += bonus;
                    score.eg += bonus / 4;
                } else {
                    const Bitboard reachableOutposts = attacks & outpostRanks
                        & ~enemyPawnSpan & ~occupied
                        & ~enemyAttacks.by_type[PAWN];
                    if (reachableOutposts) {
                        const Bitboard supportedOutposts = reachableOutposts
                            & ownAttacks.by_type[PAWN];
                        const Score bonus = type == KNIGHT
                            ? (supportedOutposts ? 16 : 9)
                            : (supportedOutposts ? 8 : 4);
                        score.mg += bonus;
                        score.eg += bonus / 4;
                    }
                }

                if (rr <= 4) {
                    const i32 behindPawn = static_cast<i32>(square)
                        + pawn_push_dir(enemy);
                    if (behindPawn >= 0 && behindPawn < 64
                        && pos.piece_on(static_cast<Square>(behindPawn))
                           == make_piece(colour, PAWN)) {
                        score.mg += 12;
                    }
                }

                if (type == BISHOP) {
                    const Bitboard sameColorPawns = ((file(square) + rank(square)) & 1)
                        ? BB::LIGHT_SQUARES : BB::DARK_SQUARES;
                    const i32 blockers = popcount(pos.pieces(colour, PAWN) & sameColorPawns);
                    score.mg -= blockers * 3;
                    score.eg -= blockers * 5;
                }
            }

            if (type == ROOK) {
                const Bitboard fileMask = BB::file_bb(square);
                if (!(pos.pieces(colour, PAWN) & fileMask)) {
                    const Score bonus = (pos.pieces(enemy, PAWN) & fileMask) ? 14 : 27;
                    score.mg += bonus;
                    score.eg += bonus / 2;
                }
                if (relative_rank(colour, square) == 6) {
                    const Bitboard enemyBackRank = colour == WHITE
                        ? BB::RANK_8 : BB::RANK_1;
                    if ((pos.pieces(enemy, KING) | pos.pieces(enemy, PAWN))
                        & enemyBackRank) {
                        score.mg += 18;
                        score.eg += 30;
                    } else {
                        score.mg += 8;
                        score.eg += 14;
                    }
                }

                Bitboard ownPawns = pos.pieces(colour, PAWN);
                while (ownPawns) {
                    const Square pawn = static_cast<Square>(pop_lsb(ownPawns));
                    if (file(pawn) == file(square)
                        && relative_rank(colour, square) < relative_rank(colour, pawn)
                        && !(pos.pieces(enemy, PAWN)
                             & pawn_span(colour, pawn))) {
                        score.mg += 8;
                        score.eg += 16;
                        break;
                    }
                }
                if (relative_rank(colour, square) >= 4) {
                    Bitboard advancedEnemyPawns = pos.pieces(enemy, PAWN)
                        & BB::rook_attack(square, occupied);
                    while (advancedEnemyPawns) {
                        const Square pawn = static_cast<Square>(
                            pop_lsb(advancedEnemyPawns));
                        if (relative_rank(enemy, pawn) >= 3)
                            score.mg += 4;
                    }
                }
                const Square king = pos.king_square(colour);
                if (relative_rank(colour, square) == 0
                    && relative_rank(colour, king) == 0
                    && std::abs(static_cast<i32>(file(square))
                              - static_cast<i32>(file(king))) == 1
                    && mobility <= 3) {
                    const i32 pawnAhead = static_cast<i32>(square)
                        + pawn_push_dir(colour);
                    const bool blockedByPawn = pawnAhead >= 0 && pawnAhead < 64
                        && pos.piece_on(static_cast<Square>(pawnAhead))
                           == make_piece(colour, PAWN);
                    const bool canCastle = pos.castling_rights() & (colour == WHITE
                        ? (WHITE_OO | WHITE_OOO) : (BLACK_OO | BLACK_OOO));
                    score.mg -= (canCastle ? 25 : 45) + (blockedByPawn ? 12 : 0);
                }
            }

            const Bitboard squareBB = BB::square_bb(square);
            if (enemyAttacks.by_type[PAWN] & squareBB) {
                const Score penalty = type == QUEEN ? 55 : type == ROOK ? 42 : 32;
                score.mg -= penalty;
                score.eg -= penalty / 3;
            }

            if ((enemyAttacks.all & squareBB) && !(ownAttacks.all & squareBB)) {
                const Score penalty = type == QUEEN ? 85 : type == ROOK ? 65 : 48;
                score.mg -= penalty;
                score.eg -= penalty / 2;
            }

            if (type == QUEEN
                && (attacks & (enemyAttacks.by_type[KNIGHT] | enemyAttacks.by_type[BISHOP]
                             | enemyAttacks.by_type[ROOK]))) {
                score.mg -= 12;
            }

        }
    }

    constexpr Score THREAT_MG[6] = { 0, 0, 25, 25, 40, 48 };
    constexpr Score THREAT_EG[6] = { 0, 0, 25, 28, 38, 42 };
    for (PieceType type : { KNIGHT, BISHOP, ROOK, QUEEN }) {
        Bitboard targets = pos.pieces(enemy, type) & ownAttacks.all;
        while (targets) {
            const Square target = static_cast<Square>(pop_lsb(targets));
            const Bitboard targetBB = BB::square_bb(target);
            const bool defendedByPawn = (enemyAttacks.by_type[PAWN] & targetBB) != 0;
            const bool isHanging = !(enemyAttacks.all & targetBB);
            const Score threatScale = isHanging ? 3 : defendedByPawn ? 1 : 2;
            threatScore.mg += THREAT_MG[type] * threatScale / 2;
            threatScore.eg += THREAT_EG[type] * threatScale / 2;
            if ((ownAttacks.twice & targetBB) && !defendedByPawn) {
                threatScore.mg += 12;
                threatScore.eg += 16;
            }
        }
    }

    Bitboard pawnThreats = (pos.pieces(enemy, KNIGHT) | pos.pieces(enemy, BISHOP)
                          | pos.pieces(enemy, ROOK) | pos.pieces(enemy, QUEEN))
                         & ownAttacks.by_type[PAWN]
                         & (~enemyAttacks.all | ownAttacks.all);
    while (pawnThreats) {
        const PieceType target = type_of(pos.piece_on(
            static_cast<Square>(pop_lsb(pawnThreats))));
        const Score bonus = target == QUEEN ? 30 : target == ROOK ? 24 : 18;
        threatScore.mg += bonus;
        threatScore.eg += bonus;
    }

    const Bitboard undefendedTargets = pos.pieces(enemy)
        & ~(enemyAttacks.all | enemyAttacks.by_type[PAWN])
        & ownAttacks.by_type[KING];
    threatScore.mg += popcount(undefendedTargets) * 3;
    threatScore.eg += popcount(undefendedTargets) * 8;

    const Square king = pos.king_square(colour);
    const Bitboard kingZone = BB::king_attack(king)
        | BB::square_bb(king)
        | BB::shift(colour == WHITE ? NORTH : SOUTH, BB::king_attack(king));

    i32 attackers = 0;
    i32 attackWeight = 0;
    for (PieceType type : { KNIGHT, BISHOP, ROOK, QUEEN }) {
        Bitboard enemyPieces = pos.pieces(enemy, type);
        while (enemyPieces) {
            const Square attacker = static_cast<Square>(pop_lsb(enemyPieces));
            const Bitboard attacks = attacks_from(type, attacker, enemy, occupied);
            if (attacks & kingZone) {
                ++attackers;
                attackWeight += type == QUEEN ? 5 : type == ROOK ? 4 : 2;
            }
        }
    }

    if (attackers >= 2 && phase > 16) {
        const Bitboard undefendedRing = kingZone
            & enemyAttacks.all & ~ownAttacks.all;
        const Bitboard multiplyAttackedRing = kingZone
            & enemyAttacks.twice & ~ownAttacks.by_type[PAWN];

        const Bitboard rookChecks = BB::rook_attack(king, occupied)
            & (enemyAttacks.by_type[ROOK] | enemyAttacks.by_type[QUEEN]);
        const Bitboard bishopChecks = BB::bishop_attack(king, occupied)
            & (enemyAttacks.by_type[BISHOP] | enemyAttacks.by_type[QUEEN]);
        const Bitboard knightChecks = BB::knight_attack(king)
            & enemyAttacks.by_type[KNIGHT];
        const Bitboard safeChecks = (rookChecks | bishopChecks | knightChecks)
            & ~ownAttacks.all & ~occupied;

        const i32 danger = std::min(
            attackWeight * attackWeight * 2
                + popcount(undefendedRing) * 8
                + popcount(multiplyAttackedRing) * 12
                + popcount(safeChecks) * 10,
            320);
        kingSafety.mg -= danger;
    }

    const i32 kingFile = file(king);
    const i32 forward = colour == WHITE ? 1 : -1;
    if (phase > 20) {
        for (i32 df = -1; df <= 1; ++df) {
            const i32 f = kingFile + df;
            const i32 shieldRank = static_cast<i32>(rank(king)) + forward;
            if (f < 0 || f >= 8 || shieldRank < 0 || shieldRank >= 8)
                continue;
            const Square shield = make_square(f, static_cast<ui8>(shieldRank));
            const i32 nextRank = shieldRank + forward;
            const Piece shieldPiece = pos.piece_on(shield);
            if (shieldPiece == make_piece(colour, PAWN))
                kingSafety.mg += 12;
            else {
                kingSafety.mg -= 8;
                if (shieldPiece == make_piece(enemy, PAWN))
                    kingSafety.mg -= 5;
            }

            if (nextRank >= 0 && nextRank < 8) {
                const Piece nextPiece = pos.piece_on(
                    make_square(f, static_cast<ui8>(nextRank)));
                if (nextPiece == make_piece(colour, PAWN))
                    kingSafety.mg += 5;
                else if (nextPiece == make_piece(enemy, PAWN))
                    kingSafety.mg -= 7;
            }
        }
    }

    const Bitboard ourMajors = pos.pieces(colour, ROOK) | pos.pieces(colour, QUEEN);
    if (ourMajors && (phase > 20)) {
        const Bitboard kingFlank = BB::file_bb(king)
            | adjacent_files(king);
        const Bitboard enemyNearKing = enemyAttacks.all & kingFlank;
        kingSafety.mg -= std::min(popcount(enemyNearKing) * 2, 24);
    }

    const i32 flankFile = file(king);
    Bitboard kingFiles = BB::file_bb(king);
    if (flankFile > 0)
        kingFiles |= BB::FILE_A << (flankFile - 1);
    if (flankFile < 7)
        kingFiles |= BB::FILE_A << (flankFile + 1);
    if (!(pos.pieces(PAWN) & kingFiles)) {
        kingSafety.mg -= 18;
        kingSafety.eg -= 48;
    }

    if (phase > 80) {
        const Bitboard centralRanks = colour == WHITE
            ? BB::RANK_2 | BB::RANK_3 | BB::RANK_4
            : BB::RANK_7 | BB::RANK_6 | BB::RANK_5;
        const Bitboard spaceMask = (BB::FILE_C | BB::FILE_D | BB::FILE_E | BB::FILE_F)
                                 & centralRanks;
        const Bitboard ownPawns = pos.pieces(colour, PAWN);
        const Bitboard safe = spaceMask & ~ownPawns
            & ~enemyAttacks.by_type[PAWN]
            & (ownAttacks.all | ~enemyAttacks.all);
        Bitboard behind = ownPawns;
        const Direction backward = colour == WHITE ? SOUTH : NORTH;
        behind |= BB::shift(backward, ownPawns);
        behind |= BB::shift(backward, BB::shift(backward, ownPawns));
        behind |= BB::shift(backward,
                            BB::shift(backward, BB::shift(backward, ownPawns)));
        const i32 safeSpace = std::min(
            16, popcount(safe) + popcount(safe & behind));

        i32 openFiles = 0;
        for (ui8 f = 0; f < 8; ++f)
            openFiles += !(pos.pieces(PAWN) & (BB::FILE_A << f));
        const i32 spaceWeight = std::max(0, popcount(own) - 2 * openFiles);
        kingSafety.mg += safeSpace * spaceWeight * spaceWeight / 18;
    }

    return score;
}

i32 game_phase(const Position& pos) {
    i32 phase = 0;
    for (Colour colour : { WHITE, BLACK })
        for (PieceType type : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING })
            phase += popcount(pos.pieces(colour, type)) * PHASE_WEIGHT[type];
    return std::max(0, std::min(MAX_PHASE, phase)) * 128 / MAX_PHASE;
}

double endgame_scale(const Position& pos, i32 phase) {
    if (phase >= 48)
        return 1.0;

    const Bitboard whiteBishops = pos.pieces(WHITE, BISHOP);
    const Bitboard blackBishops = pos.pieces(BLACK, BISHOP);
    if (popcount(whiteBishops) != 1 || popcount(blackBishops) != 1)
        return 1.0;

    const Square whiteBishop = static_cast<Square>(lsb(whiteBishops));
    const Square blackBishop = static_cast<Square>(lsb(blackBishops));
    if (((file(whiteBishop) + rank(whiteBishop)) & 1)
        == ((file(blackBishop) + rank(blackBishop)) & 1))
        return 1.0;

    const Bitboard nonBishops = pos.pieces(KNIGHT) | pos.pieces(ROOK)
                              | pos.pieces(QUEEN);
    if (nonBishops)
        return 0.85;

    const i32 pawnCount = popcount(pos.pieces(PAWN));
    return pawnCount <= 1 ? 0.40 : 0.60;
}

double endgame_initiative(const Position& pos, double endgameScore) {
    if (endgameScore == 0.0)
        return 0.0;

    i32 asymmetricFiles = 0;
    for (ui8 f = 0; f < 8; ++f) {
        const Bitboard fileMask = BB::FILE_A << f;
        if (!!(pos.pieces(WHITE, PAWN) & fileMask)
            != !!(pos.pieces(BLACK, PAWN) & fileMask))
            ++asymmetricFiles;
    }

    const i32 kingDistance = king_distance(pos.king_square(WHITE),
                                           pos.king_square(BLACK));
    const i32 initiative = asymmetricFiles * 6
        + std::abs(popcount(pos.pieces(WHITE, PAWN))
                   - popcount(pos.pieces(BLACK, PAWN))) * 3
        + std::max(0, 4 - kingDistance) * 2;
    return (endgameScore > 0.0 ? 1.0 : -1.0)
        * std::min<double>(initiative, std::abs(endgameScore) / 2.0);
}

bool insufficient_material(const Position& pos) {
    if (pos.pieces(PAWN) || pos.pieces(ROOK) || pos.pieces(QUEEN))
        return false;
    return popcount(pos.pieces(KNIGHT) | pos.pieces(BISHOP)) <= 1;
}

} // namespace

Score evaluate(const Position& pos) {
    return evaluate(pos, current_parameters);
}

Score evaluate(const Position& pos, const Parameters& parameters) {
    if (insufficient_material(pos))
        return 0;

    const i32 phase = game_phase(pos);
    ScorePair sideScore[2];
    const AttackInfo attacks[2] = {
        build_attacks(pos, WHITE),
        build_attacks(pos, BLACK)
    };

    for (Colour colour : { WHITE, BLACK }) {
        ScorePair material;
        ScorePair pieceSquare;
        for (PieceType type : { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING }) {
            Bitboard pieces = pos.pieces(colour, type);
            material.mg += popcount(pieces) * PIECE_MG[type]
                         * piece_material_scale(parameters, type, false);
            material.eg += popcount(pieces) * PIECE_EG[type]
                         * piece_material_scale(parameters, type, true);
            while (pieces) {
                const Square square = static_cast<Square>(pop_lsb(pieces));
                pieceSquare += piece_square_score(type, colour, square);
            }
        }
        material += material_imbalance(pos, colour);
        ScorePair structure;
        ScorePair passed;
        ScorePair mobility;
        ScorePair activity;
        ScorePair threats;
        ScorePair kingSafety;
        evaluate_pawns(pos, colour, structure, passed);
        activity = evaluate_pieces(pos, colour, attacks[colour],
                                   attacks[flip(colour)], phase,
                                   mobility, threats, kingSafety);

        ScorePair& total = sideScore[colour];
        total += scaled(material, parameters.material);
        total += scaled(pieceSquare, parameters.piece_square);
        total += scaled(structure, parameters.pawn_structure);
        total += scaled(passed, parameters.passed_pawns);
        total += scaled(mobility, parameters.mobility);
        total += scaled(activity, parameters.activity);
        total += scaled(threats, parameters.threats);
        total += scaled(kingSafety, parameters.king_safety);
    }

    const double mg = sideScore[WHITE].mg - sideScore[BLACK].mg;
    double eg = sideScore[WHITE].eg - sideScore[BLACK].eg;
    eg += endgame_initiative(pos, eg);
    eg *= endgame_scale(pos, phase);
    const double blended = (mg * phase + eg * (128 - phase)) / 128.0;
    const double tempo = 10.0 * parameters.tempo;
    return static_cast<Score>(
        (pos.side_to_move() == WHITE ? blended : -blended) + tempo);
}

bool load_parameters(const std::string& path, Parameters& parameters,
                     std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "unable to open parameter file: " + path;
        return false;
    }

    Parameters loaded;
    bool seen[19] = {};
    const char* names[19] = {
        "material", "piece_square", "pawn_structure", "passed_pawns",
        "mobility", "activity", "threats", "king_safety", "tempo",
        "pawn_mg", "knight_mg", "bishop_mg", "rook_mg", "queen_mg",
        "pawn_eg", "knight_eg", "bishop_eg", "rook_eg", "queen_eg"
    };
    double* values[19] = {
        &loaded.material, &loaded.piece_square, &loaded.pawn_structure,
        &loaded.passed_pawns, &loaded.mobility, &loaded.activity,
        &loaded.threats, &loaded.king_safety, &loaded.tempo,
        &loaded.pawn_mg, &loaded.knight_mg, &loaded.bishop_mg,
        &loaded.rook_mg, &loaded.queen_mg,
        &loaded.pawn_eg, &loaded.knight_eg, &loaded.bishop_eg,
        &loaded.rook_eg, &loaded.queen_eg
    };

    std::string line;
    i32 lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos)
            line.erase(comment);
        std::istringstream parser(line);
        std::string name;
        if (!(parser >> name))
            continue;
        double value;
        std::string extra;
        if (!(parser >> value) || (parser >> extra) || !std::isfinite(value)) {
            error = "invalid parameter line " + std::to_string(lineNumber);
            return false;
        }

        bool found = false;
        for (i32 i = 0; i < 19; ++i) {
            if (name != names[i])
                continue;
            if (seen[i]) {
                error = "duplicate parameter: " + name;
                return false;
            }
            const double minimum = i >= 9 ? 0.7 : 0.0;
            const double maximum = i >= 9 ? 1.3 : 3.0;
            if (value < minimum || value > maximum) {
                error = "parameter outside supported range ["
                      + std::to_string(minimum) + ", "
                      + std::to_string(maximum) + "]: " + name;
                return false;
            }
            *values[i] = value;
            seen[i] = true;
            found = true;
            break;
        }
        if (!found) {
            error = "unknown parameter: " + name;
            return false;
        }
    }
    if (input.bad()) {
        error = "error reading parameter file: " + path;
        return false;
    }

    for (i32 i = 0; i < 9; ++i) {
        if (!seen[i]) {
            error = "missing parameter: " + std::string(names[i]);
            return false;
        }
    }

    parameters = loaded;
    error.clear();
    return true;
}

bool load_parameters(const std::string& path, std::string& error) {
    Parameters loaded;
    if (!load_parameters(path, loaded, error))
        return false;
    current_parameters = loaded;
    return true;
}

bool save_parameters(const std::string& path, const Parameters& parameters,
                     std::string& error) {
    std::ofstream output(path);
    if (!output) {
        error = "unable to write parameter file: " + path;
        return false;
    }
    output << std::setprecision(10)
           << "material " << parameters.material << '\n'
           << "piece_square " << parameters.piece_square << '\n'
           << "pawn_structure " << parameters.pawn_structure << '\n'
           << "passed_pawns " << parameters.passed_pawns << '\n'
           << "mobility " << parameters.mobility << '\n'
           << "activity " << parameters.activity << '\n'
           << "threats " << parameters.threats << '\n'
           << "king_safety " << parameters.king_safety << '\n'
           << "tempo " << parameters.tempo << '\n'
           << "pawn_mg " << parameters.pawn_mg << '\n'
           << "knight_mg " << parameters.knight_mg << '\n'
           << "bishop_mg " << parameters.bishop_mg << '\n'
           << "rook_mg " << parameters.rook_mg << '\n'
           << "queen_mg " << parameters.queen_mg << '\n'
           << "pawn_eg " << parameters.pawn_eg << '\n'
           << "knight_eg " << parameters.knight_eg << '\n'
           << "bishop_eg " << parameters.bishop_eg << '\n'
           << "rook_eg " << parameters.rook_eg << '\n'
           << "queen_eg " << parameters.queen_eg << '\n';
    if (!output) {
        error = "failed while writing parameter file: " + path;
        return false;
    }
    error.clear();
    return true;
}

void reset_parameters() {
    current_parameters = Parameters{};
}

Parameters get_parameters() {
    return current_parameters;
}

} // namespace Eval
