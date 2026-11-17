#pragma once

#include "position.h"

#include <array>

namespace GNN {

constexpr ui8 SQUARE_COUNT = 64;
constexpr ui8 SQUARE_FEATURE_COUNT = 16;
constexpr ui8 GLOBAL_FEATURE_COUNT = 5;

enum class SquareFeature : ui8 {
    EMPTY = 0,
    WHITE_PAWN = 1,
    WHITE_KNIGHT = 2,
    WHITE_BISHOP = 3,
    WHITE_ROOK = 4,
    WHITE_QUEEN = 5,
    WHITE_KING = 6,
    BLACK_PAWN = 7,
    BLACK_KNIGHT = 8,
    BLACK_BISHOP = 9,
    BLACK_ROOK = 10,
    BLACK_QUEEN = 11,
    BLACK_KING = 12,
    WHITE_CONTROL_COUNT = 13,
    BLACK_CONTROL_COUNT = 14,
    EN_PASSANT_TARGET = 15
};

enum class GlobalFeature : ui8 {
    SIDE_TO_MOVE = 0,
    WHITE_KINGSIDE_CASTLING = 1,
    WHITE_QUEENSIDE_CASTLING = 2,
    BLACK_KINGSIDE_CASTLING = 3,
    BLACK_QUEENSIDE_CASTLING = 4
};

struct GraphFeatures {
    std::array<std::array<f32, SQUARE_FEATURE_COUNT>, SQUARE_COUNT> squares{};
    std::array<f32, GLOBAL_FEATURE_COUNT> global{};

    // Each target's bitboard marks source squares: A[target][source] = 1.
    std::array<ui64, SQUARE_COUNT> attack_sources{};
    std::array<ui64, SQUARE_COUNT> defence_sources{};

    Square white_king_square = NO_SQUARE;
    Square black_king_square = NO_SQUARE;
};

class GraphEncoder {
public:
    void encode(const Position& position, GraphFeatures& output) const;
};

} // namespace GNN