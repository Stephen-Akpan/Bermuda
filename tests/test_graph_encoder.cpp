#include "catch_amalgamated.hpp"
#include "bitboard.h"
#include "graph_encoder.h"
#include "zobrist.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <tuple>
#include <type_traits>

namespace {

using SquareFeatures = decltype(GNN::GraphFeatures::squares);
using SquareFeatureArray = typename SquareFeatures::value_type;
using EncodeMethod = void (GNN::GraphEncoder::*)(const Position&, GNN::GraphFeatures&) const;

static_assert(std::tuple_size<SquareFeatures>::value == 64);
static_assert(std::tuple_size<SquareFeatureArray>::value == 16);
static_assert(std::tuple_size<decltype(GNN::GraphFeatures::global)>::value == 5);
static_assert(std::tuple_size<decltype(GNN::GraphFeatures::attack_sources)>::value == 64);
static_assert(std::tuple_size<decltype(GNN::GraphFeatures::defence_sources)>::value == 64);
static_assert(std::is_same<decltype(&GNN::GraphEncoder::encode), EncodeMethod>::value);

static_assert(static_cast<ui8>(GNN::SquareFeature::EMPTY) == 0);
static_assert(static_cast<ui8>(GNN::SquareFeature::WHITE_PAWN) == 1);
static_assert(static_cast<ui8>(GNN::SquareFeature::WHITE_KNIGHT) == 2);
static_assert(static_cast<ui8>(GNN::SquareFeature::WHITE_BISHOP) == 3);
static_assert(static_cast<ui8>(GNN::SquareFeature::WHITE_ROOK) == 4);
static_assert(static_cast<ui8>(GNN::SquareFeature::WHITE_QUEEN) == 5);
static_assert(static_cast<ui8>(GNN::SquareFeature::WHITE_KING) == 6);
static_assert(static_cast<ui8>(GNN::SquareFeature::BLACK_PAWN) == 7);
static_assert(static_cast<ui8>(GNN::SquareFeature::BLACK_KNIGHT) == 8);
static_assert(static_cast<ui8>(GNN::SquareFeature::BLACK_BISHOP) == 9);
static_assert(static_cast<ui8>(GNN::SquareFeature::BLACK_ROOK) == 10);
static_assert(static_cast<ui8>(GNN::SquareFeature::BLACK_QUEEN) == 11);
static_assert(static_cast<ui8>(GNN::SquareFeature::BLACK_KING) == 12);
static_assert(static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT) == 13);
static_assert(static_cast<ui8>(GNN::SquareFeature::BLACK_CONTROL_COUNT) == 14);
static_assert(static_cast<ui8>(GNN::SquareFeature::EN_PASSANT_TARGET) == 15);

static_assert(static_cast<ui8>(GNN::GlobalFeature::SIDE_TO_MOVE) == 0);
static_assert(static_cast<ui8>(GNN::GlobalFeature::WHITE_KINGSIDE_CASTLING) == 1);
static_assert(static_cast<ui8>(GNN::GlobalFeature::WHITE_QUEENSIDE_CASTLING) == 2);
static_assert(static_cast<ui8>(GNN::GlobalFeature::BLACK_KINGSIDE_CASTLING) == 3);
static_assert(static_cast<ui8>(GNN::GlobalFeature::BLACK_QUEENSIDE_CASTLING) == 4);

void initialize_graph_tables() {
    static const bool initialized = [] {
        BB::init();
        Zobrist::init();
        return true;
    }();
    (void)initialized;
}

Position position_from_fen(const std::string& fen) {
    initialize_graph_tables();
    return Position::from_fen(fen);
}

GNN::SquareFeature identity_feature(Piece piece) {
    switch (piece) {
    case WHITE_PAWN:   return GNN::SquareFeature::WHITE_PAWN;
    case WHITE_KNIGHT: return GNN::SquareFeature::WHITE_KNIGHT;
    case WHITE_BISHOP: return GNN::SquareFeature::WHITE_BISHOP;
    case WHITE_ROOK:   return GNN::SquareFeature::WHITE_ROOK;
    case WHITE_QUEEN:  return GNN::SquareFeature::WHITE_QUEEN;
    case WHITE_KING:   return GNN::SquareFeature::WHITE_KING;
    case BLACK_PAWN:   return GNN::SquareFeature::BLACK_PAWN;
    case BLACK_KNIGHT: return GNN::SquareFeature::BLACK_KNIGHT;
    case BLACK_BISHOP: return GNN::SquareFeature::BLACK_BISHOP;
    case BLACK_ROOK:   return GNN::SquareFeature::BLACK_ROOK;
    case BLACK_QUEEN:  return GNN::SquareFeature::BLACK_QUEEN;
    case BLACK_KING:   return GNN::SquareFeature::BLACK_KING;
    default:           return GNN::SquareFeature::EMPTY;
    }
}

ui64 source_bits(std::initializer_list<Square> sources) {
    ui64 bits = 0;
    for (Square source : sources)
        bits |= BB::square_bb(source);
    return bits;
}

void check_control_targets(
    const GNN::GraphFeatures& features,
    GNN::SquareFeature controlFeature,
    std::initializer_list<Square> targets)
{
    std::array<ui8, 64> expectedCounts{};
    for (Square target : targets)
        ++expectedCounts[target];

    const ui8 featureIndex = static_cast<ui8>(controlFeature);
    for (ui8 square = 0; square < 64; ++square) {
        INFO("control feature " << static_cast<int>(featureIndex)
            << " at square index " << static_cast<int>(square));
        CHECK(features.squares[square][featureIndex] == expectedCounts[square] / 8.0f);
    }
}

} // namespace

TEST_CASE("GNN graph encoder matches the initial position", "[gnn]") {
    const Position position = position_from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    GNN::GraphFeatures features;
    GNN::GraphEncoder encoder;
    encoder.encode(position, features);

    for (ui8 square = 0; square < 64; ++square) {
        const GNN::SquareFeature expected = identity_feature(
            position.piece_on(static_cast<Square>(square)));
        for (ui8 feature = 0; feature <= 12; ++feature)
            CHECK(features.squares[square][feature] == (feature == static_cast<ui8>(expected) ? 1.0f : 0.0f));
        CHECK(features.squares[square][static_cast<ui8>(GNN::SquareFeature::EN_PASSANT_TARGET)] == 0.0f);
    }

    CHECK(features.global[static_cast<ui8>(GNN::GlobalFeature::SIDE_TO_MOVE)] == 1.0f);
    CHECK(features.global[static_cast<ui8>(GNN::GlobalFeature::WHITE_KINGSIDE_CASTLING)] == 1.0f);
    CHECK(features.global[static_cast<ui8>(GNN::GlobalFeature::WHITE_QUEENSIDE_CASTLING)] == 1.0f);
    CHECK(features.global[static_cast<ui8>(GNN::GlobalFeature::BLACK_KINGSIDE_CASTLING)] == 1.0f);
    CHECK(features.global[static_cast<ui8>(GNN::GlobalFeature::BLACK_QUEENSIDE_CASTLING)] == 1.0f);
    CHECK(features.white_king_square == e1);
    CHECK(features.black_king_square == e8);

    CHECK(features.defence_sources[d2] == source_bits({ b1, c1, d1, e1 }));
    CHECK(features.defence_sources[e2] == source_bits({ d1, e1, f1, g1 }));
    CHECK(features.defence_sources[d7] == source_bits({ b8, c8, d8, e8 }));
    CHECK(features.defence_sources[e7] == source_bits({ d8, e8, f8, g8 }));
    CHECK(features.squares[d3][static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT)] == 2.0f / 8.0f);
    CHECK(features.squares[d6][static_cast<ui8>(GNN::SquareFeature::BLACK_CONTROL_COUNT)] == 2.0f / 8.0f);
    CHECK(std::all_of(features.attack_sources.begin(), features.attack_sources.end(),
        [](ui64 sources) { return sources == 0; }));
}

TEST_CASE("GNN graph encoder counts white and black pawn diagonals", "[gnn]") {
    const Position position = position_from_fen(
        "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1");
    GNN::GraphFeatures features;
    GNN::GraphEncoder encoder;
    encoder.encode(position, features);

    check_control_targets(features, GNN::SquareFeature::WHITE_CONTROL_COUNT,
        { d5, f5, d1, f1, d2, e2, f2 });
    check_control_targets(features, GNN::SquareFeature::BLACK_CONTROL_COUNT,
        { c4, e4, d7, e7, f7, d8, f8 });
    CHECK(features.attack_sources[d5] == BB::square_bb(e4));
    CHECK(features.attack_sources[e4] == BB::square_bb(d5));
    CHECK(features.defence_sources[d5] == 0);
    CHECK(features.defence_sources[e4] == 0);
    CHECK(features.squares[e5][static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT)] == 0.0f);
    CHECK(features.squares[d4][static_cast<ui8>(GNN::SquareFeature::BLACK_CONTROL_COUNT)] == 0.0f);
}

TEST_CASE("GNN graph encoder counts every knight target", "[gnn]") {
    const Position position = position_from_fen(
        "7k/8/8/8/4N3/8/8/K7 w - - 0 1");
    GNN::GraphFeatures features;
    GNN::GraphEncoder encoder;
    encoder.encode(position, features);

    check_control_targets(features, GNN::SquareFeature::WHITE_CONTROL_COUNT,
        { d6, f6, c5, g5, c3, g3, d2, f2, b1, a2, b2 });
    check_control_targets(features, GNN::SquareFeature::BLACK_CONTROL_COUNT,
        { g7, h7, g8 });
    for (Square target : { d6, f6, c5, g5, c3, g3, d2, f2 }) {
        CHECK(features.attack_sources[target] == 0);
        CHECK(features.defence_sources[target] == 0);
    }
}

TEST_CASE("GNN graph encoder uses geometric king attacks", "[gnn]") {
    const Position position = position_from_fen(
        "k7/8/8/8/4K2r/8/8/8 w - - 0 1");
    GNN::GraphFeatures features;
    GNN::GraphEncoder encoder;
    encoder.encode(position, features);

    check_control_targets(features, GNN::SquareFeature::WHITE_CONTROL_COUNT,
        { d3, d4, d5, e3, e5, f3, f4, f5 });
    check_control_targets(features, GNN::SquareFeature::BLACK_CONTROL_COUNT,
        { g4, f4, e4, h3, h2, h1, h5, h6, h7, h8, a7, b7, b8 });
    CHECK(features.attack_sources[e4] == BB::square_bb(h4));
    CHECK(features.squares[f4][static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT)] == 1.0f / 8.0f);
    CHECK(features.squares[f4][static_cast<ui8>(GNN::SquareFeature::BLACK_CONTROL_COUNT)] == 1.0f / 8.0f);
}

TEST_CASE("GNN graph encoder counts bishop rays exactly", "[gnn]") {
    const Position position = position_from_fen(
        "k7/8/8/8/3B4/8/8/7K w - - 0 1");
    GNN::GraphFeatures features;
    GNN::GraphEncoder encoder;
    encoder.encode(position, features);

    check_control_targets(features, GNN::SquareFeature::WHITE_CONTROL_COUNT,
        { e5, f6, g7, h8, c5, b6, a7, e3, f2, g1, c3, b2, a1, g1, g2, h2 });
    check_control_targets(features, GNN::SquareFeature::BLACK_CONTROL_COUNT,
        { a7, b7, b8 });
    CHECK(std::all_of(features.attack_sources.begin(), features.attack_sources.end(),
        [](ui64 sources) { return sources == 0; }));
    CHECK(std::all_of(features.defence_sources.begin(), features.defence_sources.end(),
        [](ui64 sources) { return sources == 0; }));
}

TEST_CASE("GNN graph encoder counts rook rays exactly", "[gnn]") {
    const Position position = position_from_fen(
        "7k/8/8/8/4R3/8/8/K7 w - - 0 1");
    GNN::GraphFeatures features;
    GNN::GraphEncoder encoder;
    encoder.encode(position, features);

    check_control_targets(features, GNN::SquareFeature::WHITE_CONTROL_COUNT,
        { e1, e2, e3, e5, e6, e7, e8, a4, b4, c4, d4, f4, g4, h4, a2, b1, b2 });
    check_control_targets(features, GNN::SquareFeature::BLACK_CONTROL_COUNT,
        { g7, h7, g8 });
    CHECK(std::all_of(features.attack_sources.begin(), features.attack_sources.end(),
        [](ui64 sources) { return sources == 0; }));
    CHECK(std::all_of(features.defence_sources.begin(), features.defence_sources.end(),
        [](ui64 sources) { return sources == 0; }));
}

TEST_CASE("GNN graph encoder counts queen rays and occupied endpoints", "[gnn]") {
    const Position position = position_from_fen(
        "7k/8/8/8/3Q4/8/8/K7 w - - 0 1");
    GNN::GraphFeatures features;
    GNN::GraphEncoder encoder;
    encoder.encode(position, features);

    check_control_targets(features, GNN::SquareFeature::WHITE_CONTROL_COUNT,
        { a4, b4, c4, e4, f4, g4, h4,
          d1, d2, d3, d5, d6, d7, d8,
          e5, f6, g7, h8, c5, b6, a7, e3, f2, g1, c3, b2, a1,
          b1, a2, b2 });
    check_control_targets(features, GNN::SquareFeature::BLACK_CONTROL_COUNT,
        { g7, g8, h7 });
    CHECK(features.attack_sources[h8] == BB::square_bb(d4));
    CHECK(features.defence_sources[a1] == BB::square_bb(d4));
    CHECK(features.attack_sources[g7] == 0);
    CHECK(features.defence_sources[g7] == 0);
}

TEST_CASE("GNN graph encoder creates exact edges and stops at both blocker colors", "[gnn]") {
    const Position position = position_from_fen(
        "7k/8/4P3/8/4R3/8/4p3/K7 w - - 0 1");
    GNN::GraphFeatures features;
    GNN::GraphEncoder encoder;
    encoder.encode(position, features);

    CHECK(features.defence_sources[e6] == BB::square_bb(e4));
    CHECK(features.attack_sources[e2] == BB::square_bb(e4));
    CHECK(features.attack_sources[e6] == 0);
    CHECK(features.defence_sources[e2] == 0);
    CHECK(features.squares[e5][static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT)] == 1.0f / 8.0f);
    CHECK(features.squares[e6][static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT)] == 1.0f / 8.0f);
    CHECK(features.squares[e3][static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT)] == 1.0f / 8.0f);
    CHECK(features.squares[e2][static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT)] == 1.0f / 8.0f);
    CHECK(features.squares[e7][static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT)] == 0.0f);
    CHECK(features.squares[e1][static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT)] == 0.0f);
}

TEST_CASE("GNN graph encoder preserves pinned rook attacks", "[gnn]") {
    const Position position = position_from_fen(
        "k3r3/8/8/8/8/8/4R3/4K3 w - - 0 1");
    GNN::GraphFeatures features;
    GNN::GraphEncoder encoder;
    encoder.encode(position, features);

    CHECK(features.attack_sources[e8] == BB::square_bb(e2));
    CHECK(features.attack_sources[e2] == BB::square_bb(e8));
    CHECK(features.defence_sources[e1] == BB::square_bb(e2));
    CHECK(features.squares[e7][static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT)] == 1.0f / 8.0f);
    CHECK(features.squares[e7][static_cast<ui8>(GNN::SquareFeature::BLACK_CONTROL_COUNT)] == 1.0f / 8.0f);
    CHECK(features.squares[d2][static_cast<ui8>(GNN::SquareFeature::WHITE_CONTROL_COUNT)] == 2.0f / 8.0f);
}

TEST_CASE("GNN graph encoder handles side, all castling flags, and en passant", "[gnn]") {
    const Position position = position_from_fen(
        "r3k2r/8/8/3pP3/8/8/8/R3K2R b Kq d6 0 1");
    GNN::GraphFeatures features;
    GNN::GraphEncoder encoder;
    encoder.encode(position, features);

    CHECK(features.global[static_cast<ui8>(GNN::GlobalFeature::SIDE_TO_MOVE)] == -1.0f);
    CHECK(features.global[static_cast<ui8>(GNN::GlobalFeature::WHITE_KINGSIDE_CASTLING)] == 1.0f);
    CHECK(features.global[static_cast<ui8>(GNN::GlobalFeature::WHITE_QUEENSIDE_CASTLING)] == 0.0f);
    CHECK(features.global[static_cast<ui8>(GNN::GlobalFeature::BLACK_KINGSIDE_CASTLING)] == 0.0f);
    CHECK(features.global[static_cast<ui8>(GNN::GlobalFeature::BLACK_QUEENSIDE_CASTLING)] == 1.0f);
    CHECK(features.white_king_square == e1);
    CHECK(features.black_king_square == e8);
    for (ui8 square = 0; square < 64; ++square)
        CHECK(features.squares[square][static_cast<ui8>(GNN::SquareFeature::EN_PASSANT_TARGET)]
            == (square == d6 ? 1.0f : 0.0f));

    const Position noRights = position_from_fen(
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    encoder.encode(noRights, features);
    CHECK(features.global[static_cast<ui8>(GNN::GlobalFeature::SIDE_TO_MOVE)] == 1.0f);
    for (ui8 index = 1; index < 5; ++index)
        CHECK(features.global[index] == 0.0f);
    for (const auto& square : features.squares)
        CHECK(square[static_cast<ui8>(GNN::SquareFeature::EN_PASSANT_TARGET)] == 0.0f);
}

TEST_CASE("GNN graph encoder is deterministic, overwrites output, and preserves Position", "[gnn]") {
    const Position position = position_from_fen(
        "r3k2r/8/8/3pP3/8/8/8/R3K2R b Kq d6 17 42");
    const std::string originalFen = position.to_fen();
    const ui64 originalHash = position.hash();

    GNN::GraphFeatures first;
    GNN::GraphFeatures second;
    GNN::GraphFeatures reused;
    for (auto& square : reused.squares)
        square.fill(42.0f);
    reused.global.fill(42.0f);
    reused.attack_sources.fill(~ui64{0});
    reused.defence_sources.fill(~ui64{0});
    reused.white_king_square = a1;
    reused.black_king_square = a1;

    GNN::GraphEncoder encoder;
    encoder.encode(position, first);
    encoder.encode(position, second);
    encoder.encode(position, reused);

    CHECK(first.squares == second.squares);
    CHECK(first.global == second.global);
    CHECK(first.attack_sources == second.attack_sources);
    CHECK(first.defence_sources == second.defence_sources);
    CHECK(first.white_king_square == second.white_king_square);
    CHECK(first.black_king_square == second.black_king_square);
    CHECK(reused.squares == first.squares);
    CHECK(reused.global == first.global);
    CHECK(reused.attack_sources == first.attack_sources);
    CHECK(reused.defence_sources == first.defence_sources);
    CHECK(reused.white_king_square == first.white_king_square);
    CHECK(reused.black_king_square == first.black_king_square);
    CHECK(position.to_fen() == originalFen);
    CHECK(position.hash() == originalHash);
}

TEST_CASE("GNN graph encoder overwrites reused output", "[gnn]") {
    const Position position = position_from_fen(
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    GNN::GraphFeatures features;
    for (auto& square : features.squares)
        square.fill(42.0f);
    features.global.fill(42.0f);
    features.attack_sources.fill(~ui64{0});
    features.defence_sources.fill(~ui64{0});
    features.white_king_square = a1;
    features.black_king_square = a1;

    GNN::GraphEncoder encoder;
    encoder.encode(position, features);
    const GNN::GraphFeatures first = features;
    encoder.encode(position, features);

    CHECK(features.squares == first.squares);
    CHECK(features.global == first.global);
    CHECK(features.attack_sources == first.attack_sources);
    CHECK(features.defence_sources == first.defence_sources);
    CHECK(features.white_king_square == first.white_king_square);
    CHECK(features.black_king_square == first.black_king_square);
    CHECK(features.squares[d4][static_cast<ui8>(GNN::SquareFeature::EMPTY)] == 1.0f);
    CHECK(std::all_of(features.squares[d4].begin() + 1, features.squares[d4].end(),
        [](f32 value) { return value == 0.0f; }));
}