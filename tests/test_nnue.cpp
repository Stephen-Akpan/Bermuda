#include "catch_amalgamated.hpp"
#include "bitboard.h"
#include "eval.h"
#include "nnue.h"
#include "position.h"
#include "uci.h"
#include "zobrist.h"

#include <cstdio>
#include <cstring>
#include <fstream>

namespace {

void initialize_nnue_tables() {
    static const bool initialized = [] {
        BB::init();
        Zobrist::init();
        return true;
    }();
    (void)initialized;
}

void write_u32_le(std::ofstream& output, ui32 value) {
    const ui8 bytes[4] = {
        static_cast<ui8>(value),
        static_cast<ui8>(value >> 8),
        static_cast<ui8>(value >> 16),
        static_cast<ui8>(value >> 24),
    };
    output.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void write_f32_le(std::ofstream& output, f32 value) {
    ui32 bits;
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32_le(output, bits);
}

std::string write_fixture_network(bool all_features = false) {
    const std::string path = "bermuda-nnue-load-test.nnue";
    std::ofstream output(path, std::ios::binary);
    const char magic[8] = { 'B', 'M', 'N', 'N', 'U', 'E', '1', '\0' };
    constexpr ui32 hidden = NNUE::HIDDEN_SIZE;
    output.write(magic, sizeof(magic));
    write_u32_le(output, 768);
    write_u32_le(output, hidden);
    write_f32_le(output, 400.0f);

    for (i32 feature = 0; feature < 768; ++feature)
        for (ui32 neuron = 0; neuron < hidden; ++neuron)
            write_f32_le(output, all_features
                ? (static_cast<i32>((feature * 17 + neuron * 3) % 29) - 14) * 0.0002f
                : feature == 324 && neuron == 0 ? 1.0f : 0.0f);
    for (ui32 neuron = 0; neuron < hidden; ++neuron)
        write_f32_le(output, 0.0f);
    for (ui32 neuron = 0; neuron < 2 * hidden; ++neuron)
        write_f32_le(output, neuron == 0 ? 2.0f : 0.0f);
    write_f32_le(output, 0.0f);
    return path;
}

} // namespace

TEST_CASE("NNUE loads and evaluates both side-to-move perspectives", "[nnue]") {
    initialize_nnue_tables();
    const std::string path = write_fixture_network();
    std::string error;

    REQUIRE(NNUE::load(path, error));
    REQUIRE(error.empty());
    REQUIRE(NNUE::is_loaded());

    const Position whiteToMove = Position::from_fen(
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    const Position blackToMove = Position::from_fen(
        "4k3/8/8/8/8/8/8/4K3 b - - 0 1");
    CHECK(NNUE::evaluate(whiteToMove) == 800);
    CHECK(NNUE::evaluate(blackToMove) == 800);
    CHECK(Eval::evaluate(whiteToMove) == 800);

    NNUE::clear();
    std::remove(path.c_str());
}

TEST_CASE("Incremental NNUE accumulators match full rebuilds for special moves", "[nnue]") {
    initialize_nnue_tables();
    const std::string path = write_fixture_network(true);
    std::string error;
    REQUIRE(NNUE::load(path, error));

    struct MoveCase {
        const char* fen;
        Move move;
    };
    const MoveCase cases[] = {
        { "4k3/8/8/8/8/8/8/4K3 w - - 0 1", Move::make(e1, e2) },
        { "p3k3/8/8/8/8/8/8/R3K3 w - - 0 1", Move::make(a1, a8) },
        { "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
          Move::make_castling(e1, g1) },
        { "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", Move::make_ep(e5, d6) },
        { "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
          Move::make_promo(a7, a8, PROMO_QUEEN) },
    };

    for (const MoveCase& test : cases) {
        Position before = Position::from_fen(test.fen);
        NNUE::Accumulator parent;
        NNUE::refresh(before, parent);
        NNUE::Accumulator child;
        NNUE::update_for_move(before, test.move, parent, child);
        Position after = before;
        after.make_move(test.move);
        REQUIRE(child.valid);
        CHECK(std::abs(NNUE::evaluate(after, child) - NNUE::evaluate(after)) <= 1);
    }

    NNUE::clear();
    std::remove(path.c_str());
}

TEST_CASE("Invalid NNUE files do not activate a network", "[nnue]") {
    const std::string path = "bermuda-nnue-invalid-test.nnue";
    {
        std::ofstream output(path, std::ios::binary);
        output << "not an NNUE file";
    }

    NNUE::clear();
    std::string error;
    CHECK_FALSE(NNUE::load(path, error));
    CHECK_FALSE(error.empty());
    CHECK_FALSE(NNUE::is_loaded());
    std::remove(path.c_str());
}

TEST_CASE("EvalFile accepts an empty value after UCI input trimming", "[nnue]") {
    const std::string path = write_fixture_network();
    UCI::cmd_setoption("setoption name EvalFile value " + path);
    REQUIRE(NNUE::is_loaded());

    UCI::cmd_setoption("setoption name EvalFile value");
    CHECK_FALSE(NNUE::is_loaded());
    CHECK(UCI::options.eval_file.empty());
    std::remove(path.c_str());
}