#include "bitboard.h"
#include "graph_encoder.h"
#include "graph_evaluator.h"
#include "position.h"
#include "zobrist.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char REFERENCE_MAGIC[8] = { 'B', 'M', 'G', 'R', 'E', 'F', '0', '1' };
constexpr ui64 MAX_CASES = 100;

void read_exact(std::istream& input, char* destination, std::size_t count) {
    if (!input.read(destination, static_cast<std::streamsize>(count)))
        throw std::runtime_error("truncated parity reference file");
}

ui32 read_u32_le(std::istream& input) {
    ui8 bytes[4];
    read_exact(input, reinterpret_cast<char*>(bytes), sizeof(bytes));
    return static_cast<ui32>(bytes[0])
        | (static_cast<ui32>(bytes[1]) << 8)
        | (static_cast<ui32>(bytes[2]) << 16)
        | (static_cast<ui32>(bytes[3]) << 24);
}

f32 read_f32_le(std::istream& input) {
    const ui32 bits = read_u32_le(input);
    f32 value;
    std::memcpy(&value, &bits, sizeof(value));
    if (!std::isfinite(value))
        throw std::runtime_error("non-finite parity reference value");
    return value;
}

void read_f32_array(std::istream& input, f32* destination, std::size_t count) {
    for (std::size_t index = 0; index < count; ++index)
        destination[index] = read_f32_le(input);
}

struct ErrorStats {
    f64 absolute_sum = 0.0;
    f32 maximum = 0.0f;
    ui64 count = 0;

    void add(f32 actual, f32 expected) {
        const f32 difference = std::abs(actual - expected);
        maximum = std::max(maximum, difference);
        absolute_sum += difference;
        ++count;
    }

    f64 mean() const { return count == 0 ? 0.0 : absolute_sum / count; }
};

int run(const std::string& modelPath, const std::string& referencePath) {
    BB::init();
    Zobrist::init();

    GNN::GraphEvaluator evaluator;
    std::string loadError;
    if (!evaluator.load(modelPath, loadError))
        throw std::runtime_error("model load failed: " + loadError);

    std::ifstream input(referencePath, std::ios::binary);
    if (!input)
        throw std::runtime_error("unable to open parity reference file");

    char magic[sizeof(REFERENCE_MAGIC)];
    read_exact(input, magic, sizeof(magic));
    if (!std::equal(std::begin(REFERENCE_MAGIC), std::end(REFERENCE_MAGIC), magic))
        throw std::runtime_error("invalid parity reference magic");
    const ui32 caseCount = read_u32_le(input);
    if (caseCount == 0 || caseCount > MAX_CASES)
        throw std::runtime_error("invalid parity reference case count");

    std::cout << std::setprecision(10);
    std::cout << "case,expected_output,cpp_output,graph_max_abs,graph_mean_abs,king_indices_match\n";
    for (ui32 caseIndex = 0; caseIndex < caseCount; ++caseIndex) {
        const ui32 fenLength = read_u32_le(input);
        if (fenLength == 0 || fenLength > 1024)
            throw std::runtime_error("invalid parity reference FEN length");
        std::string fen(fenLength, '\0');
        read_exact(input, fen.data(), fen.size());
        std::array<std::array<f32, 16>, 64> expectedSquares{};
        std::array<f32, 5> expectedGlobal{};
        std::array<f32, 64 * 64> expectedAttack{};
        std::array<f32, 64 * 64> expectedDefence{};
        for (auto& square : expectedSquares)
            read_f32_array(input, square.data(), square.size());
        read_f32_array(input, expectedGlobal.data(), expectedGlobal.size());
        read_f32_array(input, expectedAttack.data(), expectedAttack.size());
        read_f32_array(input, expectedDefence.data(), expectedDefence.size());
        const Square expectedWhiteKing = static_cast<Square>(read_u32_le(input));
        const Square expectedBlackKing = static_cast<Square>(read_u32_le(input));
        const f32 expectedOutput = read_f32_le(input);

        const Position position = Position::from_fen(fen);
        GNN::GraphFeatures encoded;
        GNN::GraphEncoder encoder;
        encoder.encode(position, encoded);

        ErrorStats featureErrors;
        for (std::size_t square = 0; square < 64; ++square) {
            for (std::size_t feature = 0; feature < 16; ++feature)
                featureErrors.add(encoded.squares[square][feature], expectedSquares[square][feature]);
        }
        for (std::size_t feature = 0; feature < 5; ++feature)
            featureErrors.add(encoded.global[feature], expectedGlobal[feature]);
        for (std::size_t target = 0; target < 64; ++target) {
            for (std::size_t source = 0; source < 64; ++source) {
                const f32 attackValue = static_cast<f32>((encoded.attack_sources[target] >> source) & 1ULL);
                const f32 defenceValue = static_cast<f32>((encoded.defence_sources[target] >> source) & 1ULL);
                featureErrors.add(attackValue, expectedAttack[target * 64 + source]);
                featureErrors.add(defenceValue, expectedDefence[target * 64 + source]);
            }
        }
        const bool kingsMatch = encoded.white_king_square == expectedWhiteKing
            && encoded.black_king_square == expectedBlackKing;
        featureErrors.add(kingsMatch ? 0.0f : 1.0f, 0.0f);

        const f32 actualOutput = evaluator.evaluate(encoded);
        std::cout << caseIndex << ',' << expectedOutput << ',' << actualOutput << ','
                  << featureErrors.maximum << ',' << featureErrors.mean() << ','
                  << (kingsMatch ? 1 : 0) << '\n';
    }
    if (input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("unexpected trailing parity reference data");
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: graph_eval_parity <model.bmgnn> <reference.bin>\n";
        return 2;
    }
    try {
        return run(argv[1], argv[2]);
    } catch (const std::exception& exception) {
        std::cerr << "graph parity error: " << exception.what() << '\n';
        return 1;
    }
}
