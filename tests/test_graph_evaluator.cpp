#include "catch_amalgamated.hpp"
#include "eval.h"
#include "graph_evaluator.h"
#include "nnue.h"
#include "uci.h"
#include "zobrist.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

constexpr ui32 SHA256_TEST_CONSTANTS[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

ui32 rotate_right(ui32 value, ui32 amount) {
    return (value >> amount) | (value << (32 - amount));
}

std::array<ui8, 32> test_sha256(const std::vector<ui8>& data) {
    std::array<ui32, 8> state = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    const ui64 bitLength = static_cast<ui64>(data.size()) * 8;
    const std::size_t paddedLength = ((data.size() + 9 + 63) / 64) * 64;
    std::vector<ui8> padded(paddedLength, 0);
    std::copy(data.begin(), data.end(), padded.begin());
    padded[data.size()] = 0x80;
    for (ui32 byteIndex = 0; byteIndex < 8; ++byteIndex)
        padded[paddedLength - 1 - byteIndex] = static_cast<ui8>(bitLength >> (byteIndex * 8));

    for (std::size_t blockOffset = 0; blockOffset < paddedLength; blockOffset += 64) {
        ui32 schedule[64]{};
        for (ui32 wordIndex = 0; wordIndex < 16; ++wordIndex) {
            const std::size_t byteOffset = blockOffset + wordIndex * 4;
            schedule[wordIndex] =
                (static_cast<ui32>(padded[byteOffset]) << 24)
                | (static_cast<ui32>(padded[byteOffset + 1]) << 16)
                | (static_cast<ui32>(padded[byteOffset + 2]) << 8)
                | static_cast<ui32>(padded[byteOffset + 3]);
        }
        for (ui32 wordIndex = 16; wordIndex < 64; ++wordIndex) {
            const ui32 sigma0 = rotate_right(schedule[wordIndex - 15], 7)
                ^ rotate_right(schedule[wordIndex - 15], 18)
                ^ (schedule[wordIndex - 15] >> 3);
            const ui32 sigma1 = rotate_right(schedule[wordIndex - 2], 17)
                ^ rotate_right(schedule[wordIndex - 2], 19)
                ^ (schedule[wordIndex - 2] >> 10);
            schedule[wordIndex] = schedule[wordIndex - 16] + sigma0
                + schedule[wordIndex - 7] + sigma1;
        }

        ui32 workingA = state[0];
        ui32 workingB = state[1];
        ui32 workingC = state[2];
        ui32 workingD = state[3];
        ui32 workingE = state[4];
        ui32 workingF = state[5];
        ui32 workingG = state[6];
        ui32 workingH = state[7];
        for (ui32 round = 0; round < 64; ++round) {
            const ui32 sum1 = rotate_right(workingE, 6)
                ^ rotate_right(workingE, 11) ^ rotate_right(workingE, 25);
            const ui32 choice = (workingE & workingF) ^ (~workingE & workingG);
            const ui32 temporary1 = workingH + sum1 + choice
                + SHA256_TEST_CONSTANTS[round] + schedule[round];
            const ui32 sum0 = rotate_right(workingA, 2)
                ^ rotate_right(workingA, 13) ^ rotate_right(workingA, 22);
            const ui32 majority = (workingA & workingB)
                ^ (workingA & workingC) ^ (workingB & workingC);
            const ui32 temporary2 = sum0 + majority;
            workingH = workingG;
            workingG = workingF;
            workingF = workingE;
            workingE = workingD + temporary1;
            workingD = workingC;
            workingC = workingB;
            workingB = workingA;
            workingA = temporary1 + temporary2;
        }
        state[0] += workingA;
        state[1] += workingB;
        state[2] += workingC;
        state[3] += workingD;
        state[4] += workingE;
        state[5] += workingF;
        state[6] += workingG;
        state[7] += workingH;
    }

    std::array<ui8, 32> digest{};
    for (std::size_t wordIndex = 0; wordIndex < state.size(); ++wordIndex) {
        for (ui32 byteIndex = 0; byteIndex < 4; ++byteIndex)
            digest[wordIndex * 4 + byteIndex] = static_cast<ui8>(state[wordIndex] >> (24 - byteIndex * 8));
    }
    return digest;
}

void append_u8(std::vector<ui8>& bytes, ui8 value) {
    bytes.push_back(value);
}

void append_u16(std::vector<ui8>& bytes, ui16 value) {
    append_u8(bytes, static_cast<ui8>(value));
    append_u8(bytes, static_cast<ui8>(value >> 8));
}

void append_u32(std::vector<ui8>& bytes, ui32 value) {
    for (ui32 byteIndex = 0; byteIndex < 4; ++byteIndex)
        append_u8(bytes, static_cast<ui8>(value >> (byteIndex * 8)));
}

void append_u64(std::vector<ui8>& bytes, ui64 value) {
    for (ui32 byteIndex = 0; byteIndex < 8; ++byteIndex)
        append_u8(bytes, static_cast<ui8>(value >> (byteIndex * 8)));
}

void append_f32(std::vector<ui8>& bytes, f32 value) {
    ui32 bits;
    std::memcpy(&bits, &value, sizeof(bits));
    append_u32(bytes, bits);
}

struct FixtureTensor {
    std::string name;
    std::vector<ui32> dimensions;
    std::vector<f32> values;
};

FixtureTensor make_tensor(const char* name, std::initializer_list<ui32> dimensions) {
    std::size_t elementCount = 1;
    for (ui32 dimension : dimensions)
        elementCount *= dimension;
    return { name, dimensions, std::vector<f32>(elementCount, 0.0f) };
}

std::vector<ui8> make_model_file(ui32 architectureVersion = 1, bool badFirstDimension = false) {
    std::vector<FixtureTensor> tensors = {
        make_tensor("square_projection.weight", { 32, 16 }),
        make_tensor("square_projection.bias", { 32 }),
        make_tensor("square_embedding.weight", { 64, 32 }),
        make_tensor("global_projection.weight", { 32, 5 }),
        make_tensor("global_projection.bias", { 32 }),
        make_tensor("self_transform.weight", { 32, 32 }),
        make_tensor("attack_transform.weight", { 32, 32 }),
        make_tensor("defence_transform.weight", { 32, 32 }),
        make_tensor("layer_norm.weight", { 32 }),
        make_tensor("layer_norm.bias", { 32 }),
        make_tensor("value_head.0.weight", { 64, 133 }),
        make_tensor("value_head.0.bias", { 64 }),
        make_tensor("value_head.2.weight", { 1, 64 }),
        make_tensor("value_head.2.bias", { 1 }),
    };
    for (std::size_t tensorIndex = 0; tensorIndex < tensors.size(); ++tensorIndex) {
        for (std::size_t valueIndex = 0; valueIndex < tensors[tensorIndex].values.size(); ++valueIndex) {
            const i32 signedValue = static_cast<i32>((valueIndex * 13 + tensorIndex * 7) % 31) - 15;
            tensors[tensorIndex].values[valueIndex] = static_cast<f32>(signedValue) * 0.0005f;
            if (tensorIndex == 8)
                tensors[tensorIndex].values[valueIndex] += 1.0f;
        }
    }

    std::vector<ui8> bytes = { 'B', 'M', 'G', 'R', 'P', 'H', '0', '1' };
    append_u32(bytes, 1);
    append_u32(bytes, architectureVersion);
    append_u32(bytes, 1);
    append_u32(bytes, 14);
    append_u64(bytes, 14561);
    append_f32(bytes, 0.75f);
    const std::string baselineId(64, 'a');
    append_u32(bytes, static_cast<ui32>(baselineId.size()));
    bytes.insert(bytes.end(), baselineId.begin(), baselineId.end());

    for (std::size_t tensorIndex = 0; tensorIndex < tensors.size(); ++tensorIndex) {
        const FixtureTensor& tensor = tensors[tensorIndex];
        append_u16(bytes, static_cast<ui16>(tensor.name.size()));
        append_u8(bytes, static_cast<ui8>(tensor.dimensions.size()));
        bytes.insert(bytes.end(), tensor.name.begin(), tensor.name.end());
        for (std::size_t dimensionIndex = 0; dimensionIndex < tensor.dimensions.size(); ++dimensionIndex) {
            ui32 dimension = tensor.dimensions[dimensionIndex];
            if (badFirstDimension && tensorIndex == 0 && dimensionIndex == 0)
                ++dimension;
            append_u32(bytes, dimension);
        }
        for (f32 value : tensor.values)
            append_f32(bytes, value);
    }
    const auto checksum = test_sha256(bytes);
    bytes.insert(bytes.end(), checksum.begin(), checksum.end());
    return bytes;
}

std::vector<ui8> make_nnue_file() {
    std::vector<ui8> bytes = { 'B', 'M', 'N', 'N', 'U', 'E', '1', 0 };
    append_u32(bytes, 768);
    append_u32(bytes, NNUE::HIDDEN_SIZE);
    append_f32(bytes, 400.0f);
    for (std::size_t index = 0; index < 768 * NNUE::HIDDEN_SIZE; ++index)
        append_f32(bytes, 0.0f);
    for (ui32 index = 0; index < NNUE::HIDDEN_SIZE; ++index)
        append_f32(bytes, 0.0f);
    for (ui32 index = 0; index < 2 * NNUE::HIDDEN_SIZE; ++index)
        append_f32(bytes, 0.0f);
    append_f32(bytes, 0.2f);
    return bytes;
}

std::filesystem::path temporary_model_path() {
    static ui64 sequence = 0;
    const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path()
        / ("bermuda-graph-evaluator-" + std::to_string(timestamp)
            + "-" + std::to_string(++sequence) + ".bmgnn");
}

class TemporaryModelFile {
public:
    explicit TemporaryModelFile(const std::vector<ui8>& bytes)
        : path(temporary_model_path())
    {
        std::ofstream output(path, std::ios::binary);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!output)
            throw std::runtime_error("failed to write C++ model fixture");
    }

    ~TemporaryModelFile() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    const std::string string() const { return path.string(); }

private:
    std::filesystem::path path;
};

GNN::GraphFeatures fixture_features() {
    GNN::GraphFeatures features{};
    for (auto& square : features.squares)
        square[static_cast<ui8>(GNN::SquareFeature::EMPTY)] = 1.0f;
    const auto placePiece = [&features](Square square, GNN::SquareFeature pieceFeature) {
        features.squares[square][static_cast<ui8>(GNN::SquareFeature::EMPTY)] = 0.0f;
        features.squares[square][static_cast<ui8>(pieceFeature)] = 1.0f;
    };
    placePiece(e1, GNN::SquareFeature::WHITE_KING);
    placePiece(e8, GNN::SquareFeature::BLACK_KING);
    placePiece(e4, GNN::SquareFeature::WHITE_KNIGHT);
    placePiece(d4, GNN::SquareFeature::WHITE_QUEEN);
    placePiece(d6, GNN::SquareFeature::BLACK_BISHOP);
    placePiece(f6, GNN::SquareFeature::WHITE_ROOK);
    features.global = { 1.0f, 1.0f, 0.0f, 0.0f, 1.0f };
    features.attack_sources[d6] = BB::square_bb(e4) | BB::square_bb(d4);
    features.defence_sources[f6] = BB::square_bb(e4) | BB::square_bb(d4);
    features.white_king_square = e1;
    features.black_king_square = e8;
    return features;
}

std::string load_error(const std::vector<ui8>& bytes) {
    TemporaryModelFile file(bytes);
    GNN::GraphEvaluator evaluator;
    std::string error;
    CHECK_FALSE(evaluator.load(file.string(), error));
    CHECK_FALSE(error.empty());
    return error;
}

} // namespace

TEST_CASE("C++ GraphEvaluator loads the documented binary model", "[gnn][graph-evaluator]") {
    const std::vector<ui8> bytes = make_model_file();
    TemporaryModelFile file(bytes);
    GNN::GraphEvaluator evaluator;
    std::string error;

    REQUIRE(evaluator.load(file.string(), error));
    CHECK(error.empty());
    CHECK(evaluator.is_loaded());
    CHECK(evaluator.metadata().format_version == 1);
    CHECK(evaluator.metadata().architecture_version == 1);
    CHECK(evaluator.metadata().tensor_count == 14);
    CHECK(evaluator.metadata().parameter_count == 14561);
    CHECK(evaluator.metadata().residual_multiplier == 0.75f);
    CHECK(evaluator.metadata().baseline_checkpoint_id == std::string(64, 'a'));
}

TEST_CASE("C++ GraphEvaluator matches the PyTorch tensor calculation", "[gnn][graph-evaluator]") {
    const std::vector<ui8> bytes = make_model_file();
    TemporaryModelFile file(bytes);
    GNN::GraphEvaluator evaluator;
    std::string error;
    REQUIRE(evaluator.load(file.string(), error));
    const GNN::GraphFeatures features = fixture_features();

    const f32 expectedPyTorchOutput = 0.006478350609540939f;
    CHECK(evaluator.evaluate(features) == Catch::Approx(expectedPyTorchOutput).margin(2e-6f));
    CHECK(evaluator.evaluate_residual(features)
        == Catch::Approx(expectedPyTorchOutput * 0.75f).margin(2e-6f));

    GNN::GraphEvaluatorProfile profile;
    CHECK(evaluator.evaluate_profiled(features, profile) == evaluator.evaluate(features));
    CHECK(profile.input_validation_us >= 0.0);
    CHECK(profile.input_projection_us >= 0.0);
    CHECK(profile.self_transform_us >= 0.0);
    CHECK(profile.attack_transform_us >= 0.0);
    CHECK(profile.defence_transform_us >= 0.0);
    CHECK(profile.aggregation_us >= 0.0);
    CHECK(profile.layer_norm_us >= 0.0);
    CHECK(profile.pooling_and_value_head_us >= 0.0);
    CHECK(profile.input_validation_us + profile.input_projection_us
        + profile.self_transform_us + profile.attack_transform_us
        + profile.defence_transform_us + profile.aggregation_us
        + profile.layer_norm_us
        + profile.pooling_and_value_head_us > 0.0);
}

TEST_CASE("Reordered GraphEvaluator preserves relation aggregation semantics", "[gnn][graph-evaluator]") {
    const std::vector<ui8> bytes = make_model_file();
    TemporaryModelFile file(bytes);
    GNN::GraphEvaluator evaluator;
    std::string error;
    REQUIRE(evaluator.load(file.string(), error));

    GNN::GraphFeatures features = fixture_features();
    const f32 reference = evaluator.evaluate(features);
    const f32 reordered = evaluator.evaluate_reordered(features);
    CHECK(reordered == Catch::Approx(reference).margin(2e-6f));

    GNN::GraphEvaluatorProfile profile;
    CHECK(evaluator.evaluate_reordered_profiled(features, profile) == reordered);
    CHECK(profile.self_transform_us >= 0.0);
    CHECK(profile.attack_transform_us >= 0.0);
    CHECK(profile.defence_transform_us >= 0.0);
    CHECK(profile.aggregation_us >= 0.0);
    CHECK(profile.layer_norm_us >= 0.0);

    features.attack_sources.fill(0);
    features.defence_sources.fill(0);
    CHECK(evaluator.evaluate_reordered(features)
        == Catch::Approx(evaluator.evaluate(features)).margin(2e-6f));
}

TEST_CASE("C++ GraphEvaluator inference is deterministic", "[gnn][graph-evaluator]") {
    const std::vector<ui8> bytes = make_model_file();
    TemporaryModelFile file(bytes);
    GNN::GraphEvaluator evaluator;
    std::string error;
    REQUIRE(evaluator.load(file.string(), error));
    const GNN::GraphFeatures features = fixture_features();

    const f32 first = evaluator.evaluate(features);
    for (int repeat = 0; repeat < 5; ++repeat)
        CHECK(evaluator.evaluate(features) == first);
}

TEST_CASE("C++ GraphEvaluator rejects corrupt and incompatible model files", "[gnn][graph-evaluator]") {
    CHECK(load_error(make_model_file(2)).find("architecture") != std::string::npos);
    CHECK(load_error(make_model_file(1, true)).find("dimensions") != std::string::npos);

    std::vector<ui8> badChecksum = make_model_file();
    badChecksum[50] ^= 0x01;
    CHECK(load_error(badChecksum).find("SHA-256") != std::string::npos);

    std::vector<ui8> truncated = make_model_file();
    truncated.pop_back();
    CHECK(load_error(truncated).find("SHA-256") != std::string::npos);
}

TEST_CASE("C++ GraphEvaluator failed load does not replace a loaded model", "[gnn][graph-evaluator]") {
    TemporaryModelFile validFile(make_model_file());
    TemporaryModelFile invalidFile(make_model_file(2));
    GNN::GraphEvaluator evaluator;
    std::string error;
    REQUIRE(evaluator.load(validFile.string(), error));
    const f32 expected = evaluator.evaluate(fixture_features());

    CHECK_FALSE(evaluator.load(invalidFile.string(), error));
    CHECK(evaluator.is_loaded());
    CHECK(evaluator.evaluate(fixture_features()) == expected);
}

TEST_CASE("C++ GraphEvaluator rejects inference before loading", "[gnn][graph-evaluator]") {
    GNN::GraphEvaluator evaluator;
    CHECK_THROWS_AS(evaluator.evaluate(fixture_features()), std::logic_error);
}

TEST_CASE("Test fixture SHA-256 writer matches the standard vector", "[gnn][graph-evaluator]") {
    const std::vector<ui8> input = { 'a', 'b', 'c' };
    const auto digest = test_sha256(input);
    const ui8 expected[32] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
    };
    CHECK(std::equal(digest.begin(), digest.end(), expected));
}

Score expected_combined_score(const Position& position,
                              const NNUE::Accumulator& accumulator,
                              GNN::GraphEvaluator& graphModel,
                              f32 runtimeScale) {
    GNN::GraphFeatures features;
    GNN::GraphEncoder encoder;
    encoder.encode(position, features);
    const f32 whiteProbability = NNUE::predict_white_probability(position, accumulator);
    const f32 probabilityDelta = graphModel.evaluate_residual(features) * runtimeScale;
    const f32 combinedProbability = std::clamp(
        whiteProbability + probabilityDelta, 1e-6f, 1.0f - 1e-6f);
    const f64 whiteCentipawns = NNUE::eval_scale() * std::log(
        static_cast<f64>(combinedProbability)
        / static_cast<f64>(1.0f - combinedProbability));
    const Score whiteScore = static_cast<Score>(std::lround(whiteCentipawns));
    return position.side_to_move() == WHITE ? whiteScore : -whiteScore;
}

void initialize_graph_integration_tables() {
    static const bool initialized = [] {
        BB::init();
        Zobrist::init();
        return true;
    }();
    (void)initialized;
}

TEST_CASE("Eval preserves NNUE-only output and combines scaled white residuals", "[gnn][graph-integration]") {
    initialize_graph_integration_tables();
    NNUE::clear();
    Eval::clear_graph_model();
    std::string error;
    CHECK_FALSE(Eval::set_graph_residual_enabled(true, error));

    TemporaryModelFile graphFile(make_model_file());
    TemporaryModelFile nnueFile(make_nnue_file());
    REQUIRE(Eval::load_graph_model(graphFile.string(), error));
    REQUIRE(NNUE::load(nnueFile.string(), error));
    CHECK_FALSE(Eval::graph_residual_enabled());

    GNN::GraphEvaluator graphModel;
    REQUIRE(graphModel.load(graphFile.string(), error));
    const Position whiteToMove = Position::from_fen(
        "4k3/8/8/8/8/2N5/8/4K3 w - - 0 1");
    const Position blackToMove = Position::from_fen(
        "4k3/8/8/8/8/2N5/8/4K3 b - - 0 1");
    NNUE::Accumulator whiteAccumulator;
    NNUE::refresh(whiteToMove, whiteAccumulator);
    NNUE::Accumulator blackAccumulator;
    NNUE::refresh(blackToMove, blackAccumulator);

    const Score whiteBaseline = NNUE::evaluate(whiteToMove, whiteAccumulator);
    const Score blackBaseline = NNUE::evaluate(blackToMove, blackAccumulator);
    CHECK(Eval::evaluate(whiteToMove) == whiteBaseline);
    CHECK(Eval::evaluate(blackToMove, blackAccumulator) == blackBaseline);

    REQUIRE(Eval::set_graph_residual_scale(0.5f, error));
    REQUIRE(Eval::set_graph_residual_enabled(true, error));
    CHECK(Eval::graph_residual_enabled());
    CHECK(Eval::evaluate(whiteToMove) == expected_combined_score(
        whiteToMove, whiteAccumulator, graphModel, 0.5f));
    CHECK(Eval::evaluate(blackToMove, blackAccumulator) == expected_combined_score(
        blackToMove, blackAccumulator, graphModel, 0.5f));

    REQUIRE(Eval::set_graph_residual_scale(0.0f, error));
    CHECK(Eval::evaluate(whiteToMove) == whiteBaseline);
    CHECK_FALSE(Eval::set_graph_residual_scale(-0.1f, error));
    CHECK_FALSE(error.empty());

    NNUE::clear();
    Eval::clear_graph_model();
}

TEST_CASE("UCI exposes configurable graph model enablement and scaling", "[gnn][graph-integration]") {
    initialize_graph_integration_tables();
    NNUE::clear();
    Eval::clear_graph_model();
    UCI::options.graph_eval = false;
    UCI::options.graph_scale = 1.0f;
    UCI::options.eval_file.clear();
    UCI::options.graph_eval_file.clear();

    TemporaryModelFile graphFile(make_model_file());
    TemporaryModelFile nnueFile(make_nnue_file());
    UCI::cmd_setoption("setoption name GraphEval value true");
    CHECK_FALSE(UCI::options.graph_eval);
    UCI::cmd_setoption("setoption name GraphEvalFile value " + graphFile.string());
    CHECK(Eval::graph_model_loaded());
    CHECK_FALSE(UCI::options.graph_eval);
    UCI::cmd_setoption("setoption name EvalFile value " + nnueFile.string());
    UCI::cmd_setoption("setoption name GraphScale value 0.25");
    UCI::cmd_setoption("setoption name GraphEval value true");
    CHECK(UCI::options.graph_eval);
    CHECK(Eval::graph_residual_enabled());
    CHECK(UCI::options.graph_scale == 0.25f);

    UCI::cmd_setoption("setoption name EvalFile value");
    CHECK_FALSE(UCI::options.graph_eval);
    UCI::cmd_setoption("setoption name GraphEvalFile value");
    CHECK_FALSE(Eval::graph_model_loaded());
    CHECK(UCI::options.graph_eval_file.empty());
    UCI::cmd_setoption("setoption name GraphScale value 1.0");
    NNUE::clear();
}