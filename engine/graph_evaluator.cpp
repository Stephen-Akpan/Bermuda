#include "graph_evaluator.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

namespace GNN {
namespace {

constexpr ui8 FILE_MAGIC[8] = { 'B', 'M', 'G', 'R', 'P', 'H', '0', '1' };
constexpr ui32 FORMAT_VERSION = 1;
constexpr ui32 ARCHITECTURE_VERSION = 1;
constexpr ui32 FLOAT32_CODE = 1;
constexpr ui32 TENSOR_COUNT = 14;
constexpr ui64 PARAMETER_COUNT = 14561;
constexpr std::size_t FILE_HEADER_SIZE = 40;
constexpr std::size_t CHECKSUM_SIZE = 32;
constexpr std::size_t MAX_MODEL_FILE_SIZE = 1024 * 1024;
constexpr f32 LAYER_NORM_EPSILON = 1e-5f;
constexpr f32 RELU_ZERO = 0.0f;

static_assert(sizeof(f32) == 4, "GraphEvaluator export requires 32-bit float");
static_assert(std::numeric_limits<f32>::is_iec559, "GraphEvaluator export requires IEEE-754 float");

constexpr ui32 SHA256_ROUND_CONSTANTS[64] = {
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

std::array<ui8, CHECKSUM_SIZE> sha256(const ui8* data, std::size_t length) {
    std::array<ui32, 8> state = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    const ui64 bitLength = static_cast<ui64>(length) * 8;
    const std::size_t paddedLength = ((length + 9 + 63) / 64) * 64;
    std::vector<ui8> padded(paddedLength, 0);
    if (length != 0)
        std::memcpy(padded.data(), data, length);
    padded[length] = 0x80;
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
                + SHA256_ROUND_CONSTANTS[round] + schedule[round];
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

    std::array<ui8, CHECKSUM_SIZE> digest{};
    for (std::size_t wordIndex = 0; wordIndex < state.size(); ++wordIndex) {
        for (ui32 byteIndex = 0; byteIndex < 4; ++byteIndex)
            digest[wordIndex * 4 + byteIndex] = static_cast<ui8>(state[wordIndex] >> (24 - byteIndex * 8));
    }
    return digest;
}

class ByteReader {
public:
    ByteReader(const std::vector<ui8>& bytes, std::size_t length)
        : bytes(bytes), length(length) {}

    ui8 read_u8() {
        require(1);
        return bytes[offset++];
    }

    ui16 read_u16() {
        const ui16 low = read_u8();
        const ui16 high = read_u8();
        return static_cast<ui16>(low | (high << 8));
    }

    ui32 read_u32() {
        ui32 value = 0;
        for (ui32 byteIndex = 0; byteIndex < 4; ++byteIndex)
            value |= static_cast<ui32>(read_u8()) << (byteIndex * 8);
        return value;
    }

    ui64 read_u64() {
        ui64 value = 0;
        for (ui32 byteIndex = 0; byteIndex < 8; ++byteIndex)
            value |= static_cast<ui64>(read_u8()) << (byteIndex * 8);
        return value;
    }

    f32 read_f32() {
        const ui32 bits = read_u32();
        f32 value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    std::string read_string(std::size_t size) {
        require(size);
        const char* begin = reinterpret_cast<const char*>(bytes.data() + offset);
        std::string value(begin, size);
        offset += size;
        return value;
    }

    void read_tensor(
        const char* expectedName,
        std::initializer_list<ui32> expectedDimensions,
        f32* destination,
        std::size_t elementCount)
    {
        const ui16 nameLength = read_u16();
        const ui8 rank = read_u8();
        if (rank != expectedDimensions.size())
            throw std::runtime_error(std::string("tensor rank mismatch: ") + expectedName);
        const std::string name = read_string(nameLength);
        if (name != expectedName)
            throw std::runtime_error(std::string("tensor order/name mismatch: ") + name);
        std::size_t declaredElements = 1;
        for (ui32 expectedDimension : expectedDimensions) {
            const ui32 dimension = read_u32();
            if (dimension != expectedDimension)
                throw std::runtime_error(std::string("tensor dimensions mismatch: ") + expectedName);
            declaredElements *= dimension;
        }
        if (declaredElements != elementCount)
            throw std::runtime_error(std::string("tensor element count mismatch: ") + expectedName);
        for (std::size_t elementIndex = 0; elementIndex < elementCount; ++elementIndex) {
            destination[elementIndex] = read_f32();
            if (!std::isfinite(destination[elementIndex]))
                throw std::runtime_error(std::string("non-finite tensor value: ") + expectedName);
        }
    }

    std::size_t consumed() const { return offset; }

private:
    void require(std::size_t size) const {
        if (size > length - offset)
            throw std::runtime_error("truncated GraphEvaluator model file");
    }

    const std::vector<ui8>& bytes;
    std::size_t length;
    std::size_t offset = 0;
};

bool is_hex_identifier(const std::string& value) {
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char character) {
        return (character >= '0' && character <= '9')
            || (character >= 'a' && character <= 'f')
            || (character >= 'A' && character <= 'F');
    });
}

f32 relu(f32 value) {
    return value > RELU_ZERO ? value : RELU_ZERO;
}

} // namespace

bool GraphEvaluator::load(const std::string& path, std::string& error) {
    error.clear();
    try {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input)
            throw std::runtime_error("unable to open GraphEvaluator model file: " + path);
        const std::streamoff fileLength = input.tellg();
        if (fileLength < static_cast<std::streamoff>(FILE_HEADER_SIZE + CHECKSUM_SIZE)
            || fileLength > static_cast<std::streamoff>(MAX_MODEL_FILE_SIZE))
            throw std::runtime_error("invalid GraphEvaluator model file size");
        std::vector<ui8> bytes(static_cast<std::size_t>(fileLength));
        input.seekg(0, std::ios::beg);
        if (!input.read(reinterpret_cast<char*>(bytes.data()), fileLength))
            throw std::runtime_error("failed to read complete GraphEvaluator model file");

        const std::size_t contentLength = bytes.size() - CHECKSUM_SIZE;
        const auto digest = sha256(bytes.data(), contentLength);
        if (!std::equal(digest.begin(), digest.end(), bytes.begin() + contentLength))
            throw std::runtime_error("GraphEvaluator model SHA-256 mismatch");

        ByteReader reader(bytes, contentLength);
        for (ui8 expectedByte : FILE_MAGIC) {
            if (reader.read_u8() != expectedByte)
                throw std::runtime_error("invalid GraphEvaluator model magic");
        }

        GraphModelMetadata candidateMetadata;
        candidateMetadata.format_version = reader.read_u32();
        candidateMetadata.architecture_version = reader.read_u32();
        const ui32 dtypeCode = reader.read_u32();
        candidateMetadata.tensor_count = reader.read_u32();
        candidateMetadata.parameter_count = reader.read_u64();
        candidateMetadata.residual_multiplier = reader.read_f32();
        const ui32 baselineIdentifierLength = reader.read_u32();

        if (candidateMetadata.format_version != FORMAT_VERSION)
            throw std::runtime_error("unsupported GraphEvaluator export format version");
        if (candidateMetadata.architecture_version != ARCHITECTURE_VERSION)
            throw std::runtime_error("unsupported GraphEvaluator architecture version");
        if (dtypeCode != FLOAT32_CODE)
            throw std::runtime_error("GraphEvaluator model is not float32");
        if (candidateMetadata.tensor_count != TENSOR_COUNT
            || candidateMetadata.parameter_count != PARAMETER_COUNT)
            throw std::runtime_error("GraphEvaluator model tensor/parameter count mismatch");
        if (!std::isfinite(candidateMetadata.residual_multiplier)
            || candidateMetadata.residual_multiplier <= 0.0f)
            throw std::runtime_error("invalid GraphEvaluator residual multiplier");

        candidateMetadata.baseline_checkpoint_id = reader.read_string(baselineIdentifierLength);
        if (!is_hex_identifier(candidateMetadata.baseline_checkpoint_id))
            throw std::runtime_error("invalid baseline checkpoint identifier");

        Parameters candidateParameters{};
        reader.read_tensor("square_projection.weight", { 32, 16 },
            candidateParameters.square_projection_weight.data(), candidateParameters.square_projection_weight.size());
        reader.read_tensor("square_projection.bias", { 32 },
            candidateParameters.square_projection_bias.data(), candidateParameters.square_projection_bias.size());
        reader.read_tensor("square_embedding.weight", { 64, 32 },
            candidateParameters.square_embedding_weight.data(), candidateParameters.square_embedding_weight.size());
        reader.read_tensor("global_projection.weight", { 32, 5 },
            candidateParameters.global_projection_weight.data(), candidateParameters.global_projection_weight.size());
        reader.read_tensor("global_projection.bias", { 32 },
            candidateParameters.global_projection_bias.data(), candidateParameters.global_projection_bias.size());
        reader.read_tensor("self_transform.weight", { 32, 32 },
            candidateParameters.self_transform_weight.data(), candidateParameters.self_transform_weight.size());
        reader.read_tensor("attack_transform.weight", { 32, 32 },
            candidateParameters.attack_transform_weight.data(), candidateParameters.attack_transform_weight.size());
        reader.read_tensor("defence_transform.weight", { 32, 32 },
            candidateParameters.defence_transform_weight.data(), candidateParameters.defence_transform_weight.size());
        reader.read_tensor("layer_norm.weight", { 32 },
            candidateParameters.layer_norm_weight.data(), candidateParameters.layer_norm_weight.size());
        reader.read_tensor("layer_norm.bias", { 32 },
            candidateParameters.layer_norm_bias.data(), candidateParameters.layer_norm_bias.size());
        reader.read_tensor("value_head.0.weight", { 64, 133 },
            candidateParameters.value_hidden_weight.data(), candidateParameters.value_hidden_weight.size());
        reader.read_tensor("value_head.0.bias", { 64 },
            candidateParameters.value_hidden_bias.data(), candidateParameters.value_hidden_bias.size());
        reader.read_tensor("value_head.2.weight", { 1, 64 },
            candidateParameters.value_output_weight.data(), candidateParameters.value_output_weight.size());
        reader.read_tensor("value_head.2.bias", { 1 },
            candidateParameters.value_output_bias.data(), candidateParameters.value_output_bias.size());
        if (reader.consumed() != contentLength)
            throw std::runtime_error("unexpected trailing data in GraphEvaluator model file");

        parameters = candidateParameters;
        model_metadata = candidateMetadata;
        loaded = true;
        return true;
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

f32 GraphEvaluator::evaluate(const GraphFeatures& features) const {
    return evaluate_impl<false, false>(features, nullptr);
}

f32 GraphEvaluator::evaluate_profiled(
    const GraphFeatures& features,
    GraphEvaluatorProfile& profile) const
{
    profile = GraphEvaluatorProfile{};
    return evaluate_impl<true, false>(features, &profile);
}

f32 GraphEvaluator::evaluate_reordered(const GraphFeatures& features) const {
    return evaluate_impl<false, true>(features, nullptr);
}

f32 GraphEvaluator::evaluate_reordered_profiled(
    const GraphFeatures& features,
    GraphEvaluatorProfile& profile) const
{
    profile = GraphEvaluatorProfile{};
    return evaluate_impl<true, true>(features, &profile);
}

template <bool Profile, bool Reordered>
f32 GraphEvaluator::evaluate_impl(
    const GraphFeatures& features,
    GraphEvaluatorProfile* profile) const
{
    using Clock = std::chrono::steady_clock;
    Clock::time_point stageStarted;
    if constexpr (Profile)
        stageStarted = Clock::now();

    if (!loaded)
        throw std::logic_error("GraphEvaluator model has not been loaded");
    if (features.white_king_square >= SQUARE_COUNT || features.black_king_square >= SQUARE_COUNT)
        throw std::invalid_argument("GraphEvaluator requires valid white and black king squares");
    if constexpr (Profile) {
        profile->input_validation_us =
            std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();
        stageStarted = Clock::now();
    }

    constexpr std::size_t hiddenSize = 32;
    std::array<f32, SQUARE_COUNT * hiddenSize> initialHidden{};
    std::array<f32, hiddenSize> globalProjection{};
    for (std::size_t outputIndex = 0; outputIndex < hiddenSize; ++outputIndex) {
        f32 sum = 0.0f;
        for (std::size_t featureIndex = 0; featureIndex < GLOBAL_FEATURE_COUNT; ++featureIndex)
            sum += features.global[featureIndex]
                * parameters.global_projection_weight[outputIndex * GLOBAL_FEATURE_COUNT + featureIndex];
        globalProjection[outputIndex] = sum + parameters.global_projection_bias[outputIndex];
    }

    for (std::size_t square = 0; square < SQUARE_COUNT; ++square) {
        for (std::size_t outputIndex = 0; outputIndex < hiddenSize; ++outputIndex) {
            f32 squareProjection = 0.0f;
            for (std::size_t featureIndex = 0; featureIndex < SQUARE_FEATURE_COUNT; ++featureIndex)
                squareProjection += features.squares[square][featureIndex]
                    * parameters.square_projection_weight[outputIndex * SQUARE_FEATURE_COUNT + featureIndex];
            const f32 projected = squareProjection + parameters.square_projection_bias[outputIndex];
            const f32 embedded = parameters.square_embedding_weight[square * hiddenSize + outputIndex];
            initialHidden[square * hiddenSize + outputIndex] = relu(
                (projected + embedded) + globalProjection[outputIndex]);
        }
    }
    if constexpr (Profile) {
        profile->input_projection_us =
            std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();
        stageStarted = Clock::now();
    }

    std::array<f32, SQUARE_COUNT * hiddenSize> selfMessages{};
    std::array<f32, SQUARE_COUNT * hiddenSize> attackMessages{};
    std::array<f32, SQUARE_COUNT * hiddenSize> defenceMessages{};
    std::array<f32, SQUARE_COUNT * hiddenSize> attackAggregates{};
    std::array<f32, SQUARE_COUNT * hiddenSize> defenceAggregates{};
    const auto transformSquares = [&initialHidden, hiddenSize](const auto& weight, auto& output) {
        for (std::size_t square = 0; square < SQUARE_COUNT; ++square) {
            for (std::size_t outputIndex = 0; outputIndex < hiddenSize; ++outputIndex) {
                f32 sum = 0.0f;
                for (std::size_t inputIndex = 0; inputIndex < hiddenSize; ++inputIndex)
                    sum += initialHidden[square * hiddenSize + inputIndex]
                        * weight[outputIndex * hiddenSize + inputIndex];
                output[square * hiddenSize + outputIndex] = sum;
            }
        }
    };

    if constexpr (Reordered) {
        if constexpr (Profile)
            stageStarted = Clock::now();
        transformSquares(parameters.self_transform_weight, selfMessages);
        if constexpr (Profile) {
            profile->self_transform_us =
                std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();
            stageStarted = Clock::now();
        }

        std::array<i32, SQUARE_COUNT> attackDegrees{};
        std::array<i32, SQUARE_COUNT> defenceDegrees{};
        for (std::size_t target = 0; target < SQUARE_COUNT; ++target) {
            ui64 attackSources = features.attack_sources[target];
            ui64 defenceSources = features.defence_sources[target];
            attackDegrees[target] = popcount(attackSources);
            defenceDegrees[target] = popcount(defenceSources);
            for (std::size_t inputIndex = 0; inputIndex < hiddenSize; ++inputIndex) {
                f32 attackSum = 0.0f;
                f32 defenceSum = 0.0f;
                ui64 attackBits = attackSources;
                while (attackBits) {
                    const Square source = static_cast<Square>(pop_lsb(attackBits));
                    attackSum += initialHidden[source * hiddenSize + inputIndex];
                }
                ui64 defenceBits = defenceSources;
                while (defenceBits) {
                    const Square source = static_cast<Square>(pop_lsb(defenceBits));
                    defenceSum += initialHidden[source * hiddenSize + inputIndex];
                }
                if (attackDegrees[target] > 0)
                    attackSum /= static_cast<f32>(attackDegrees[target]);
                if (defenceDegrees[target] > 0)
                    defenceSum /= static_cast<f32>(defenceDegrees[target]);
                attackAggregates[target * hiddenSize + inputIndex] = attackSum;
                defenceAggregates[target * hiddenSize + inputIndex] = defenceSum;
            }
        }
        if constexpr (Profile) {
            profile->aggregation_us =
                std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();
            stageStarted = Clock::now();
        }

        for (std::size_t target = 0; target < SQUARE_COUNT; ++target) {
            if (attackDegrees[target] == 0)
                continue;
            for (std::size_t outputIndex = 0; outputIndex < hiddenSize; ++outputIndex) {
                f32 sum = 0.0f;
                for (std::size_t inputIndex = 0; inputIndex < hiddenSize; ++inputIndex)
                    sum += attackAggregates[target * hiddenSize + inputIndex]
                        * parameters.attack_transform_weight[outputIndex * hiddenSize + inputIndex];
                attackMessages[target * hiddenSize + outputIndex] = sum;
            }
        }
        if constexpr (Profile) {
            profile->attack_transform_us =
                std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();
            stageStarted = Clock::now();
        }

        for (std::size_t target = 0; target < SQUARE_COUNT; ++target) {
            if (defenceDegrees[target] == 0)
                continue;
            for (std::size_t outputIndex = 0; outputIndex < hiddenSize; ++outputIndex) {
                f32 sum = 0.0f;
                for (std::size_t inputIndex = 0; inputIndex < hiddenSize; ++inputIndex)
                    sum += defenceAggregates[target * hiddenSize + inputIndex]
                        * parameters.defence_transform_weight[outputIndex * hiddenSize + inputIndex];
                defenceMessages[target * hiddenSize + outputIndex] = sum;
            }
        }
        if constexpr (Profile) {
            profile->defence_transform_us =
                std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();
            stageStarted = Clock::now();
        }
    } else if constexpr (Profile) {
        stageStarted = Clock::now();
        transformSquares(parameters.self_transform_weight, selfMessages);
        profile->self_transform_us =
            std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();

        stageStarted = Clock::now();
        transformSquares(parameters.attack_transform_weight, attackMessages);
        profile->attack_transform_us =
            std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();

        stageStarted = Clock::now();
        transformSquares(parameters.defence_transform_weight, defenceMessages);
        profile->defence_transform_us =
            std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();
    } else {
        for (std::size_t source = 0; source < SQUARE_COUNT; ++source) {
            for (std::size_t outputIndex = 0; outputIndex < hiddenSize; ++outputIndex) {
                f32 selfSum = 0.0f;
                f32 attackSum = 0.0f;
                f32 defenceSum = 0.0f;
                for (std::size_t inputIndex = 0; inputIndex < hiddenSize; ++inputIndex) {
                    const f32 value = initialHidden[source * hiddenSize + inputIndex];
                    selfSum += value * parameters.self_transform_weight[outputIndex * hiddenSize + inputIndex];
                    attackSum += value * parameters.attack_transform_weight[outputIndex * hiddenSize + inputIndex];
                    defenceSum += value * parameters.defence_transform_weight[outputIndex * hiddenSize + inputIndex];
                }
                selfMessages[source * hiddenSize + outputIndex] = selfSum;
                attackMessages[source * hiddenSize + outputIndex] = attackSum;
                defenceMessages[source * hiddenSize + outputIndex] = defenceSum;
            }
        }
    }

    std::array<f32, SQUARE_COUNT * hiddenSize> combinedHidden{};
    if constexpr (!Reordered) {
        if constexpr (Profile)
            stageStarted = Clock::now();
        for (std::size_t target = 0; target < SQUARE_COUNT; ++target) {
            const ui64 attackSources = features.attack_sources[target];
            const ui64 defenceSources = features.defence_sources[target];
            const i32 attackDegree = popcount(attackSources);
            const i32 defenceDegree = popcount(defenceSources);
            for (std::size_t hiddenIndex = 0; hiddenIndex < hiddenSize; ++hiddenIndex) {
                f32 attackAggregate = 0.0f;
                f32 defenceAggregate = 0.0f;
                ui64 attackBits = attackSources;
                while (attackBits) {
                    const Square source = static_cast<Square>(pop_lsb(attackBits));
                    attackAggregate += attackMessages[source * hiddenSize + hiddenIndex];
                }
                ui64 defenceBits = defenceSources;
                while (defenceBits) {
                    const Square source = static_cast<Square>(pop_lsb(defenceBits));
                    defenceAggregate += defenceMessages[source * hiddenSize + hiddenIndex];
                }
                if (attackDegree > 0)
                    attackAggregate /= static_cast<f32>(attackDegree);
                if (defenceDegree > 0)
                    defenceAggregate /= static_cast<f32>(defenceDegree);
                combinedHidden[target * hiddenSize + hiddenIndex] =
                    ((initialHidden[target * hiddenSize + hiddenIndex]
                        + selfMessages[target * hiddenSize + hiddenIndex])
                        + attackAggregate) + defenceAggregate;
            }
        }
        if constexpr (Profile) {
            profile->aggregation_us =
                std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();
            stageStarted = Clock::now();
        }
    } else {
        for (std::size_t target = 0; target < SQUARE_COUNT; ++target) {
            for (std::size_t hiddenIndex = 0; hiddenIndex < hiddenSize; ++hiddenIndex)
                combinedHidden[target * hiddenSize + hiddenIndex] =
                    ((initialHidden[target * hiddenSize + hiddenIndex]
                        + selfMessages[target * hiddenSize + hiddenIndex])
                        + attackMessages[target * hiddenSize + hiddenIndex])
                        + defenceMessages[target * hiddenSize + hiddenIndex];
        }
    }
    if constexpr (Reordered && Profile) {
        profile->aggregation_us +=
            std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();
        stageStarted = Clock::now();
    }

    std::array<f32, SQUARE_COUNT * hiddenSize> finalHidden{};
    for (std::size_t target = 0; target < SQUARE_COUNT; ++target) {
        f32 mean = 0.0f;
        for (std::size_t hiddenIndex = 0; hiddenIndex < hiddenSize; ++hiddenIndex)
            mean += combinedHidden[target * hiddenSize + hiddenIndex];
        mean /= static_cast<f32>(hiddenSize);

        f32 variance = 0.0f;
        for (std::size_t hiddenIndex = 0; hiddenIndex < hiddenSize; ++hiddenIndex) {
            const f32 difference = combinedHidden[target * hiddenSize + hiddenIndex] - mean;
            variance += difference * difference;
        }
        variance /= static_cast<f32>(hiddenSize);
        for (std::size_t hiddenIndex = 0; hiddenIndex < hiddenSize; ++hiddenIndex) {
            const f32 normalized = (combinedHidden[target * hiddenSize + hiddenIndex] - mean)
                / std::sqrt(variance + LAYER_NORM_EPSILON);
            finalHidden[target * hiddenSize + hiddenIndex] = relu(
                normalized * parameters.layer_norm_weight[hiddenIndex]
                + parameters.layer_norm_bias[hiddenIndex]);
        }
    }
    if constexpr (Profile) {
        profile->layer_norm_us =
            std::chrono::duration<double, std::micro>(Clock::now() - stageStarted).count();
        stageStarted = Clock::now();
    }
    std::array<f32, 133> pooled{};
    for (std::size_t hiddenIndex = 0; hiddenIndex < hiddenSize; ++hiddenIndex) {
        f32 mean = 0.0f;
        f32 maximum = -std::numeric_limits<f32>::infinity();
        for (std::size_t square = 0; square < SQUARE_COUNT; ++square) {
            const f32 value = finalHidden[square * hiddenSize + hiddenIndex];
            mean += value;
            maximum = std::max(maximum, value);
        }
        pooled[hiddenIndex] = mean / static_cast<f32>(SQUARE_COUNT);
        pooled[hiddenSize + hiddenIndex] = maximum;
        pooled[2 * hiddenSize + hiddenIndex] =
            finalHidden[features.white_king_square * hiddenSize + hiddenIndex];
        pooled[3 * hiddenSize + hiddenIndex] =
            finalHidden[features.black_king_square * hiddenSize + hiddenIndex];
    }
    for (std::size_t featureIndex = 0; featureIndex < GLOBAL_FEATURE_COUNT; ++featureIndex)
        pooled[4 * hiddenSize + featureIndex] = features.global[featureIndex];

    std::array<f32, 64> valueHidden{};
    for (std::size_t outputIndex = 0; outputIndex < valueHidden.size(); ++outputIndex) {
        f32 sum = 0.0f;
        for (std::size_t inputIndex = 0; inputIndex < pooled.size(); ++inputIndex)
            sum += pooled[inputIndex]
                * parameters.value_hidden_weight[outputIndex * pooled.size() + inputIndex];
        valueHidden[outputIndex] = relu(sum + parameters.value_hidden_bias[outputIndex]);
    }
    f32 output = parameters.value_output_bias[0];
    for (std::size_t inputIndex = 0; inputIndex < valueHidden.size(); ++inputIndex)
        output += valueHidden[inputIndex] * parameters.value_output_weight[inputIndex];
    if constexpr (Profile)
        profile->pooling_and_value_head_us += std::chrono::duration<double, std::micro>(
            Clock::now() - stageStarted).count();
    return output;
}

f32 GraphEvaluator::evaluate_residual(const GraphFeatures& features) const {
    return evaluate(features) * model_metadata.residual_multiplier;
}

f32 GraphEvaluator::evaluate_residual_reordered(const GraphFeatures& features) const {
    return evaluate_reordered(features) * model_metadata.residual_multiplier;
}

} // namespace GNN