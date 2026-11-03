#include <bermuda_tuner/dataset.h>

#include "position.h"

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace Bermuda::Tuner {
namespace {

constexpr std::uint32_t BinaryDatasetVersion = 1;
constexpr std::uint64_t MaximumPayloadSize = 1ULL * 1024 * 1024 * 1024;
constexpr std::uint64_t MaximumGames = 1000000;
constexpr std::uint64_t MaximumSamples = 10000000;
constexpr std::uint64_t MaximumStringSize = 1024 * 1024;
static_assert(sizeof(double) == sizeof(std::uint64_t),
              "binary dataset requires 64-bit double values");

void append_u32(std::vector<unsigned char>& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8)
        output.push_back(static_cast<unsigned char>(value >> shift));
}

void append_u64(std::vector<unsigned char>& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8)
        output.push_back(static_cast<unsigned char>(value >> shift));
}

void append_string(std::vector<unsigned char>& output, const std::string& value) {
    append_u64(output, value.size());
    output.insert(output.end(), value.begin(), value.end());
}

std::uint64_t checksum(const std::vector<unsigned char>& bytes) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char byte : bytes) {
        hash ^= byte;
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool write_u32(std::ostream& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8)
        output.put(static_cast<char>(value >> shift));
    return static_cast<bool>(output);
}

bool write_u64(std::ostream& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8)
        output.put(static_cast<char>(value >> shift));
    return static_cast<bool>(output);
}

class PayloadReader {
public:
    explicit PayloadReader(const std::vector<unsigned char>& bytes)
        : bytes_(bytes) {}

    bool u32(std::uint32_t& value) {
        if (remaining() < 4)
            return false;
        value = 0;
        for (unsigned int i = 0; i < 4; ++i)
            value |= static_cast<std::uint32_t>(bytes_[offset_++]) << (8 * i);
        return true;
    }

    bool u64(std::uint64_t& value) {
        if (remaining() < 8)
            return false;
        value = 0;
        for (unsigned int i = 0; i < 8; ++i)
            value |= static_cast<std::uint64_t>(bytes_[offset_++]) << (8 * i);
        return true;
    }

    bool string(std::string& value) {
        std::uint64_t size = 0;
        if (!u64(size) || size > MaximumStringSize || size > remaining())
            return false;
        value.assign(reinterpret_cast<const char*>(bytes_.data() + offset_),
                     static_cast<std::size_t>(size));
        offset_ += static_cast<std::size_t>(size);
        return true;
    }

    std::size_t remaining() const {
        return bytes_.size() - offset_;
    }

private:
    const std::vector<unsigned char>& bytes_;
    std::size_t offset_ = 0;
};

bool write_binary_impl(const std::string& path, const Dataset& dataset,
                       std::string& error) {
    if (path.empty()) {
        error = "binary dataset path must not be empty";
        return false;
    }
    if (!dataset.validate(error))
        return false;
    if (!std::numeric_limits<double>::is_iec559) {
        error = "binary datasets require IEEE-754 double precision";
        return false;
    }

    std::vector<unsigned char> payload;
    append_string(payload, dataset.source_description);
    append_u64(payload, dataset.game_ids.size());
    for (std::size_t i = 0; i < dataset.game_ids.size(); ++i) {
        const GameMetadata empty_metadata;
        const GameMetadata& game = dataset.games.empty()
            ? empty_metadata : dataset.games[i];
        append_string(payload, dataset.game_ids[i]);
        append_string(payload, game.event);
        append_string(payload, game.site);
        append_string(payload, game.date);
        append_string(payload, game.round);
        append_string(payload, game.white);
        append_string(payload, game.black);
        append_string(payload, game.result);
        append_string(payload, game.eco);
        append_string(payload, game.opening);
        append_string(payload, game.time_control);
        append_string(payload, game.termination);
        append_string(payload, game.variant);
        append_string(payload, game.source);
        append_u64(payload, game.tags.size());
        for (const auto& tag : game.tags) {
            append_string(payload, tag.first);
            append_string(payload, tag.second);
        }
        if (payload.size() > MaximumPayloadSize) {
            error = "binary dataset exceeds the supported size limit";
            return false;
        }
    }
    append_u64(payload, dataset.samples.size());
    for (const Sample& sample : dataset.samples) {
        append_u64(payload, sample.game);
        std::uint64_t result_bits = 0;
        std::memcpy(&result_bits, &sample.result, sizeof(result_bits));
        append_u64(payload, result_bits);
        append_string(payload, sample.position.to_fen());
        if (payload.size() > MaximumPayloadSize) {
            error = "binary dataset exceeds the supported size limit";
            return false;
        }
    }

    const std::string temporary = path + ".tmp";
    const std::string backup = path + ".bak";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "unable to create binary dataset: " + temporary;
        return false;
    }
    output.write("BMTD", 4);
    if (!write_u32(output, BinaryDatasetVersion)
        || !write_u64(output, payload.size())
        || !write_u64(output, checksum(payload))
        || !write_u64(output, dataset.game_ids.size())
        || !write_u64(output, dataset.samples.size())
        || !write_u64(output, dataset_fingerprint(dataset))) {
        error = "failed writing binary dataset header: " + temporary;
        return false;
    }
    output.write(reinterpret_cast<const char*>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
    output.close();
    if (!output) {
        error = "failed writing binary dataset: " + temporary;
        return false;
    }

    std::error_code fs_error;
    std::filesystem::remove(backup, fs_error);
    fs_error.clear();
    if (std::filesystem::exists(path)) {
        std::filesystem::rename(path, backup, fs_error);
        if (fs_error) {
            error = "unable to preserve previous binary dataset: "
                  + fs_error.message();
            return false;
        }
    }
    fs_error.clear();
    std::filesystem::rename(temporary, path, fs_error);
    if (fs_error) {
        error = "unable to finalize binary dataset: " + fs_error.message();
        if (std::filesystem::exists(backup)) {
            std::error_code restore_error;
            std::filesystem::rename(backup, path, restore_error);
        }
        return false;
    }
    error.clear();
    return true;
}

bool load_binary_impl(const std::string& path, Dataset& dataset,
                      std::string& error) {
    if (path.empty()) {
        error = "binary dataset path must not be empty";
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "unable to open binary dataset: " + path;
        return false;
    }
    input.seekg(0, std::ios::end);
    const std::streamoff file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    constexpr std::streamoff HeaderSize = 48;
    if (file_size < HeaderSize
        || static_cast<std::uint64_t>(file_size - HeaderSize)
               > MaximumPayloadSize) {
        error = "invalid or oversized binary dataset: " + path;
        return false;
    }
    std::array<char, 4> magic{};
    std::uint32_t version = 0;
    std::uint64_t payload_size = 0;
    std::uint64_t expected_checksum = 0;
    std::uint64_t expected_games = 0;
    std::uint64_t expected_samples = 0;
    std::uint64_t expected_fingerprint = 0;
    input.read(magic.data(), magic.size());
    auto read_u32 = [&input](std::uint32_t& value) {
        value = 0;
        for (unsigned int i = 0; i < 4; ++i) {
            const int byte = input.get();
            if (byte == std::char_traits<char>::eof())
                return false;
            value |= static_cast<std::uint32_t>(
                         static_cast<unsigned char>(byte)) << (8 * i);
        }
        return true;
    };
    auto read_u64 = [&input](std::uint64_t& value) {
        value = 0;
        for (unsigned int i = 0; i < 8; ++i) {
            const int byte = input.get();
            if (byte == std::char_traits<char>::eof())
                return false;
            value |= static_cast<std::uint64_t>(
                         static_cast<unsigned char>(byte)) << (8 * i);
        }
        return true;
    };
    if (!input || magic != std::array<char, 4>{{'B', 'M', 'T', 'D'}}
        || !read_u32(version) || !read_u64(payload_size)
        || !read_u64(expected_checksum) || !read_u64(expected_games)
        || !read_u64(expected_samples) || !read_u64(expected_fingerprint)) {
        error = "invalid binary dataset header: " + path;
        return false;
    }
    if (version != BinaryDatasetVersion) {
        error = "unsupported binary dataset version "
              + std::to_string(version);
        return false;
    }
    if (payload_size != static_cast<std::uint64_t>(file_size - HeaderSize)) {
        error = "binary dataset length does not match its header: " + path;
        return false;
    }
    std::vector<unsigned char> payload(static_cast<std::size_t>(payload_size));
    input.read(reinterpret_cast<char*>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
    if (!input || checksum(payload) != expected_checksum) {
        error = "binary dataset is truncated or failed its checksum: " + path;
        return false;
    }

    PayloadReader reader(payload);
    Dataset candidate;
    if (!reader.string(candidate.source_description)) {
        error = "invalid binary dataset source metadata";
        return false;
    }
    std::uint64_t game_count = 0;
    if (!reader.u64(game_count) || game_count == 0
        || game_count > MaximumGames
        || game_count > reader.remaining() / 120) {
        error = "invalid game count in binary dataset";
        return false;
    }
    candidate.game_ids.reserve(static_cast<std::size_t>(game_count));
    candidate.games.reserve(static_cast<std::size_t>(game_count));
    for (std::uint64_t i = 0; i < game_count; ++i) {
        GameMetadata game;
        if (!reader.string(game.id)
            || !reader.string(game.event) || !reader.string(game.site)
            || !reader.string(game.date) || !reader.string(game.round)
            || !reader.string(game.white) || !reader.string(game.black)
            || !reader.string(game.result) || !reader.string(game.eco)
            || !reader.string(game.opening)
            || !reader.string(game.time_control)
            || !reader.string(game.termination)
            || !reader.string(game.variant) || !reader.string(game.source)) {
            error = "truncated game metadata in binary dataset";
            return false;
        }
        std::uint64_t tag_count = 0;
        if (!reader.u64(tag_count) || tag_count > 10000) {
            error = "invalid PGN tag count in binary dataset";
            return false;
        }
        for (std::uint64_t tag = 0; tag < tag_count; ++tag) {
            std::string key;
            std::string value;
            if (!reader.string(key) || !reader.string(value)
                || !game.tags.emplace(std::move(key), std::move(value)).second) {
                error = "invalid PGN tags in binary dataset";
                return false;
            }
        }
        candidate.game_ids.push_back(game.id);
        candidate.games.push_back(std::move(game));
    }
    std::uint64_t sample_count = 0;
    if (!reader.u64(sample_count) || sample_count == 0
        || sample_count > MaximumSamples
        || sample_count > reader.remaining() / 24) {
        error = "invalid position count in binary dataset";
        return false;
    }
    candidate.samples.reserve(static_cast<std::size_t>(sample_count));
    for (std::uint64_t i = 0; i < sample_count; ++i) {
        std::uint64_t game_index = 0;
        std::uint64_t result_bits = 0;
        std::string fen;
        if (!reader.u64(game_index) || !reader.u64(result_bits)
            || !reader.string(fen) || game_index >= game_count) {
            error = "invalid position record in binary dataset";
            return false;
        }
        double result = 0.0;
        std::memcpy(&result, &result_bits, sizeof(result));
        try {
            candidate.samples.push_back({
                Position::from_fen(fen), result,
                static_cast<std::size_t>(game_index)
            });
        } catch (const std::exception& exception) {
            error = "invalid FEN in binary dataset: "
                  + std::string(exception.what());
            return false;
        }
    }
    if (reader.remaining() != 0 || !candidate.validate(error)) {
        if (error.empty())
            error = "unexpected trailing data in binary dataset";
        return false;
    }
    if (candidate.game_ids.size() != expected_games
        || candidate.samples.size() != expected_samples
        || dataset_fingerprint(candidate) != expected_fingerprint) {
        error = "binary dataset identity does not match its header";
        return false;
    }
    dataset = std::move(candidate);
    error.clear();
    return true;
}

} // namespace

bool save_binary_dataset(const std::string& path, const Dataset& dataset,
                         std::string& error) {
    return write_binary_impl(path, dataset, error);
}

bool load_binary_dataset(const std::string& path, Dataset& dataset,
                         std::string& error) {
    return load_binary_impl(path, dataset, error);
}

} // namespace Bermuda::Tuner
