#include <bermuda_tuner/dataset.h>
#include "position.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
namespace Bermuda::Tuner {

bool Dataset::validate(std::string& error) const {
    if (samples.empty() || game_ids.empty()) {
        error = "dataset must contain games and positions";
        return false;
    }
    if (!games.empty() && games.size() != game_ids.size()) {
        error = "dataset metadata count does not match game IDs";
        return false;
    }

    std::unordered_map<std::string, std::size_t> game_indices;
    for (std::size_t i = 0; i < game_ids.size(); ++i) {
        if (game_ids[i].empty()
            || !game_indices.emplace(game_ids[i], i).second) {
            error = "dataset game IDs must be non-empty and unique";
            return false;
        }
        if (!games.empty() && games[i].id != game_ids[i]) {
            error = "dataset metadata ID does not match its game ID";
            return false;
        }
    }

    std::vector<double> game_results(game_ids.size(), -1.0);
    for (const Sample& sample : samples) {
        if (sample.game >= game_ids.size()) {
            error = "sample references a game outside the dataset";
            return false;
        }
        if (!std::isfinite(sample.result)
            || (sample.result != 0.0 && sample.result != 0.5
                && sample.result != 1.0)) {
            error = "sample result must be 0, 0.5, or 1";
            return false;
        }
        double& expected_result = game_results[sample.game];
        if (expected_result < 0.0)
            expected_result = sample.result;
        else if (expected_result != sample.result) {
            error = "samples from one game have conflicting results";
            return false;
        }
    }
    if (std::find(game_results.begin(), game_results.end(), -1.0)
        != game_results.end()) {
        error = "dataset contains a game with no positions";
        return false;
    }

    error.clear();
    return true;
}

namespace {
const std::string* find_metadata_value(const GameMetadata& game,
                                       const std::string& field);

void populate_statistics(const Dataset& dataset, ImportStatistics& statistics) {
    statistics.games_accepted = dataset.game_ids.size();
    statistics.positions_generated = dataset.samples.size();
    std::vector<std::size_t> positions_per_game(dataset.game_ids.size(), 0);
    for (const Sample& sample : dataset.samples)
        ++positions_per_game[sample.game];
    if (!positions_per_game.empty()) {
        std::sort(positions_per_game.begin(), positions_per_game.end());
        statistics.minimum_positions_per_game = positions_per_game.front();
        statistics.maximum_positions_per_game = positions_per_game.back();
        const std::size_t middle = positions_per_game.size() / 2;
        statistics.median_positions_per_game = positions_per_game.size() % 2
            ? static_cast<double>(positions_per_game[middle])
            : (positions_per_game[middle - 1]
               + positions_per_game[middle]) / 2.0;
        statistics.mean_positions_per_game =
            static_cast<double>(dataset.samples.size()) / positions_per_game.size();
    }

    static const char* const tracked_fields[] = {
        "Event", "Site", "Date", "Round", "White", "Black", "ECO",
        "Opening", "TimeControl", "Termination", "Variant"
    };
    for (const GameMetadata& game : dataset.games) {
        for (const char* field : tracked_fields) {
            const std::string* value = find_metadata_value(game, field);
            if (!value || value->empty())
                ++statistics.missing_metadata[field];
        }
    }
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

const std::string* find_metadata_value(const GameMetadata& game,
                                       const std::string& field) {
    const std::string normalized = lowercase(field);
    if (normalized == "result") return &game.result;
    if (normalized == "source") return &game.source;
    for (const auto& tag : game.tags) {
        if (lowercase(tag.first) == normalized)
            return &tag.second;
    }
    if (normalized == "event") return &game.event;
    if (normalized == "site") return &game.site;
    if (normalized == "date") return &game.date;
    if (normalized == "round") return &game.round;
    if (normalized == "white") return &game.white;
    if (normalized == "black") return &game.black;
    if (normalized == "eco") return &game.eco;
    if (normalized == "opening") return &game.opening;
    if (normalized == "timecontrol") return &game.time_control;
    if (normalized == "termination") return &game.termination;
    if (normalized == "variant") return &game.variant;
    return nullptr;
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    for (const unsigned char ch : value) {
        switch (ch) {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (ch < 0x20) {
                constexpr char digits[] = "0123456789abcdef";
                escaped += "\\u00";
                escaped += digits[ch >> 4];
                escaped += digits[ch & 0x0f];
            } else {
                escaped += static_cast<char>(ch);
            }
        }
    }
    return escaped;
}

std::string encode_field(const std::string& value) {
    constexpr char digits[] = "0123456789ABCDEF";
    std::string encoded;
    for (const unsigned char ch : value) {
        if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z')
            || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_'
            || ch == '.') {
            encoded += static_cast<char>(ch);
        } else {
            encoded += '%';
            encoded += digits[ch >> 4];
            encoded += digits[ch & 0x0f];
        }
    }
    return encoded;
}

bool decode_field(const std::string& encoded, std::string& value) {
    auto hex_value = [](char ch) -> int {
        if (ch >= '0' && ch <= '9') return ch - '0';
        if (ch >= 'a' && ch <= 'f') return ch - 'a' + 10;
        if (ch >= 'A' && ch <= 'F') return ch - 'A' + 10;
        return -1;
    };
    value.clear();
    for (std::size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] != '%') {
            value += encoded[i];
            continue;
        }
        if (i + 2 >= encoded.size())
            return false;
        const int high = hex_value(encoded[i + 1]);
        const int low = hex_value(encoded[i + 2]);
        if (high < 0 || low < 0)
            return false;
        value += static_cast<char>((high << 4) | low);
        i += 2;
    }
    return true;
}

bool finalize_file(const std::string& temporary, const std::string& path,
                   std::string& error) {
    const std::string backup = path + ".bak";
    std::error_code fs_error;
    std::filesystem::remove(backup, fs_error);
    fs_error.clear();
    if (std::filesystem::exists(path)) {
        std::filesystem::rename(path, backup, fs_error);
        if (fs_error) {
            error = "unable to preserve previous file: " + fs_error.message();
            return false;
        }
    }
    fs_error.clear();
    std::filesystem::rename(temporary, path, fs_error);
    if (fs_error) {
        error = "unable to finalize file: " + fs_error.message();
        if (std::filesystem::exists(backup)) {
            std::error_code restore_error;
            std::filesystem::rename(backup, path, restore_error);
        }
        return false;
    }
    error.clear();
    return true;
}

std::string get_tag(const std::map<std::string, std::string>& tags,
                    const std::string& key) {
    const auto found = tags.find(key);
    return found == tags.end() ? std::string{} : found->second;
}

bool parse_result(const std::string& token, double& result) {
    if (token == "1-0" || token == "1") {
        result = 1.0;
        return true;
    }
    if (token == "1/2-1/2" || token == "0.5") {
        result = 0.5;
        return true;
    }
    if (token == "0-1" || token == "0") {
        result = 0.0;
        return true;
    }
    return false;
}

bool load_dataset_impl(const std::string& path, std::vector<Sample>& samples,
                       std::vector<std::string>& game_ids,
                       std::vector<GameMetadata>& games,
                       std::string& source_description,
                       std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "unable to open dataset: " + path;
        return false;
    }

    std::unordered_map<std::string, std::size_t> game_indices;
    std::vector<double> game_results;
    std::unordered_map<std::string, std::map<std::string, std::string>>
        metadata_by_id;
    std::unordered_map<std::string, std::string> result_by_id;
    std::string line;
    i32 line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (line.rfind("#@source\t", 0) == 0) {
            std::string decoded;
            if (!decode_field(line.substr(9), decoded)) {
                error = "malformed source metadata on dataset row "
                      + std::to_string(line_number);
                return false;
            }
            source_description = std::move(decoded);
            continue;
        }
        if (line.rfind("#@result\t", 0) == 0) {
            const std::size_t separator = line.find('\t', 9);
            std::string id;
            std::string result;
            if (separator == std::string::npos
                || !decode_field(line.substr(9, separator - 9), id)
                || !decode_field(line.substr(separator + 1), result)
                || id.empty()) {
                error = "malformed result metadata on dataset row "
                      + std::to_string(line_number);
                return false;
            }
            result_by_id[id] = std::move(result);
            continue;
        }
        if (line.rfind("#@meta\t", 0) == 0) {
            const std::size_t id_end = line.find('\t', 7);
            const std::size_t key_end = id_end == std::string::npos
                ? std::string::npos : line.find('\t', id_end + 1);
            std::string id;
            std::string key;
            std::string value;
            if (id_end == std::string::npos || key_end == std::string::npos
                || !decode_field(line.substr(7, id_end - 7), id)
                || !decode_field(line.substr(id_end + 1,
                                              key_end - id_end - 1), key)
                || !decode_field(line.substr(key_end + 1), value)
                || id.empty() || key.empty()) {
                error = "malformed game metadata on dataset row "
                      + std::to_string(line_number);
                return false;
            }
            metadata_by_id[id][key] = value;
            continue;
        }
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos)
            line.erase(comment);
        std::istringstream parser(line);
        std::string game_id;
        std::string result_token;
        if (!(parser >> game_id))
            continue;

        std::string fen;
        if (!(parser >> result_token) || !std::getline(parser, fen)) {
            error = "malformed dataset row " + std::to_string(line_number)
                  + " (expected GAME_ID RESULT FEN)";
            return false;
        }
        const std::size_t first = fen.find_first_not_of(" \t");
        if (first == std::string::npos) {
            error = "missing FEN on dataset row "
                  + std::to_string(line_number);
            return false;
        }
        fen.erase(0, first);

        double result;
        if (!parse_result(result_token, result)) {
            error = "invalid result on dataset row "
                  + std::to_string(line_number) + ": " + result_token;
            return false;
        }

        try {
            Position position = Position::from_fen(fen);
            auto insertion = game_indices.emplace(game_id, game_ids.size());
            if (insertion.second) {
                game_ids.push_back(game_id);
                game_results.push_back(result);
            } else if (game_results[insertion.first->second] != result) {
                error = "conflicting results for game " + game_id
                      + " on dataset row " + std::to_string(line_number);
                return false;
            }
            samples.push_back({ position, result, insertion.first->second });
        } catch (const std::exception& exception) {
            error = "invalid FEN on dataset row "
                  + std::to_string(line_number) + ": " + exception.what();
            return false;
        }
    }
    if (!input.eof()) {
        error = "error reading dataset: " + path;
        return false;
    }
    if (samples.size() < 2 || game_ids.size() < 2) {
        error = "dataset needs positions from at least two games";
        return false;
    }
    for (const auto& metadata : metadata_by_id) {
        if (game_indices.find(metadata.first) == game_indices.end()) {
            error = "metadata references unknown game ID: " + metadata.first;
            return false;
        }
    }
    for (const auto& result : result_by_id) {
        if (game_indices.find(result.first) == game_indices.end()) {
            error = "result metadata references unknown game ID: "
                  + result.first;
            return false;
        }
    }
    if (source_description.empty())
        source_description = path;
    for (const std::string& id : game_ids) {
        GameMetadata game;
        game.id = id;
        game.source = source_description;
        const auto result = result_by_id.find(id);
        if (result != result_by_id.end())
            game.result = result->second;
        else {
            const double numeric_result = game_results[game_indices.at(id)];
            game.result = numeric_result == 1.0 ? "1-0"
                : numeric_result == 0.5 ? "1/2-1/2" : "0-1";
        }
        const auto metadata = metadata_by_id.find(id);
        if (metadata != metadata_by_id.end())
            game.tags = metadata->second;
        game.event = get_tag(game.tags, "Event");
        game.site = get_tag(game.tags, "Site");
        game.date = get_tag(game.tags, "Date");
        game.round = get_tag(game.tags, "Round");
        game.white = get_tag(game.tags, "White");
        game.black = get_tag(game.tags, "Black");
        if (game.result.empty())
            game.result = get_tag(game.tags, "Result");
        game.eco = get_tag(game.tags, "ECO");
        game.opening = get_tag(game.tags, "Opening");
        game.time_control = get_tag(game.tags, "TimeControl");
        game.termination = get_tag(game.tags, "Termination");
        game.variant = get_tag(game.tags, "Variant");
        games.push_back(std::move(game));
    }
    error.clear();
    return true;
}

bool write_prepared_datasets_impl(const Config& config,
                                  const Dataset& dataset,
                                  std::string& error) {
    if (dataset.game_ids.size() < 4) {
        error = "preparation needs at least four completed games so both splits "
                "contain multiple games";
        return false;
    }
    std::mt19937_64 random(config.split_seed);
    std::vector<std::size_t> shuffled_games(dataset.game_ids.size());
    for (std::size_t i = 0; i < shuffled_games.size(); ++i)
        shuffled_games[i] = i;
    std::shuffle(shuffled_games.begin(), shuffled_games.end(), random);

    const std::size_t requested_validation_count = std::max<std::size_t>(
        2, (dataset.game_ids.size() * static_cast<std::size_t>(config.validation_percent)
            + 99) / 100);
    const std::size_t validation_count = std::min(
        requested_validation_count, dataset.game_ids.size() - 2);
    std::vector<bool> validation_game(dataset.game_ids.size(), false);
    for (std::size_t i = 0; i < validation_count; ++i)
        validation_game[shuffled_games[i]] = true;

    Dataset train_dataset;
    Dataset validation_dataset;
    train_dataset.source_description = dataset.source_description;
    validation_dataset.source_description = dataset.source_description;
    std::vector<std::size_t> train_game_map(dataset.game_ids.size());
    std::vector<std::size_t> validation_game_map(dataset.game_ids.size());
    for (std::size_t game = 0; game < dataset.game_ids.size(); ++game) {
        Dataset& output = validation_game[game]
            ? validation_dataset : train_dataset;
        std::vector<std::size_t>& mapping = validation_game[game]
            ? validation_game_map : train_game_map;
        mapping[game] = output.game_ids.size();
        output.game_ids.push_back(dataset.game_ids[game]);
        if (!dataset.games.empty())
            output.games.push_back(dataset.games[game]);
        else {
            GameMetadata metadata;
            metadata.id = dataset.game_ids[game];
            metadata.source = dataset.source_description;
            output.games.push_back(std::move(metadata));
        }
    }
    for (const Sample& sample : dataset.samples) {
        Dataset& output = validation_game[sample.game]
            ? validation_dataset : train_dataset;
        const auto& mapping = validation_game[sample.game]
            ? validation_game_map : train_game_map;
        Sample copy = sample;
        copy.game = mapping[sample.game];
        output.samples.push_back(std::move(copy));
    }
    if (config.format == "btd") {
        if (!save_binary_dataset(config.train_output_path, train_dataset, error)
            || !save_binary_dataset(config.validation_output_path,
                                    validation_dataset, error))
            return false;
    } else if (config.format != "text") {
        error = "unsupported dataset format: " + config.format;
        return false;
    } else {
    const std::string train_temp = config.train_output_path + ".tmp";
    const std::string validation_temp = config.validation_output_path + ".tmp";
    std::ofstream train(train_temp, std::ios::trunc);
    std::ofstream validation(validation_temp, std::ios::trunc);
    if (!train || !validation) {
        error = "unable to create prepared dataset output files";
        return false;
    }

    const auto write_text = [](std::ostream& output,
                               const Dataset& split) {
        output << "#@source\t" << encode_field(split.source_description)
               << '\n';
        for (std::size_t game_index = 0;
             game_index < split.game_ids.size(); ++game_index) {
            std::map<std::string, std::string> tags =
                split.games.empty() ? std::map<std::string, std::string>{}
                                    : split.games[game_index].tags;
            const GameMetadata& metadata = split.games.empty()
                ? GameMetadata{} : split.games[game_index];
            output << "#@result\t"
                   << encode_field(split.game_ids[game_index]) << '\t'
                   << encode_field(metadata.result) << '\n';
            const auto preserve_field = [&tags](const std::string& key,
                                                const std::string& value) {
                if (!value.empty())
                    tags.emplace(key, value);
            };
            preserve_field("Event", metadata.event);
            preserve_field("Site", metadata.site);
            preserve_field("Date", metadata.date);
            preserve_field("Round", metadata.round);
            preserve_field("White", metadata.white);
            preserve_field("Black", metadata.black);
            preserve_field("Result", metadata.result);
            preserve_field("ECO", metadata.eco);
            preserve_field("Opening", metadata.opening);
            preserve_field("TimeControl", metadata.time_control);
            preserve_field("Termination", metadata.termination);
            preserve_field("Variant", metadata.variant);
            for (const auto& tag : tags) {
                output << "#@meta\t" << encode_field(split.game_ids[game_index])
                       << '\t' << encode_field(tag.first)
                       << '\t' << encode_field(tag.second) << '\n';
            }
        }
        for (const Sample& sample : split.samples) {
            output << split.game_ids[sample.game] << ' '
               << (sample.result == 1.0 ? "1-0"
                   : sample.result == 0.5 ? "1/2-1/2" : "0-1")
               << ' ' << sample.position.to_fen() << '\n';
        }
    };
    write_text(train, train_dataset);
    write_text(validation, validation_dataset);
    train.close();
    validation.close();
    if (!train || !validation) {
        error = "failed writing prepared datasets";
        std::filesystem::remove(train_temp);
        std::filesystem::remove(validation_temp);
        return false;
    }

    if (!finalize_file(train_temp, config.train_output_path, error)) {
        return false;
    }
    if (!finalize_file(validation_temp, config.validation_output_path, error)) {
        return false;
    }
    }

    const std::string manifest_path = config.train_output_path
                                    + ".manifest.json";
    const std::string manifest_temp = manifest_path + ".tmp";
    std::ofstream manifest(manifest_temp, std::ios::trunc);
    if (!manifest) {
        error = "unable to create split manifest: " + manifest_temp;
        return false;
    }
    manifest << "{\n"
             << "  \"format_version\": 1,\n"
             << "  \"split_policy\": \"game-level-random\",\n"
             << "  \"split_seed\": " << config.split_seed << ",\n"
             << "  \"validation_percent\": " << config.validation_percent << ",\n"
             << "  \"train_path\": \"" << json_escape(config.train_output_path)
             << "\",\n"
             << "  \"validation_path\": \""
             << json_escape(config.validation_output_path) << "\",\n"
             << "  \"train_games\": " << train_dataset.game_ids.size() << ",\n"
             << "  \"validation_games\": "
             << validation_dataset.game_ids.size() << ",\n"
             << "  \"train_positions\": " << train_dataset.samples.size() << ",\n"
             << "  \"validation_positions\": "
             << validation_dataset.samples.size() << ",\n"
             << "  \"train_fingerprint\": "
             << dataset_fingerprint(train_dataset) << ",\n"
             << "  \"validation_fingerprint\": "
             << dataset_fingerprint(validation_dataset)
             << ",\n  \"train_game_ids\": [";
    for (std::size_t i = 0; i < train_dataset.game_ids.size(); ++i) {
        if (i != 0)
            manifest << ", ";
        manifest << '"' << json_escape(train_dataset.game_ids[i]) << '"';
    }
    manifest << "],\n  \"validation_game_ids\": [";
    for (std::size_t i = 0; i < validation_dataset.game_ids.size(); ++i) {
        if (i != 0)
            manifest << ", ";
        manifest << '"' << json_escape(validation_dataset.game_ids[i]) << '"';
    }
    manifest << "]\n}\n";
    manifest.close();
    if (!manifest) {
        error = "failed writing split manifest: " + manifest_temp;
        return false;
    }
    if (!finalize_file(manifest_temp, manifest_path, error)) {
        return false;
    }
    error.clear();
    return true;
}

std::uint64_t dataset_fingerprint_impl(const Dataset& dataset) {
    std::uint64_t hash = 14695981039346656037ULL;
    auto add = [&hash](const std::string& value) {
        for (const unsigned char ch : value) {
            hash ^= ch;
            hash *= 1099511628211ULL;
        }
        hash ^= 0xff;
        hash *= 1099511628211ULL;
    };
    add("BermudaDatasetIdentity1");
    add(dataset.source_description);
    add(std::to_string(dataset.game_ids.size()));
    const GameMetadata empty_metadata;
    for (std::size_t game = 0; game < dataset.game_ids.size(); ++game) {
        add(dataset.game_ids[game]);
        const GameMetadata& metadata = dataset.games.empty()
            ? empty_metadata : dataset.games[game];
        add(metadata.event);
        add(metadata.site);
        add(metadata.date);
        add(metadata.round);
        add(metadata.white);
        add(metadata.black);
        add(metadata.result);
        add(metadata.eco);
        add(metadata.opening);
        add(metadata.time_control);
        add(metadata.termination);
        add(metadata.variant);
        add(metadata.source);
        add(std::to_string(metadata.tags.size()));
        for (const auto& tag : metadata.tags) {
            add(tag.first);
            add(tag.second);
        }
    }
    add(std::to_string(dataset.samples.size()));
    for (const Sample& sample : dataset.samples) {
        add(dataset.game_ids[sample.game]);
        add(sample.position.to_fen());
        add(std::to_string(sample.result));
    }
    return hash;
}


} // namespace
bool load_dataset(const std::string& path, Dataset& dataset, std::string& error) {
    ImportStatistics statistics;
    return load_dataset(path, dataset, statistics, error);
}
bool load_dataset(const std::string& path, Dataset& dataset,
                  ImportStatistics& statistics, std::string& error) {
    Dataset candidate;
    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
        [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    bool binary_format = extension == ".btd";
    if (!binary_format) {
        std::ifstream probe(path, std::ios::binary);
        char magic[4] = {};
        probe.read(magic, sizeof(magic));
        binary_format = probe.gcount() == 4
                     && std::string(magic, sizeof(magic)) == "BMTD";
    }
    if (binary_format) {
        if (!load_binary_dataset(path, candidate, error))
            return false;
    } else {
        if (!load_dataset_impl(path, candidate.samples, candidate.game_ids,
                               candidate.games, candidate.source_description,
                               error))
            return false;
    }
    if (!candidate.validate(error))
        return false;
    statistics.games_encountered = candidate.game_ids.size();
    populate_statistics(candidate, statistics);
    dataset = std::move(candidate);
    return true;
}
bool write_prepared_datasets(const Config& config, const Dataset& dataset, std::string& error) {
    if (config.train_output_path.empty()
        || config.validation_output_path.empty()
        || config.train_output_path == config.validation_output_path
        || config.validation_percent <= 0 || config.validation_percent >= 100
        || (config.format != "text" && config.format != "btd")) {
        error = "prepared dataset output configuration is invalid";
        return false;
    }
    if (!dataset.validate(error))
        return false;
    return write_prepared_datasets_impl(config, dataset, error);
}
std::uint64_t dataset_fingerprint(const Dataset& dataset) {
    return dataset_fingerprint_impl(dataset);
}

bool filter_dataset_metadata(
    Dataset& dataset,
    const std::vector<std::pair<std::string, std::string>>& metadata_filters,
    ImportStatistics& statistics, std::string& error) {
    if (!dataset.validate(error))
        return false;
    if (metadata_filters.empty()) {
        populate_statistics(dataset, statistics);
        error.clear();
        return true;
    }
    if (dataset.games.size() != dataset.game_ids.size()) {
        error = "metadata filters require preserved game metadata";
        return false;
    }
    std::vector<bool> keep(dataset.games.size(), true);
    for (std::size_t game = 0; game < dataset.games.size(); ++game) {
        for (const auto& filter : metadata_filters) {
            const std::string* actual = find_metadata_value(
                dataset.games[game], filter.first);
            if (!actual || *actual != filter.second) {
                keep[game] = false;
                break;
            }
        }
    }
    Dataset filtered;
    filtered.source_description = dataset.source_description;
    std::vector<std::size_t> remap(dataset.games.size(), dataset.games.size());
    for (std::size_t game = 0; game < dataset.games.size(); ++game) {
        if (!keep[game])
            continue;
        remap[game] = filtered.game_ids.size();
        filtered.game_ids.push_back(dataset.game_ids[game]);
        filtered.games.push_back(dataset.games[game]);
    }
    for (const Sample& sample : dataset.samples) {
        if (!keep[sample.game])
            continue;
        Sample copy = sample;
        copy.game = remap[sample.game];
        filtered.samples.push_back(std::move(copy));
    }
    const std::size_t filtered_out = dataset.game_ids.size()
                                   - filtered.game_ids.size();
    statistics.games_skipped += filtered_out;
    statistics.skip_reasons["metadata_filter"] += filtered_out;
    if (!filtered.samples.empty() && !filtered.game_ids.empty()) {
        if (!filtered.validate(error))
            return false;
        dataset = std::move(filtered);
        populate_statistics(dataset, statistics);
        error.clear();
        return true;
    }
    error = "metadata filters matched no usable games";
    return false;
}
} // namespace Bermuda::Tuner
