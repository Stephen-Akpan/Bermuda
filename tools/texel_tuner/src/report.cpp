#include <bermuda_tuner/dataset.h>
#include <bermuda_tuner/parameter_space.h>
#include <bermuda_tuner/report.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace Bermuda::Tuner {
namespace {

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

bool atomic_write(const std::string& path, const std::string& contents,
                  std::string& error) {
    const std::filesystem::path destination(path);
    std::error_code fs_error;
    if (!destination.parent_path().empty()) {
        std::filesystem::create_directories(destination.parent_path(), fs_error);
        if (fs_error) {
            error = "unable to create report directory: " + fs_error.message();
            return false;
        }
    }
    const std::string temporary = path + ".tmp";
    const std::string backup = path + ".bak";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        error = "unable to create report file: " + temporary;
        return false;
    }
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    output.close();
    if (!output) {
        error = "failed writing report file: " + temporary;
        return false;
    }

    std::filesystem::remove(backup, fs_error);
    fs_error.clear();
    if (std::filesystem::exists(destination)) {
        std::filesystem::rename(destination, backup, fs_error);
        if (fs_error) {
            error = "unable to preserve previous report file: "
                  + fs_error.message();
            return false;
        }
    }
    fs_error.clear();
    std::filesystem::rename(temporary, destination, fs_error);
    if (fs_error) {
        error = "unable to finalize report file: " + fs_error.message();
        if (std::filesystem::exists(backup)) {
            std::error_code restore_error;
            std::filesystem::rename(backup, destination, restore_error);
        }
        return false;
    }
    error.clear();
    return true;
}

Dataset split_dataset(const Dataset& source,
                     const std::vector<std::size_t>& indices) {
    Dataset split;
    split.source_description = source.source_description;
    std::vector<std::size_t> mapping(source.game_ids.size(),
                                     source.game_ids.size());
    for (const std::size_t sample_index : indices) {
        const std::size_t game = source.samples[sample_index].game;
        if (mapping[game] == source.game_ids.size()) {
            mapping[game] = split.game_ids.size();
            split.game_ids.push_back(source.game_ids[game]);
            if (!source.games.empty())
                split.games.push_back(source.games[game]);
            else {
                GameMetadata metadata;
                metadata.id = source.game_ids[game];
                split.games.push_back(std::move(metadata));
            }
        }
    }
    for (const std::size_t sample_index : indices) {
        Sample sample = source.samples[sample_index];
        sample.game = mapping[sample.game];
        split.samples.push_back(std::move(sample));
    }
    return split;
}

void write_parameter_array(std::ostream& output,
                           const ParameterVector& parameters) {
    output << '[';
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        if (i != 0)
            output << ", ";
        output << parameters[i];
    }
    output << ']';
}

void write_named_parameters(std::ostream& output,
                            const ParameterVector& parameters) {
    output << '{';
    const auto& descriptors = parameter_descriptors();
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        if (i != 0)
            output << ',';
        output << "\n    \"" << descriptors[i].name << "\": "
               << parameters[i];
    }
    output << "\n  }";
}

} // namespace

bool write_experiment_config(const Config& config, const Dataset& dataset,
                             const ImportStatistics& statistics,
                             const std::vector<std::size_t>& training,
                             const std::vector<std::size_t>& validation,
                             const ParameterVector& initial_weights,
                             std::string& error) {
    const auto created_at = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const std::filesystem::path config_path =
        std::filesystem::path(config.report_directory) / "config.json";
    if (config.resume && std::filesystem::exists(config_path)) {
        std::ifstream existing(config_path);
        if (!existing) {
            error = "unable to read existing experiment config: "
                  + config_path.string();
            return false;
        }
        const std::string stored((std::istreambuf_iterator<char>(existing)),
                                 std::istreambuf_iterator<char>());
        if (stored.empty() || stored.front() != '{') {
            error = "existing experiment config is missing or invalid: "
                  + config_path.string();
            return false;
        }
        std::ostringstream expected_identity;
        expected_identity << "\"dataset_fingerprint\": "
                          << dataset_fingerprint(dataset) << ',';
        if (stored.find(expected_identity.str()) == std::string::npos
            || stored.find("\"parameter_family\": \""
                           + json_escape(config.family) + "\"")
                   == std::string::npos
            || stored.find("\"weighting\": \""
                           + json_escape(config.weighting) + "\"")
                   == std::string::npos
            || stored.find("\"optimizer_seed\": "
                           + std::to_string(config.optimizer_seed) + ',')
                   == std::string::npos) {
            error = "existing experiment config does not match the checkpoint";
            return false;
        }
        error.clear();
        return true;
    }
    const Dataset train = split_dataset(dataset, training);
    const Dataset holdout = split_dataset(dataset, validation);
    const auto& descriptors = parameter_descriptors();
    std::ostringstream json;
    json << std::setprecision(17)
         << "{\n"
         << "  \"format_version\": 1,\n"
         << "  \"created_at_unix_ms\": " << created_at << ",\n"
         << "  \"experiment_id\": \"bermuda-" << created_at << "\",\n"
         << "  \"tuner_version\": \"1\",\n"
         << "  \"source_revision\": \"unavailable\",\n"
         << "  \"engine_evaluation_identity\": \"Bermuda Eval::Parameters\",\n"
         << "  \"source\": \"" << json_escape(dataset.source_description)
         << "\",\n"
         << "  \"dataset_fingerprint\": " << dataset_fingerprint(dataset)
         << ",\n"
         << "  \"train_fingerprint\": " << dataset_fingerprint(train) << ",\n"
         << "  \"validation_fingerprint\": " << dataset_fingerprint(holdout)
         << ",\n"
         << "  \"games\": " << dataset.game_ids.size() << ",\n"
         << "  \"positions\": " << dataset.samples.size() << ",\n"
         << "  \"training_games\": " << train.game_ids.size() << ",\n"
         << "  \"training_positions\": " << train.samples.size() << ",\n"
         << "  \"validation_games\": " << holdout.game_ids.size() << ",\n"
         << "  \"validation_positions\": " << holdout.samples.size() << ",\n"
         << "  \"split_policy\": \"game-level-random\",\n"
         << "  \"split_seed\": " << config.split_seed << ",\n"
         << "  \"optimizer\": \"SPSA\",\n"
         << "  \"iterations\": " << config.iterations << ",\n"
         << "  \"batch_size\": " << config.batch_size << ",\n"
         << "  \"optimizer_seed\": " << config.optimizer_seed << ",\n"
         << "  \"objective_k\": " << config.logistic_k << ",\n"
         << "  \"weighting\": \"" << json_escape(config.weighting) << "\",\n"
         << "  \"parameter_family\": \"" << json_escape(config.family)
         << "\",\n"
         << "  \"report_interval\": " << config.report_interval << ",\n"
         << "  \"checkpoint_interval\": " << config.checkpoint_interval << ",\n"
         << "  \"spsa_schedule\": {\"perturbation_a\":0.1,"
            "\"perturbation_exponent\":0.101,\"learning_rate_a\":0.5,"
            "\"learning_rate_exponent\":0.602,\"stability_fraction\":0.1},\n"
         << "  \"dataset_format\": \"" << json_escape(config.format) << "\",\n"
         << "  \"output_path\": \"" << json_escape(config.output_path)
         << "\",\n"
         << "  \"initial_parameters_path\": \""
         << json_escape(config.initial_path) << "\",\n"
         << "  \"checkpoint_path\": \""
         << json_escape(config.checkpoint_path) << "\",\n"
         << "  \"report_directory\": \""
         << json_escape(config.report_directory) << "\",\n"
         << "  \"import_statistics\": {\n"
         << "    \"games_encountered\": " << statistics.games_encountered
         << ",\n"
         << "    \"games_accepted\": " << statistics.games_accepted << ",\n"
         << "    \"games_skipped\": " << statistics.games_skipped << ",\n"
         << "    \"games_rejected\": " << statistics.games_rejected << ",\n"
         << "    \"positions_generated\": " << statistics.positions_generated
         << ",\n"
         << "    \"minimum_positions_per_game\": "
         << statistics.minimum_positions_per_game << ",\n"
         << "    \"median_positions_per_game\": "
         << statistics.median_positions_per_game << ",\n"
         << "    \"maximum_positions_per_game\": "
         << statistics.maximum_positions_per_game << ",\n"
         << "    \"mean_positions_per_game\": "
         << statistics.mean_positions_per_game << ",\n"
         << "    \"missing_metadata\": {";
    bool first_missing = true;
    for (const auto& entry : statistics.missing_metadata) {
        if (!first_missing)
            json << ',';
        first_missing = false;
        json << "\n      \"" << json_escape(entry.first) << "\": "
             << entry.second;
    }
    if (!statistics.missing_metadata.empty())
        json << '\n' << "    ";
    json << "},\n    \"skip_reasons\": {";
    bool first_skip_reason = true;
    for (const auto& entry : statistics.skip_reasons) {
        if (!first_skip_reason)
            json << ',';
        first_skip_reason = false;
        json << "\n      \"" << json_escape(entry.first) << "\": "
             << entry.second;
    }
    if (!statistics.skip_reasons.empty())
        json << '\n' << "    ";
    json << "}\n  },\n"
         << "  \"initial_parameters\": ";
    write_named_parameters(json, initial_weights);
    json << ",\n  \"active_parameters\": [";
    bool first = true;
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
        if (!parameter_enabled(i, config.family))
            continue;
        if (!first)
            json << ", ";
        first = false;
        json << "{\"name\":\"" << descriptors[i].name
             << "\",\"minimum\":" << descriptors[i].minimum
             << ",\"maximum\":" << descriptors[i].maximum << '}';
    }
    json << "],\n  \"metadata_filters\": [";
    for (std::size_t i = 0; i < config.metadata_filters.size(); ++i) {
        if (i != 0)
            json << ", ";
        json << "{\"key\":\"" << json_escape(config.metadata_filters[i].first)
             << "\",\"value\":\""
             << json_escape(config.metadata_filters[i].second) << "\"}";
    }
    json << "]\n}\n";
    return atomic_write(config_path.string(), json.str(), error);
}

bool write_progress_csv(const Config& config, const OptimizerState& state,
                        double validation_loss, std::string& error) {
    const std::filesystem::path path =
        std::filesystem::path(config.report_directory) / "progress.csv";
    std::vector<std::string> rows;
    if (std::filesystem::exists(path)) {
        std::ifstream input(path);
        if (!input) {
            error = "unable to read existing progress log: " + path.string();
            return false;
        }
        std::string line;
        bool header = true;
        while (std::getline(input, line)) {
            if (header) {
                if (line != "iteration,training_loss,validation_loss,best_validation_loss,best_iteration") {
                    error = "invalid progress CSV header: " + path.string();
                    return false;
                }
                header = false;
                continue;
            }
            const std::size_t comma = line.find(',');
            if (comma == std::string::npos) {
                error = "malformed progress CSV row: " + path.string();
                return false;
            }
            try {
                std::size_t parsed_characters = 0;
                const int iteration = std::stoi(
                    line.substr(0, comma), &parsed_characters);
                if (parsed_characters != comma)
                    throw std::invalid_argument("invalid iteration");
                if (iteration < state.completed_iterations)
                    rows.push_back(line);
            } catch (const std::exception&) {
                error = "malformed iteration in progress CSV: " + path.string();
                return false;
            }
        }
        if (!input.eof()) {
            error = "error reading progress CSV: " + path.string();
            return false;
        }
    }
    std::ostringstream csv;
    csv << std::setprecision(17)
        << "iteration,training_loss,validation_loss,best_validation_loss,best_iteration\n";
    for (const std::string& row : rows)
        csv << row << '\n';
    csv << state.completed_iterations << ',' << state.current_training << ','
        << validation_loss << ',' << state.best_validation << ','
        << state.best_iteration << '\n';
    return atomic_write(path.string(), csv.str(), error);
}

bool reset_progress_csv(const Config& config, std::string& error) {
    const std::string header =
        "iteration,training_loss,validation_loss,best_validation_loss,best_iteration\n";
    return atomic_write(
        (std::filesystem::path(config.report_directory) / "progress.csv").string(),
        header, error);
}

bool write_experiment_report(const Config& config, const Dataset& dataset,
                             const ImportStatistics& statistics,
                             const OptimizerState& state, double elapsed_seconds,
                             std::string& error) {
    const auto& descriptors = parameter_descriptors();
    std::ostringstream json;
    json << std::setprecision(17)
         << "{\n"
         << "  \"format_version\": 1,\n"
         << "  \"completion_status\": \"completed\",\n"
         << "  \"dataset_fingerprint\": " << dataset_fingerprint(dataset)
         << ",\n"
         << "  \"games\": " << dataset.game_ids.size() << ",\n"
         << "  \"positions\": " << dataset.samples.size() << ",\n"
         << "  \"games_encountered\": " << statistics.games_encountered << ",\n"
         << "  \"games_accepted\": " << statistics.games_accepted << ",\n"
         << "  \"games_skipped\": " << statistics.games_skipped << ",\n"
         << "  \"games_rejected\": " << statistics.games_rejected << ",\n"
         << "  \"positions_generated\": " << statistics.positions_generated
         << ",\n"
         << "  \"initial_training_loss\": " << state.initial_training << ",\n"
         << "  \"best_training_loss\": " << state.best_training << ",\n"
         << "  \"initial_validation_loss\": " << state.initial_validation
         << ",\n"
         << "  \"initial_train_validation_gap\": "
         << state.initial_validation - state.initial_training << ",\n"
         << "  \"current_training_loss\": " << state.current_training << ",\n"
         << "  \"current_validation_loss\": " << state.current_validation
         << ",\n"
         << "  \"best_validation_loss\": " << state.best_validation << ",\n"
         << "  \"best_train_validation_gap\": "
         << state.best_validation - state.best_training << ",\n"
         << "  \"minimum_positions_per_game\": "
         << statistics.minimum_positions_per_game << ",\n"
         << "  \"median_positions_per_game\": "
         << statistics.median_positions_per_game << ",\n"
         << "  \"maximum_positions_per_game\": "
         << statistics.maximum_positions_per_game << ",\n"
         << "  \"mean_positions_per_game\": "
         << statistics.mean_positions_per_game << ",\n"
         << "  \"best_iteration\": " << state.best_iteration << ",\n"
         << "  \"completed_iterations\": " << state.completed_iterations << ",\n"
         << "  \"elapsed_seconds\": " << elapsed_seconds << ",\n"
         << "  \"config_file\": \"config.json\",\n"
         << "  \"source\": \"" << json_escape(dataset.source_description)
         << "\",\n"
         << "  \"weighting\": \"" << json_escape(config.weighting) << "\",\n"
         << "  \"iterations\": " << config.iterations << ",\n"
         << "  \"batch_size\": " << config.batch_size << ",\n"
         << "  \"optimizer_seed\": " << config.optimizer_seed << ",\n"
         << "  \"split_seed\": " << config.split_seed << ",\n"
         << "  \"objective_k\": " << config.logistic_k << ",\n"
         << "  \"spsa_schedule\": {\"perturbation_a\":0.1,"
            "\"perturbation_exponent\":0.101,\"learning_rate_a\":0.5,"
            "\"learning_rate_exponent\":0.602,\"stability_fraction\":0.1},\n"
         << "  \"report_interval\": " << config.report_interval << ",\n"
         << "  \"checkpoint_interval\": " << config.checkpoint_interval
         << ",\n"
         << "  \"initial_parameters_path\": \""
         << json_escape(config.initial_path) << "\",\n"
         << "  \"dataset_format\": \"" << json_escape(config.format)
         << "\",\n"
         << "  \"report_directory\": \""
         << json_escape(config.report_directory) << "\",\n"
         << "  \"parameter_family\": \"" << json_escape(config.family)
         << "\",\n"
         << "  \"candidate_path\": \"" << json_escape(config.output_path)
         << "\",\n"
         << "  \"initial_parameters\": ";
    write_named_parameters(json, state.initial_weights);
    json << ",\n  \"best_validation_parameters\": ";
    write_named_parameters(json, state.best_weights);
    json << ",\n  \"final_current_parameters\": ";
    write_parameter_array(json, state.weights);
    json << ",\n  \"metadata_filters\": [";
    for (std::size_t i = 0; i < config.metadata_filters.size(); ++i) {
        if (i != 0)
            json << ", ";
        json << "{\"key\":\"" << json_escape(config.metadata_filters[i].first)
             << "\",\"value\":\""
             << json_escape(config.metadata_filters[i].second) << "\"}";
    }
    json << "],\n  \"missing_metadata\": {";
    bool first_missing = true;
    for (const auto& entry : statistics.missing_metadata) {
        if (!first_missing)
            json << ',';
        first_missing = false;
        json << "\n    \"" << json_escape(entry.first) << "\": "
             << entry.second;
    }
    if (!statistics.missing_metadata.empty())
        json << '\n' << "  ";
    json << "},\n  \"skip_reasons\": {";
    bool first_skip_reason = true;
    for (const auto& entry : statistics.skip_reasons) {
        if (!first_skip_reason)
            json << ',';
        first_skip_reason = false;
        json << "\n    \"" << json_escape(entry.first) << "\": "
             << entry.second;
    }
    if (!statistics.skip_reasons.empty())
        json << '\n' << "  ";
    json << "},\n  \"warnings\": [";
    for (std::size_t i = 0; i < statistics.warnings.size(); ++i) {
        if (i != 0)
            json << ", ";
        json << '"' << json_escape(statistics.warnings[i]) << '"';
    }
    json << "],\n  \"parameter_order\": [";
    for (std::size_t i = 0; i < descriptors.size(); ++i) {
        if (i != 0)
            json << ", ";
        json << '"' << descriptors[i].name << '"';
    }
    json << "]\n}\n";
    return atomic_write(
        (std::filesystem::path(config.report_directory) / "report.json").string(),
        json.str(), error);
}

} // namespace Bermuda::Tuner
