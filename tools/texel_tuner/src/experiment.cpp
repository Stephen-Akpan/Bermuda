#include <bermuda_tuner/checkpoint.h>
#include <bermuda_tuner/dataset.h>
#include <bermuda_tuner/objective.h>
#include <bermuda_tuner/optimizer.h>
#include <bermuda_tuner/parameter_space.h>
#include <bermuda_tuner/report.h>
#include <bermuda_tuner/texel_tuner.h>

#include "bitboard.h"
#include "eval.h"
#include "zobrist.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <filesystem>
#include <chrono>
#include <system_error>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace Bermuda::Tuner {
namespace {

bool validate_config(const Config& config, std::string& error) {
    if (config.iterations <= 0 || config.batch_size <= 0
        || !std::isfinite(config.logistic_k) || config.logistic_k <= 0.0
        || config.validation_percent <= 0 || config.validation_percent >= 100) {
        error = "iterations, batch size, and logistic K must be positive and finite";
        return false;
    }
    if (!parameter_family_supported(config.family)) {
        error = "unsupported tuning family: " + config.family;
        return false;
    }
    WeightingMode weighting;
    if (!parse_weighting_mode(config.weighting, weighting)
        || config.report_interval <= 0 || config.checkpoint_interval <= 0) {
        error = "weighting and report/checkpoint intervals must be valid";
        return false;
    }
    if (config.format != "text" && config.format != "btd") {
        error = "dataset format must be text or btd";
        return false;
    }
    if (config.prepare) {
        if (config.pgn_path.empty() || config.train_output_path.empty()
            || config.validation_output_path.empty()
            || !config.data_path.empty() || !config.train_data_path.empty()
            || !config.validation_data_path.empty()) {
            error = "--prepare-pgn requires --train-out and --validation-out only";
            return false;
        }
        return true;
    }

    const bool paired_data = !config.train_data_path.empty()
                          && !config.validation_data_path.empty();
    const bool partial_paired_data = config.train_data_path.empty()
                                  != config.validation_data_path.empty();
    const int ordinary_inputs = !config.data_path.empty()
                              + !config.pgn_path.empty();
    if (partial_paired_data || (paired_data && ordinary_inputs != 0)
        || (!paired_data && ordinary_inputs != 1)) {
        error = "specify --data, --pgn, or both --train-data and --validation-data";
        return false;
    }
    if (config.resume && config.checkpoint_path.empty()) {
        error = "--resume requires --checkpoint FILE";
        return false;
    }
    if (config.output_path.empty()) {
        error = "parameter output path must not be empty";
        return false;
    }
    if (config.validation_percent <= 0 || config.validation_percent >= 100) {
        error = "validation percentage must be between 1 and 99";
        return false;
    }
    return true;
}

void log_message(const std::function<void(const std::string&)>& log,
                 const std::string& message) {
    if (log)
        log(message);
}

void append_import_summary(std::ostringstream& output,
                           const ImportStatistics& statistics) {
    output << "\nImport: " << statistics.games_encountered << " encountered, "
           << statistics.games_accepted << " accepted, "
           << statistics.games_skipped << " skipped, "
           << statistics.games_rejected << " rejected"
           << "\nPositions/game min/median/max/mean: "
           << statistics.minimum_positions_per_game << '/'
           << statistics.median_positions_per_game << '/'
           << statistics.maximum_positions_per_game << '/'
           << statistics.mean_positions_per_game;
    if (!statistics.missing_metadata.empty()) {
        output << "\nMissing metadata:";
        for (const auto& entry : statistics.missing_metadata)
            output << ' ' << entry.first << '=' << entry.second;
    }
    if (!statistics.skip_reasons.empty()) {
        output << "\nSkip reasons:";
        for (const auto& entry : statistics.skip_reasons)
            output << ' ' << entry.first << '=' << entry.second;
    }
}

bool load_experiment_data(const Config& config, Dataset& dataset,
                          std::vector<std::size_t>& training,
                          std::vector<std::size_t>& validation,
                          ImportStatistics& statistics,
                          std::string& error) {
    if (!config.train_data_path.empty()) {
        Dataset train;
        Dataset holdout;
        ImportStatistics train_statistics;
        ImportStatistics validation_statistics;
        if (!load_dataset(config.train_data_path, train, train_statistics, error)
            || !load_dataset(config.validation_data_path, holdout,
                             validation_statistics, error))
            return false;
        for (const std::string& game_id : train.game_ids) {
            if (std::find(holdout.game_ids.begin(), holdout.game_ids.end(),
                          game_id) != holdout.game_ids.end()) {
                error = "game appears in both prepared splits: " + game_id;
                return false;
            }
        }
        if (!filter_dataset_metadata(train, config.metadata_filters,
                                     train_statistics, error)
            || !filter_dataset_metadata(holdout, config.metadata_filters,
                                        validation_statistics, error))
            return false;
        dataset = std::move(train);
        dataset.source_description = config.train_data_path + " + "
                                   + config.validation_data_path;
        const std::size_t validation_offset = dataset.samples.size();
        const std::size_t validation_game_offset = dataset.game_ids.size();
        for (std::size_t i = 0; i < holdout.samples.size(); ++i) {
            Sample& sample = holdout.samples[i];
            sample.game += validation_game_offset;
            validation.push_back(validation_offset + i);
            dataset.samples.push_back(std::move(sample));
        }
        dataset.game_ids.insert(dataset.game_ids.end(), holdout.game_ids.begin(),
                                holdout.game_ids.end());
        dataset.games.insert(dataset.games.end(), holdout.games.begin(),
                             holdout.games.end());
        statistics.games_encountered = train_statistics.games_encountered
                                     + validation_statistics.games_encountered;
        statistics.games_skipped = train_statistics.games_skipped
                                 + validation_statistics.games_skipped;
        statistics.games_rejected = train_statistics.games_rejected
                                  + validation_statistics.games_rejected;
        for (const auto& entry : train_statistics.skip_reasons)
            statistics.skip_reasons[entry.first] += entry.second;
        for (const auto& entry : validation_statistics.skip_reasons)
            statistics.skip_reasons[entry.first] += entry.second;
        statistics.missing_metadata.clear();
        training.resize(validation_offset);
        for (std::size_t i = 0; i < training.size(); ++i)
            training[i] = i;
        return filter_dataset_metadata(dataset, {}, statistics, error);
    }

    if (config.pgn_path.empty()) {
        if (!load_dataset(config.data_path, dataset, statistics, error))
            return false;
        if (!filter_dataset_metadata(dataset, config.metadata_filters,
                                     statistics, error))
            return false;
    } else if (!load_pgn_dataset(config.pgn_path, dataset, statistics,
                                 config.metadata_filters, error)) {
        return false;
    }

    std::mt19937_64 split_random(config.split_seed);
    std::vector<std::size_t> shuffled_games(dataset.game_ids.size());
    for (std::size_t i = 0; i < shuffled_games.size(); ++i)
        shuffled_games[i] = i;
    std::shuffle(shuffled_games.begin(), shuffled_games.end(), split_random);

    const std::size_t validation_game_count =
        std::max<std::size_t>(1, dataset.game_ids.size() / 10);
    std::vector<bool> validation_game(dataset.game_ids.size(), false);
    for (std::size_t i = 0; i < validation_game_count; ++i)
        validation_game[shuffled_games[i]] = true;

    for (std::size_t i = 0; i < dataset.samples.size(); ++i)
        (validation_game[dataset.samples[i].game] ? validation : training)
            .push_back(i);
    if (training.empty() || validation.empty()) {
        error = "unable to split dataset into training and validation games";
        return false;
    }
    return true;
}

bool save_parameters_atomic(const std::string& path,
                            const Eval::Parameters& parameters,
                            std::string& error) {
    const std::string temporary = path + ".tmp";
    const std::string backup = path + ".bak";
    if (!Eval::save_parameters(temporary, parameters, error))
        return false;
    std::error_code fs_error;
    std::filesystem::remove(backup, fs_error);
    fs_error.clear();
    if (std::filesystem::exists(path)) {
        std::filesystem::rename(path, backup, fs_error);
        if (fs_error) {
            error = "unable to preserve previous candidate: "
                  + fs_error.message();
            return false;
        }
    }
    fs_error.clear();
    std::filesystem::rename(temporary, path, fs_error);
    if (fs_error) {
        error = "unable to finalize candidate parameters: "
              + fs_error.message();
        if (std::filesystem::exists(backup)) {
            std::error_code restore_error;
            std::filesystem::rename(backup, path, restore_error);
        }
        return false;
    }
    error.clear();
    return true;
}

} // namespace

int run(const Config& config) {
    return run(config, {});
}

int run(const Config& config,
        const std::function<void(const std::string&)>& log) {
    std::string error;
    if (!validate_config(config, error)) {
        log_message(log, error);
        return 2;
    }

    BB::init();
    Zobrist::init();

    if (config.prepare) {
        Dataset dataset;
        ImportStatistics statistics;
        if (!load_pgn_dataset(config.pgn_path, dataset, statistics,
                              config.metadata_filters, error)
            || !write_prepared_datasets(config, dataset, error)) {
            log_message(log, error);
            return 2;
        }
        std::ostringstream prepared_message;
        prepared_message << "Prepared " << dataset.game_ids.size() << " games and "
                         << dataset.samples.size() << " positions.";
        append_import_summary(prepared_message, statistics);
        log_message(log, prepared_message.str());
        return 0;
    }

    Eval::Parameters initial;
    if (!config.initial_path.empty()
        && !Eval::load_parameters(config.initial_path, initial, error)) {
        log_message(log, "Unable to load initial parameters: " + error);
        return 2;
    }
    const ParameterVector initial_vector = to_vector(initial);
    if (!validate_parameters(initial_vector, error)) {
        log_message(log, error);
        return 2;
    }

    Dataset dataset;
    std::vector<std::size_t> training;
    std::vector<std::size_t> validation;
    ImportStatistics statistics;
    if (!load_experiment_data(config, dataset, training, validation,
                              statistics, error)) {
        log_message(log, error);
        return 2;
    }
    if (training.empty() || validation.empty()) {
        log_message(log, "training and validation sets must both contain positions");
        return 2;
    }

    const std::uint64_t fingerprint = dataset_fingerprint(dataset);
    OptimizerState state;
    state.random.seed(config.optimizer_seed);
    state.weights = initial_vector;
    state.best_weights = state.weights;
    state.initial_weights = initial_vector;
    WeightingMode weighting = WeightingMode::Position;
    if (!parse_weighting_mode(config.weighting, weighting)) {
        log_message(log, "unsupported objective weighting mode");
        return 2;
    }
    if (!objective(dataset, validation, to_parameters(state.weights),
                   config.logistic_k, weighting, state.best_validation, error)
        || !objective(dataset, training, to_parameters(state.weights),
                      config.logistic_k, weighting,
                      state.current_training, error)) {
        log_message(log, error);
        return 2;
    }
    state.initial_validation = state.best_validation;
    state.current_validation = state.best_validation;
    state.initial_training = state.current_training;
    state.best_training = state.current_training;
    state.best_iteration = 0;
    if (config.resume
        && !load_checkpoint(config, fingerprint, state, error)) {
        log_message(log, error);
        return 2;
    }
    const std::string report_directory = config.report_directory.empty()
        ? config.output_path + ".experiment" : config.report_directory;
    Config report_config = config;
    report_config.report_directory = report_directory;
    if (!write_experiment_config(report_config, dataset, statistics, training,
                                 validation, state.initial_weights, error)) {
        log_message(log, error);
        return 3;
    }
    if (!config.resume && !reset_progress_csv(report_config, error)) {
        log_message(log, error);
        return 3;
    }

    std::ostringstream summary;
    summary << "Positions: " << dataset.samples.size()
            << " across " << dataset.game_ids.size() << " games"
            << " (train " << training.size()
            << ", validation " << validation.size() << ')';
    append_import_summary(summary, statistics);
    summary << "\nWeighting: " << config.weighting;
    if (config.resume) {
        summary << "\nResuming at iteration " << state.completed_iterations
                << " with train MSE " << state.current_training
                << ", best validation MSE " << state.best_validation;
    } else {
        summary << "\nInitial train MSE: " << state.current_training
                << ", validation MSE: " << state.best_validation;
    }
    log_message(log, summary.str());

    std::string persistence_error;
    const auto experiment_started = std::chrono::steady_clock::now();
    const ProgressCallback progress = [&](const OptimizerState& current,
                                          double validation_loss) {
        std::ostringstream progress_message;
        progress_message << "Iteration " << current.completed_iterations
                         << " train MSE " << current.current_training
                         << " validation MSE " << validation_loss
                         << " best " << current.best_validation
                         << " at iteration " << current.best_iteration;
        log_message(log, progress_message.str());

        if (!write_progress_csv(report_config, current, validation_loss,
                                persistence_error))
            return false;
        if (!config.checkpoint_path.empty()
            && (current.completed_iterations % config.checkpoint_interval == 0
                || current.completed_iterations == config.iterations)) {
            if (!save_checkpoint(config, fingerprint, current,
                                 persistence_error))
                return false;
            if (!save_parameters_atomic(config.output_path,
                                        to_parameters(current.best_weights),
                                        persistence_error)) {
                persistence_error = "Unable to save best candidate: "
                                  + persistence_error;
                return false;
            }
        }
        return true;
    };

    if (!run_spsa(dataset, training, validation, config, state, progress, error)) {
        log_message(log, persistence_error.empty() ? error : persistence_error);
        return 2;
    }
    if (!save_parameters_atomic(config.output_path,
                                to_parameters(state.best_weights), error)) {
        log_message(log, "Unable to save tuned parameters: " + error);
        return 2;
    }
    const double elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - experiment_started).count();
    if (!write_experiment_report(report_config, dataset, statistics, state,
                                 elapsed_seconds, error)) {
        log_message(log, error);
        return 3;
    }

    std::ostringstream final_message;
    final_message << "Initial validation MSE: " << state.initial_validation
                  << ", best validation MSE: " << state.best_validation
                  << "\nSaved tuned parameters to " << config.output_path;
    const auto& descriptors = parameter_descriptors();
    for (std::size_t i = 0; i < ParameterCount; ++i)
        final_message << '\n' << descriptors[i].name << ' ' << state.best_weights[i];
    log_message(log, final_message.str());
    return 0;
}

} // namespace Bermuda::Tuner
