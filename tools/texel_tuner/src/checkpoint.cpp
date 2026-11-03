#include <bermuda_tuner/checkpoint.h>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
namespace Bermuda::Tuner {
namespace {
bool save_checkpoint_impl(const Config& config, std::uint64_t fingerprint,
                          const OptimizerState& state, std::string& error) {
    const std::string temporary = config.checkpoint_path + ".tmp";
    const std::string backup = config.checkpoint_path + ".bak";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) {
        error = "unable to create optimizer checkpoint: " + temporary;
        return false;
    }
    output.precision(17);
    output << "TRIANGLE_OPTIMIZER 6\n"
           << fingerprint << '\n'
           << config.family << '\n'
           << config.weighting << ' ' << config.report_interval << ' '
           << config.checkpoint_interval << ' '
           << config.iterations << ' ' << config.batch_size << ' '
           << config.logistic_k << ' ' << config.split_seed << ' '
           << config.optimizer_seed << '\n'
           << state.completed_iterations << ' ' << state.best_iteration << '\n'
           << state.best_validation << ' ' << state.initial_validation << ' '
           << state.initial_training << ' ' << state.best_training << ' '
           << state.current_training << ' '
           << state.current_validation << '\n';
    for (double value : state.initial_weights)
        output << value << ' ';
    output << '\n';
    for (double value : state.weights)
        output << value << ' ';
    output << '\n';
    for (double value : state.best_weights)
        output << value << ' ';
    output << '\n' << state.random << '\n';
    output.close();
    if (!output) {
        error = "failed writing optimizer checkpoint: " + temporary;
        return false;
    }

    std::error_code fs_error;
    std::filesystem::remove(backup, fs_error);
    fs_error.clear();
    if (std::filesystem::exists(config.checkpoint_path)) {
        std::filesystem::rename(config.checkpoint_path, backup, fs_error);
        if (fs_error) {
            error = "unable to rotate optimizer checkpoint: "
                  + fs_error.message();
            return false;
        }
    }
    fs_error.clear();
    std::filesystem::rename(temporary, config.checkpoint_path, fs_error);
    if (fs_error) {
        error = "unable to finalize optimizer checkpoint: "
              + fs_error.message();
        if (std::filesystem::exists(backup)) {
            std::error_code restore_error;
            std::filesystem::rename(backup, config.checkpoint_path, restore_error);
        }
        return false;
    }
    error.clear();
    return true;
}

bool read_checkpoint_file_impl(const std::string& path, const Config& config,
                          std::uint64_t expected_fingerprint,
                          OptimizerState& state, std::string& error) {
    std::ifstream input(path);
    if (!input) {
        error = "unable to open " + path;
        return false;
    }
    OptimizerState loaded = state;
    std::string magic;
    int version = 0;
    std::uint64_t fingerprint = 0;
    std::string family;
    std::string weighting;
    int report_interval = 0;
    int checkpoint_interval = 0;
    int iterations = 0;
    int batch_size = 0;
    double logistic_k = 0.0;
    std::uint64_t split_seed = 0;
    std::uint64_t optimizer_seed = 0;
    if (!(input >> magic >> version >> fingerprint >> family
          >> weighting >> report_interval
          >> checkpoint_interval
          >> iterations >> batch_size >> logistic_k
          >> split_seed >> optimizer_seed
          >> loaded.completed_iterations >> loaded.best_iteration
          >> loaded.best_validation
          >> loaded.initial_validation >> loaded.initial_training
          >> loaded.best_training >> loaded.current_training
          >> loaded.current_validation)
        || magic != "TRIANGLE_OPTIMIZER" || version != 6) {
        error = "invalid optimizer checkpoint header";
        return false;
    }
    if (fingerprint != expected_fingerprint || family != config.family
        || weighting != config.weighting
        || report_interval != config.report_interval
        || checkpoint_interval != config.checkpoint_interval
        || iterations != config.iterations || batch_size != config.batch_size
        || logistic_k != config.logistic_k
        || split_seed != config.split_seed
        || optimizer_seed != config.optimizer_seed) {
        error = "checkpoint does not match datasets or optimizer options";
        return false;
    }
    for (double& value : loaded.initial_weights) {
        if (!(input >> value) || !std::isfinite(value)) {
            error = "invalid initial weights in optimizer checkpoint";
            return false;
        }
    }
    for (double& value : loaded.weights) {
        if (!(input >> value) || !std::isfinite(value)) {
            error = "invalid current weights in optimizer checkpoint";
            return false;
        }
    }
    for (double& value : loaded.best_weights) {
        if (!(input >> value) || !std::isfinite(value)) {
            error = "invalid best weights in optimizer checkpoint";
            return false;
        }
    }
    if (!(input >> loaded.random)
        || loaded.completed_iterations < 0
        || loaded.completed_iterations > config.iterations
        || loaded.best_iteration < 0
        || loaded.best_iteration > loaded.completed_iterations
        || !std::isfinite(loaded.best_validation)
        || !std::isfinite(loaded.initial_validation)
        || !std::isfinite(loaded.initial_training)
        || !std::isfinite(loaded.best_training)
        || !std::isfinite(loaded.current_training)
        || !std::isfinite(loaded.current_validation)) {
        error = "invalid optimizer state in checkpoint";
        return false;
    }
    if (loaded.initial_weights != state.initial_weights) {
        error = "checkpoint initial parameters do not match this experiment";
        return false;
    }
    if (!validate_parameters(loaded.initial_weights, error)
        || !validate_parameters(loaded.weights, error)
        || !validate_parameters(loaded.best_weights, error)) {
        error = "checkpoint contains out-of-bounds parameters: " + error;
        return false;
    }
    state = std::move(loaded);
    error.clear();
    return true;
}

bool load_checkpoint_impl(const Config& config, std::uint64_t fingerprint,
                          OptimizerState& state, std::string& error) {
    if (read_checkpoint_file_impl(config.checkpoint_path, config, fingerprint,
                             state, error))
        return true;
    const std::string primary_error = error;
    if (read_checkpoint_file_impl(config.checkpoint_path + ".bak", config,
                             fingerprint, state, error)) {
        error.clear();
        return true;
    }
    error = "unable to resume optimizer: " + primary_error
          + "; backup: " + error;
    return false;
}


} // namespace
bool save_checkpoint(const Config& config, std::uint64_t fingerprint, const OptimizerState& state, std::string& error) {
    return save_checkpoint_impl(config, fingerprint, state, error);
}
bool load_checkpoint(const Config& config, std::uint64_t fingerprint, OptimizerState& state, std::string& error) {
    return load_checkpoint_impl(config, fingerprint, state, error);
}
} // namespace Bermuda::Tuner
