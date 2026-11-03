#include <bermuda_tuner/texel_tuner.h>
#include <bermuda_tuner/parameter_space.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void print_usage() {
    std::cout
        << "Usage: bermuda_tuner (--data FILE | --pgn FILE | --train-data FILE --validation-data FILE) [options]\n"
        << "Dataset rows: GAME_ID RESULT FEN\n"
        << "RESULT is 1-0, 1/2-1/2, 0-1, or 1, 0.5, 0 from White's perspective.\n"
        << "Options:\n"
        << "  --prepare-pgn FILE  Validate PGN and create split datasets\n"
        << "  --train-out FILE    Training dataset output for --prepare-pgn\n"
        << "  --validation-out FILE Validation dataset output for --prepare-pgn\n"
        << "  --validation-percent N Validation games percent (default 10)\n"
        << "  --split-seed N      Deterministic train/validation split seed\n"
        << "  --pgn FILE          Import PGN games and use all pre-move positions\n"
        << "  --train-data FILE   Use prepared training dataset\n"
        << "  --validation-data FILE Use prepared validation dataset\n"
        << "  --data FILE         Use prepared dataset and split by game\n"
        << "  --out FILE          Parameter output file (default eval-params.txt)\n"
        << "  --init FILE         Start from an existing parameter file\n"
        << "  --checkpoint FILE   Save resumable optimizer state\n"
        << "  --checkpoint-interval N Checkpoint/report interval (default 10)\n"
        << "  --resume            Resume from --checkpoint\n"
        << "  --family NAME       all, material, positional, piece-values, piece-values-mg/eg, or a parameter name\n"
        << "  --weighting MODE    position (default) or game\n"
        << "  --metadata-filter KEY=VALUE Keep only PGN games with this tag (repeatable)\n"
        << "  --format FORMAT     Prepared dataset format: text (default) or btd\n"
        << "  --report-dir DIR    Experiment config, progress, and report directory\n"
        << "  --report-interval N Progress/report interval (default 10)\n"
        << "  --iterations N      SPSA iterations (default 300)\n"
        << "  --batch N           Training positions per iteration (default 4096)\n"
        << "  --seed N            SPSA random seed (default 1)\n"
        << "  --k VALUE           Texel logistic K (default 1.13)\n";
}

bool parse_positive_integer(const std::string& value, int& result) {
    try {
        std::size_t used = 0;
        const long parsed = std::stol(value, &used);
        if (used != value.size() || parsed <= 0
            || parsed > std::numeric_limits<int>::max())
            return false;
        result = static_cast<int>(parsed);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parse_config(int argc, char** argv, Bermuda::Tuner::Config& config,
                  bool& help_requested) {
    help_requested = false;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--help" || option == "-h") {
            print_usage();
            help_requested = true;
            return false;
        }
        if (option == "--resume") {
            config.resume = true;
            continue;
        }
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << option << '\n';
            return false;
        }
        const std::string value = argv[++i];
        if (option == "--prepare-pgn") {
            config.prepare = true;
            config.pgn_path = value;
        } else if (option == "--train-out") {
            config.train_output_path = value;
        } else if (option == "--validation-out") {
            config.validation_output_path = value;
        } else if (option == "--validation-percent") {
            if (!parse_positive_integer(value, config.validation_percent)
                || config.validation_percent >= 100) {
                std::cerr << "Invalid validation percentage: " << value << '\n';
                return false;
            }
        } else if (option == "--split-seed") {
            try {
                if (value.empty() || value.front() == '-')
                    throw std::invalid_argument("negative seed");
                std::size_t used = 0;
                config.split_seed = std::stoull(value, &used);
                if (used != value.size())
                    throw std::invalid_argument("trailing characters");
            } catch (const std::exception&) {
                std::cerr << "Invalid split seed: " << value << '\n';
                return false;
            }
        } else if (option == "--train-data") {
            config.train_data_path = value;
        } else if (option == "--validation-data") {
            config.validation_data_path = value;
        } else if (option == "--checkpoint") {
            config.checkpoint_path = value;
        } else if (option == "--checkpoint-interval") {
            if (!parse_positive_integer(value, config.checkpoint_interval)) {
                std::cerr << "Invalid checkpoint interval: " << value << '\n';
                return false;
            }
        } else if (option == "--report-dir") {
            config.report_directory = value;
        } else if (option == "--report-interval") {
            if (!parse_positive_integer(value, config.report_interval)) {
                std::cerr << "Invalid report interval: " << value << '\n';
                return false;
            }
        } else if (option == "--weighting") {
            config.weighting = value;
        } else if (option == "--format") {
            config.format = value;
        } else if (option == "--metadata-filter") {
            const std::size_t separator = value.find('=');
            if (separator == std::string::npos || separator == 0) {
                std::cerr << "Metadata filter must be KEY=VALUE: " << value << '\n';
                return false;
            }
            config.metadata_filters.emplace_back(
                value.substr(0, separator), value.substr(separator + 1));
        } else if (option == "--data") {
            config.data_path = value;
        } else if (option == "--pgn") {
            config.pgn_path = value;
        } else if (option == "--out") {
            config.output_path = value;
        } else if (option == "--init") {
            config.initial_path = value;
        } else if (option == "--family") {
            config.family = value;
        } else if (option == "--iterations") {
            if (!parse_positive_integer(value, config.iterations)) {
                std::cerr << "Invalid iteration count: " << value << '\n';
                return false;
            }
        } else if (option == "--batch") {
            if (!parse_positive_integer(value, config.batch_size)) {
                std::cerr << "Invalid batch size: " << value << '\n';
                return false;
            }
        } else if (option == "--seed") {
            try {
                if (value.empty() || value.front() == '-')
                    throw std::invalid_argument("negative seed");
                std::size_t used = 0;
                config.optimizer_seed = std::stoull(value, &used);
                if (used != value.size())
                    throw std::invalid_argument("trailing characters");
            } catch (const std::exception&) {
                std::cerr << "Invalid random seed: " << value << '\n';
                return false;
            }
        } else if (option == "--k") {
            try {
                std::size_t used = 0;
                config.logistic_k = std::stod(value, &used);
                if (used != value.size() || !std::isfinite(config.logistic_k)
                    || config.logistic_k <= 0.0)
                    throw std::invalid_argument("invalid K");
            } catch (const std::exception&) {
                std::cerr << "Invalid logistic K: " << value << '\n';
                return false;
            }
        } else {
            std::cerr << "Unknown option: " << option << '\n';
            return false;
        }
    }

    if (config.prepare) {
        if (config.pgn_path.empty() || config.train_output_path.empty()
            || config.validation_output_path.empty()
            || !config.data_path.empty() || !config.train_data_path.empty()
            || !config.validation_data_path.empty()) {
            std::cerr << "--prepare-pgn requires --train-out and --validation-out only\n";
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
        std::cerr << "Specify --data, --pgn, or both --train-data and --validation-data\n";
        print_usage();
        return false;
    }
    if (config.resume && config.checkpoint_path.empty()) {
        std::cerr << "--resume requires --checkpoint FILE\n";
        return false;
    }
    if (!Bermuda::Tuner::parameter_family_supported(config.family)) {
        std::cerr << "Unsupported tuning family: " << config.family << '\n';
        return false;
    }
    if (config.weighting != "position" && config.weighting != "game") {
        std::cerr << "Unsupported weighting mode: " << config.weighting << '\n';
        return false;
    }
    if (config.format != "text" && config.format != "btd") {
        std::cerr << "Unsupported dataset format: " << config.format << '\n';
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    Bermuda::Tuner::Config config;
    bool help_requested = false;
    if (!parse_config(argc, argv, config, help_requested))
        return help_requested ? 0 : 2;
    return Bermuda::Tuner::run(config, [](const std::string& message) {
        std::cout << message << '\n';
    });
}
