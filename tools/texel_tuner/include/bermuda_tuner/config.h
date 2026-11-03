#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace Bermuda::Tuner {

struct Config {
    std::string data_path;
    std::string pgn_path;
    std::string train_data_path;
    std::string validation_data_path;
    std::string train_output_path;
    std::string validation_output_path;
    std::string output_path = "eval-params.txt";
    std::string initial_path;
    std::string checkpoint_path;
    std::string report_directory;
    bool prepare = false;
    bool resume = false;
    std::string family = "all";
    std::string weighting = "position";
    std::string format = "text";
    int iterations = 300;
    int batch_size = 4096;
    int report_interval = 10;
    int checkpoint_interval = 10;
    std::uint64_t split_seed = 1;
    std::uint64_t optimizer_seed = 1;
    int validation_percent = 10;
    double logistic_k = 1.13;
    std::vector<std::pair<std::string, std::string>> metadata_filters;
};

} // namespace Bermuda::Tuner
