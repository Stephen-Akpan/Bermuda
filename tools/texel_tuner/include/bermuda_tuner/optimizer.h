#pragma once

#include <bermuda_tuner/parameter_space.h>
#include <bermuda_tuner/config.h>
#include <bermuda_tuner/types.h>

#include <functional>
#include <random>
#include <string>
#include <vector>

namespace Bermuda::Tuner {

struct OptimizerState {
    int completed_iterations = 0;
    int best_iteration = 0;
    double best_validation = 0.0;
    double initial_validation = 0.0;
    double initial_training = 0.0;
    double best_training = 0.0;
    double current_training = 0.0;
    double current_validation = 0.0;
    ParameterVector initial_weights{};
    ParameterVector weights{};
    ParameterVector best_weights{};
    std::mt19937_64 random;
};

using ProgressCallback =
    std::function<bool(const OptimizerState&, double validation_loss)>;

bool run_spsa(const Dataset& dataset,
              const std::vector<std::size_t>& training,
              const std::vector<std::size_t>& validation,
              const Config& config, OptimizerState& state,
              const ProgressCallback& progress, std::string& error);

} // namespace Bermuda::Tuner
