#pragma once

#include <bermuda_tuner/config.h>
#include <bermuda_tuner/optimizer.h>
#include <bermuda_tuner/types.h>

#include <cstddef>
#include <string>
#include <vector>

namespace Bermuda::Tuner {

bool write_experiment_config(const Config& config, const Dataset& dataset,
                             const ImportStatistics& statistics,
                             const std::vector<std::size_t>& training,
                             const std::vector<std::size_t>& validation,
                             const ParameterVector& initial_weights,
                             std::string& error);
bool reset_progress_csv(const Config& config, std::string& error);
bool write_progress_csv(const Config& config, const OptimizerState& state,
                        double validation_loss, std::string& error);
bool write_experiment_report(const Config& config, const Dataset& dataset,
                             const ImportStatistics& statistics,
                             const OptimizerState& state, double elapsed_seconds,
                             std::string& error);

} // namespace Bermuda::Tuner
