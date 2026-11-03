#pragma once

#include <bermuda_tuner/types.h>

#include "eval.h"

#include <string>
#include <vector>

namespace Bermuda::Tuner {

enum class WeightingMode {
    Position,
    Game
};

bool parse_weighting_mode(const std::string& value, WeightingMode& weighting);
double predicted_score(const Sample& sample, const Eval::Parameters& parameters,
                      double logistic_k);
bool objective(const Dataset& dataset, const std::vector<std::size_t>& indices,
               const Eval::Parameters& parameters, double logistic_k,
               double& loss, std::string& error);
bool objective(const Dataset& dataset, const std::vector<std::size_t>& indices,
               const Eval::Parameters& parameters, double logistic_k,
               WeightingMode weighting, double& loss, std::string& error);

} // namespace Bermuda::Tuner
