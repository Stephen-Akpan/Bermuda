#pragma once

#include "eval.h"

#include <array>
#include <cstddef>
#include <string_view>

namespace Bermuda::Tuner {

constexpr std::size_t ParameterCount = 19;
using ParameterVector = std::array<double, ParameterCount>;

enum class ParameterGroup {
    EvaluationTerms,
    MaterialGlobal,
    PieceValuesMg,
    PieceValuesEg
};

struct ParameterDescriptor {
    std::string_view name;
    double minimum;
    double maximum;
    double default_value;
    ParameterGroup group;
};

const std::array<ParameterDescriptor, ParameterCount>& parameter_descriptors();
ParameterVector to_vector(const Eval::Parameters& parameters);
Eval::Parameters to_parameters(const ParameterVector& values);
bool parameter_enabled(std::size_t index, const std::string& family);
bool parameter_family_supported(const std::string& family);
void clamp_parameters(ParameterVector& parameters);
bool validate_parameters(const ParameterVector& parameters, std::string& error);

} // namespace Bermuda::Tuner
