#include <bermuda_tuner/parameter_space.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace Bermuda::Tuner {
namespace {

constexpr std::array<ParameterDescriptor, ParameterCount> Descriptors = {{
    {"material", 0.0, 3.0, 1.0, ParameterGroup::MaterialGlobal},
    {"piece_square", 0.0, 3.0, 1.0, ParameterGroup::EvaluationTerms},
    {"pawn_structure", 0.0, 3.0, 1.0, ParameterGroup::EvaluationTerms},
    {"passed_pawns", 0.0, 3.0, 1.0, ParameterGroup::EvaluationTerms},
    {"mobility", 0.0, 3.0, 1.0, ParameterGroup::EvaluationTerms},
    {"activity", 0.0, 3.0, 1.0, ParameterGroup::EvaluationTerms},
    {"threats", 0.0, 3.0, 1.0, ParameterGroup::EvaluationTerms},
    {"king_safety", 0.0, 3.0, 1.0, ParameterGroup::EvaluationTerms},
    {"tempo", 0.0, 3.0, 1.0, ParameterGroup::EvaluationTerms},
    {"pawn_mg", 0.7, 1.3, 1.0, ParameterGroup::PieceValuesMg},
    {"knight_mg", 0.7, 1.3, 1.0, ParameterGroup::PieceValuesMg},
    {"bishop_mg", 0.7, 1.3, 1.0, ParameterGroup::PieceValuesMg},
    {"rook_mg", 0.7, 1.3, 1.0, ParameterGroup::PieceValuesMg},
    {"queen_mg", 0.7, 1.3, 1.0, ParameterGroup::PieceValuesMg},
    {"pawn_eg", 0.7, 1.3, 1.0, ParameterGroup::PieceValuesEg},
    {"knight_eg", 0.7, 1.3, 1.0, ParameterGroup::PieceValuesEg},
    {"bishop_eg", 0.7, 1.3, 1.0, ParameterGroup::PieceValuesEg},
    {"rook_eg", 0.7, 1.3, 1.0, ParameterGroup::PieceValuesEg},
    {"queen_eg", 0.7, 1.3, 1.0, ParameterGroup::PieceValuesEg}
}};

} // namespace

const std::array<ParameterDescriptor, ParameterCount>& parameter_descriptors() {
    return Descriptors;
}

ParameterVector to_vector(const Eval::Parameters& parameters) {
    return {
        parameters.material, parameters.piece_square,
        parameters.pawn_structure, parameters.passed_pawns,
        parameters.mobility, parameters.activity, parameters.threats,
        parameters.king_safety, parameters.tempo,
        parameters.pawn_mg, parameters.knight_mg, parameters.bishop_mg,
        parameters.rook_mg, parameters.queen_mg,
        parameters.pawn_eg, parameters.knight_eg, parameters.bishop_eg,
        parameters.rook_eg, parameters.queen_eg
    };
}

Eval::Parameters to_parameters(const ParameterVector& values) {
    Eval::Parameters parameters;
    parameters.material = values[0];
    parameters.piece_square = values[1];
    parameters.pawn_structure = values[2];
    parameters.passed_pawns = values[3];
    parameters.mobility = values[4];
    parameters.activity = values[5];
    parameters.threats = values[6];
    parameters.king_safety = values[7];
    parameters.tempo = values[8];
    parameters.pawn_mg = values[9];
    parameters.knight_mg = values[10];
    parameters.bishop_mg = values[11];
    parameters.rook_mg = values[12];
    parameters.queen_mg = values[13];
    parameters.pawn_eg = values[14];
    parameters.knight_eg = values[15];
    parameters.bishop_eg = values[16];
    parameters.rook_eg = values[17];
    parameters.queen_eg = values[18];
    return parameters;
}

bool parameter_enabled(std::size_t index, const std::string& family) {
    if (index >= ParameterCount)
        return false;
    if (family == "all")
        return true;
    if (family == "material")
        return Descriptors[index].group == ParameterGroup::MaterialGlobal;
    if (family == "piece-values")
        return Descriptors[index].group == ParameterGroup::PieceValuesMg
            || Descriptors[index].group == ParameterGroup::PieceValuesEg;
    if (family == "positional")
        return index >= 1 && index <= 8;
    if (family == "piece-values-mg")
        return Descriptors[index].group == ParameterGroup::PieceValuesMg;
    if (family == "piece-values-eg")
        return Descriptors[index].group == ParameterGroup::PieceValuesEg;

    std::string normalized(family);
    std::replace(normalized.begin(), normalized.end(), '-', '_');
    return Descriptors[index].name == normalized;
}

bool parameter_family_supported(const std::string& family) {
    if (family == "all" || family == "material" || family == "piece-values"
        || family == "positional" || family == "piece-values-mg"
        || family == "piece-values-eg")
        return true;
    std::string normalized(family);
    std::replace(normalized.begin(), normalized.end(), '-', '_');
    for (const ParameterDescriptor& descriptor : Descriptors) {
        if (descriptor.name == normalized)
            return true;
    }
    return false;
}

void clamp_parameters(ParameterVector& parameters) {
    for (std::size_t i = 0; i < parameters.size(); ++i)
        parameters[i] = std::clamp(parameters[i], Descriptors[i].minimum,
                                   Descriptors[i].maximum);
}

bool validate_parameters(const ParameterVector& parameters, std::string& error) {
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        if (!std::isfinite(parameters[i])
            || parameters[i] < Descriptors[i].minimum
            || parameters[i] > Descriptors[i].maximum) {
            error = "invalid value for parameter "
                  + std::string(Descriptors[i].name);
            return false;
        }
    }
    error.clear();
    return true;
}

} // namespace Bermuda::Tuner
