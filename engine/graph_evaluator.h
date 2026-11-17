#pragma once

#include "graph_encoder.h"

#include <array>
#include <string>

namespace GNN {

struct GraphModelMetadata {
    ui32 format_version = 0;
    ui32 architecture_version = 0;
    ui32 tensor_count = 0;
    ui64 parameter_count = 0;
    f32 residual_multiplier = 1.0f;
    std::string baseline_checkpoint_id;
};

struct GraphEvaluatorProfile {
    double input_validation_us = 0.0;
    double input_projection_us = 0.0;
    double self_transform_us = 0.0;
    double attack_transform_us = 0.0;
    double defence_transform_us = 0.0;
    double aggregation_us = 0.0;
    double layer_norm_us = 0.0;
    double pooling_and_value_head_us = 0.0;
};

class GraphEvaluator {
public:
    bool load(const std::string& path, std::string& error);

    bool is_loaded() const { return loaded; }
    const GraphModelMetadata& metadata() const { return model_metadata; }

    // Returns the raw PyTorch GraphEvaluator output, before residual scaling.
    f32 evaluate(const GraphFeatures& features) const;

    // Returns the raw output and records per-stage inference timings.
    f32 evaluate_profiled(const GraphFeatures& features, GraphEvaluatorProfile& profile) const;

    // Experimental equivalent that aggregates neighbours before relation transforms.
    f32 evaluate_reordered(const GraphFeatures& features) const;
    f32 evaluate_reordered_profiled(
        const GraphFeatures& features,
        GraphEvaluatorProfile& profile) const;

    // Returns evaluate(features) multiplied by the exported M3 residual scale.
    f32 evaluate_residual(const GraphFeatures& features) const;
    f32 evaluate_residual_reordered(const GraphFeatures& features) const;

private:
    template <bool Profile, bool Reordered>
    f32 evaluate_impl(const GraphFeatures& features, GraphEvaluatorProfile* profile) const;

    struct Parameters {
        std::array<f32, 32 * 16> square_projection_weight{};
        std::array<f32, 32> square_projection_bias{};
        std::array<f32, 64 * 32> square_embedding_weight{};
        std::array<f32, 32 * 5> global_projection_weight{};
        std::array<f32, 32> global_projection_bias{};
        std::array<f32, 32 * 32> self_transform_weight{};
        std::array<f32, 32 * 32> attack_transform_weight{};
        std::array<f32, 32 * 32> defence_transform_weight{};
        std::array<f32, 32> layer_norm_weight{};
        std::array<f32, 32> layer_norm_bias{};
        std::array<f32, 64 * 133> value_hidden_weight{};
        std::array<f32, 64> value_hidden_bias{};
        std::array<f32, 64> value_output_weight{};
        std::array<f32, 1> value_output_bias{};
    };

    Parameters parameters{};
    GraphModelMetadata model_metadata{};
    bool loaded = false;
};

} // namespace GNN