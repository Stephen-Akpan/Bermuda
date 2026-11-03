#include <bermuda_tuner/objective.h>
#include <bermuda_tuner/optimizer.h>
#include <bermuda_tuner/parameter_space.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace Bermuda::Tuner {

bool run_spsa(const Dataset& dataset,
              const std::vector<std::size_t>& training,
              const std::vector<std::size_t>& validation,
              const Config& config, OptimizerState& state,
              const ProgressCallback& progress, std::string& error) {
    if (config.iterations <= 0 || config.batch_size <= 0
        || !std::isfinite(config.logistic_k) || config.logistic_k <= 0.0
        || !parameter_family_supported(config.family)) {
        error = "SPSA configuration is invalid";
        return false;
    }
    if (training.empty() || validation.empty()) {
        error = "SPSA requires non-empty training and validation sets";
        return false;
    }
    WeightingMode weighting;
    if (!parse_weighting_mode(config.weighting, weighting)
        || config.report_interval <= 0 || config.checkpoint_interval <= 0) {
        error = "SPSA weighting or report interval is invalid";
        return false;
    }

    std::vector<std::size_t> batch;
    batch.reserve(std::min<std::size_t>(
        static_cast<std::size_t>(config.batch_size), training.size()));
    ParameterVector direction{};
    const double stability = std::max(10.0, config.iterations * 0.1);

    for (int iteration = state.completed_iterations;
         iteration < config.iterations; ++iteration) {
        batch = training;
        std::shuffle(batch.begin(), batch.end(), state.random);
        if (batch.size() > static_cast<std::size_t>(config.batch_size))
            batch.resize(static_cast<std::size_t>(config.batch_size));

        const double perturbation = 0.1 / std::pow(iteration + 1.0, 0.101);
        const double learning_rate =
            0.5 / std::pow(stability + iteration + 1.0, 0.602);
        auto plus = state.weights;
        auto minus = state.weights;
        for (std::size_t i = 0; i < ParameterCount; ++i) {
            direction[i] = parameter_enabled(i, config.family)
                ? ((state.random() & 1) ? 1.0 : -1.0) : 0.0;
            plus[i] += perturbation * direction[i];
            minus[i] -= perturbation * direction[i];
        }
        clamp_parameters(plus);
        clamp_parameters(minus);

        double loss_plus = 0.0;
        double loss_minus = 0.0;
        if (!objective(dataset, batch, to_parameters(plus), config.logistic_k,
                       weighting,
                       loss_plus, error)
            || !objective(dataset, batch, to_parameters(minus), config.logistic_k,
                          weighting,
                          loss_minus, error))
            return false;
        const double difference = loss_plus - loss_minus;
        for (std::size_t i = 0; i < ParameterCount; ++i) {
            if (direction[i] == 0.0)
                continue;
            const double gradient = difference
                / (2.0 * perturbation * direction[i]);
            state.weights[i] -= learning_rate * gradient;
        }
        clamp_parameters(state.weights);
        state.completed_iterations = iteration + 1;

        if ((iteration + 1) % config.report_interval == 0
            || (!config.checkpoint_path.empty()
                && (iteration + 1) % config.checkpoint_interval == 0)
            || iteration + 1 == config.iterations) {
            const Eval::Parameters candidate = to_parameters(state.weights);
            double validation_loss = 0.0;
            if (!objective(dataset, validation, candidate, config.logistic_k,
                           weighting,
                           validation_loss, error)
                || !objective(dataset, training, candidate, config.logistic_k,
                              weighting,
                              state.current_training, error))
                return false;
            state.current_validation = validation_loss;
            if (validation_loss < state.best_validation) {
                state.best_validation = validation_loss;
                state.best_weights = state.weights;
                state.best_iteration = state.completed_iterations;
                state.best_training = state.current_training;
            }
            if (progress && !progress(state, validation_loss)) {
                error = "SPSA progress callback failed";
                return false;
            }
        }
    }

    error.clear();
    return true;
}

} // namespace Bermuda::Tuner
