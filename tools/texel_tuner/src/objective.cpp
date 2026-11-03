#include <bermuda_tuner/objective.h>
#include <algorithm>
#include <cmath>
#include <unordered_map>
namespace Bermuda::Tuner {
bool parse_weighting_mode(const std::string& value, WeightingMode& weighting) {
    if (value == "position") {
        weighting = WeightingMode::Position;
        return true;
    }
    if (value == "game") {
        weighting = WeightingMode::Game;
        return true;
    }
    return false;
}

double predicted_score(const Sample& sample, const Eval::Parameters& parameters,
                       double logistic_k) {
    const double sign = sample.position.side_to_move() == WHITE ? 1.0 : -1.0;
    const double centipawns = sign * Eval::evaluate(sample.position, parameters);
    const double exponent = std::clamp(
        -logistic_k * centipawns * std::log(10.0) / 400.0, -60.0, 60.0);
    return 1.0 / (1.0 + std::exp(exponent));
}

bool objective(const Dataset& dataset, const std::vector<std::size_t>& indices,
               const Eval::Parameters& parameters, double logistic_k,
               double& loss, std::string& error) {
    return objective(dataset, indices, parameters, logistic_k,
                     WeightingMode::Position, loss, error);
}

bool objective(const Dataset& dataset, const std::vector<std::size_t>& indices,
               const Eval::Parameters& parameters, double logistic_k,
               WeightingMode weighting, double& loss, std::string& error) {
    if (indices.empty()) { error = "objective requires at least one sample"; return false; }
    if (!std::isfinite(logistic_k) || logistic_k <= 0.0) { error = "logistic K must be positive and finite"; return false; }
    double total = 0.0;
    std::unordered_map<std::size_t, std::pair<double, std::size_t>> game_losses;
    for (const std::size_t index : indices) {
        if (index >= dataset.samples.size()) { error = "objective sample index is out of range"; return false; }
        if (weighting == WeightingMode::Game
            && dataset.samples[index].game >= dataset.game_ids.size()) {
            error = "game-weighted objective sample has an invalid game index";
            return false;
        }
        const double result = dataset.samples[index].result;
        if (!std::isfinite(result)
            || (result != 0.0 && result != 0.5 && result != 1.0)) {
            error = "objective sample result must be 0, 0.5, or 1";
            return false;
        }
        const double prediction = predicted_score(dataset.samples[index], parameters, logistic_k);
        if (!std::isfinite(prediction)) { error = "non-finite prediction in objective"; return false; }
        const double difference = prediction - result;
        const double squared_error = difference * difference;
        if (weighting == WeightingMode::Game) {
            auto& game_loss = game_losses[dataset.samples[index].game];
            game_loss.first += squared_error;
            ++game_loss.second;
        } else {
            total += squared_error;
        }
    }
    if (weighting == WeightingMode::Game) {
        for (const auto& game_loss : game_losses)
            total += game_loss.second.first / game_loss.second.second;
        loss = total / game_losses.size();
    } else {
        loss = total / indices.size();
    }
    if (!std::isfinite(loss)) { error = "non-finite objective loss"; return false; }
    error.clear();
    return true;
}
} // namespace Bermuda::Tuner
