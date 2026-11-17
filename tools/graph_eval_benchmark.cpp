#include "bitboard.h"
#include "eval.h"
#include "graph_encoder.h"
#include "graph_evaluator.h"
#include "nnue.h"
#include "position.h"
#include "search.h"
#include "tt.h"
#include "zobrist.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <utility>
#include <vector>

namespace {

class NullBuffer : public std::streambuf {
protected:
    int overflow(int character) override { return character; }
};

struct BenchmarkResult {
    double milliseconds;
    ui64 operations;
};

struct PairedBenchmarkResult {
    BenchmarkResult reference{ 0.0, 0 };
    BenchmarkResult reordered{ 0.0, 0 };
};

template <typename Function>
BenchmarkResult benchmark(ui64 iterations, Function&& function) {
    const auto started = std::chrono::steady_clock::now();
    for (ui64 iteration = 0; iteration < iterations; ++iteration)
        function(iteration);
    const auto elapsed = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    return { elapsed, iterations };
}

template <typename ReferenceFunction, typename ReorderedFunction>
PairedBenchmarkResult benchmark_paired(
    ui64 iterations,
    ReferenceFunction&& referenceFunction,
    ReorderedFunction&& reorderedFunction)
{
    PairedBenchmarkResult result;
    const ui64 batchSize = std::max<ui64>(1, (iterations + 9) / 10);
    ui64 offset = 0;
    ui64 batch = 0;
    while (offset < iterations) {
        const ui64 count = std::min(batchSize, iterations - offset);
        const auto runReference = [&]() {
            const BenchmarkResult part = benchmark(count, [&](ui64 index) {
                referenceFunction(offset + index);
            });
            result.reference.milliseconds += part.milliseconds;
            result.reference.operations += part.operations;
        };
        const auto runReordered = [&]() {
            const BenchmarkResult part = benchmark(count, [&](ui64 index) {
                reorderedFunction(offset + index);
            });
            result.reordered.milliseconds += part.milliseconds;
            result.reordered.operations += part.operations;
        };
        if ((batch % 2) == 0) {
            runReference();
            runReordered();
        } else {
            runReordered();
            runReference();
        }
        offset += count;
        ++batch;
    }
    return result;
}

void report(const char* label, BenchmarkResult result) {
    const double operationsPerSecond = result.operations * 1000.0 / result.milliseconds;
    const double microsecondsPerOperation = result.milliseconds * 1000.0 / result.operations;
    std::cout << label << ": " << result.milliseconds << " ms total, "
              << microsecondsPerOperation << " us/op, "
              << operationsPerSecond << " op/s\n";
}

void accumulate(GNN::GraphEvaluatorProfile& total, const GNN::GraphEvaluatorProfile& sample) {
    total.input_validation_us += sample.input_validation_us;
    total.input_projection_us += sample.input_projection_us;
    total.self_transform_us += sample.self_transform_us;
    total.attack_transform_us += sample.attack_transform_us;
    total.defence_transform_us += sample.defence_transform_us;
    total.aggregation_us += sample.aggregation_us;
    total.layer_norm_us += sample.layer_norm_us;
    total.pooling_and_value_head_us += sample.pooling_and_value_head_us;
}

void report_profile(const char* label, const GNN::GraphEvaluatorProfile& total, ui64 iterations) {
    const double scale = 1.0 / static_cast<double>(iterations);
    std::cout << label << " profile-pass stage mean (us/op): validation "
              << total.input_validation_us * scale
              << ", input projection " << total.input_projection_us * scale
              << ", self transform " << total.self_transform_us * scale
              << ", attack transform " << total.attack_transform_us * scale
              << ", defence transform " << total.defence_transform_us * scale
              << ", aggregation " << total.aggregation_us * scale
              << ", layer norm " << total.layer_norm_us * scale
              << ", pooling/value head " << total.pooling_and_value_head_us * scale
              << " (diagnostic timings; not additive to unprofiled latency)\n";
}

bool same_features(const GNN::GraphFeatures& left, const GNN::GraphFeatures& right) {
    return left.squares == right.squares
        && left.global == right.global
        && left.attack_sources == right.attack_sources
        && left.defence_sources == right.defence_sources
        && left.white_king_square == right.white_king_square
        && left.black_king_square == right.black_king_square;
}

struct OutputErrorStats {
    double absolute_sum = 0.0;
    double maximum_absolute = 0.0;
    double maximum_relative = 0.0;
    ui64 count = 0;

    void add(f32 reference, f32 candidate) {
        const double absolute = std::abs(static_cast<double>(reference) - candidate);
        const double denominator = std::max(
            std::abs(static_cast<double>(reference)),
            static_cast<double>(std::numeric_limits<f32>::epsilon()));
        absolute_sum += absolute;
        maximum_absolute = std::max(maximum_absolute, absolute);
        maximum_relative = std::max(maximum_relative, absolute / denominator);
        ++count;
    }

    double mean_absolute() const {
        return count == 0 ? 0.0 : absolute_sum / static_cast<double>(count);
    }
};

struct SearchResult {
    double milliseconds = 0.0;
    ui64 nodes = 0;
};

std::pair<SearchResult, SearchResult> benchmark_search_pair(
    const std::vector<Position>& positions,
    i32 searchDepth,
    ui64 searchRuns)
{
    NullBuffer nullBuffer;
    std::streambuf* previousOutput = std::cout.rdbuf(&nullBuffer);
    SearchResult reference{};
    SearchResult reordered{};
    bool referenceFirst = true;
    for (ui64 runIndex = 0; runIndex < searchRuns; ++runIndex) {
        for (const Position& position : positions) {
            const auto runOne = [&](bool useReordered, SearchResult& result) {
                Eval::set_graph_inference_reordered(useReordered);
                SearchInfo info;
                info.pos = position;
                info.limits.depth = searchDepth;
                TT.clear();
                TT.new_search();
                const auto started = std::chrono::steady_clock::now();
                Search::start(info);
                result.milliseconds += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - started).count();
                result.nodes += info.nodes;
            };
            if (referenceFirst) {
                runOne(false, reference);
                runOne(true, reordered);
            } else {
                runOne(true, reordered);
                runOne(false, reference);
            }
            referenceFirst = !referenceFirst;
        }
    }
    std::cout.rdbuf(previousOutput);
    Eval::set_graph_inference_reordered(false);
    return { reference, reordered };
}

void report_search(const char* label, SearchResult result) {
    const double nps = result.milliseconds > 0.0
        ? static_cast<double>(result.nodes) * 1000.0 / result.milliseconds
        : 0.0;
    std::cout << label << ": " << result.milliseconds << " ms, "
              << result.nodes << " nodes, " << nps << " NPS\n";
}

int run(const std::string& nnuePath,
        const std::string& graphPath,
        ui64 iterations,
        i32 searchDepth,
        ui64 searchRuns)
{
    BB::init();
    Zobrist::init();
    TT.resize(16);

    std::string error;
    if (!NNUE::load(nnuePath, error))
        throw std::runtime_error("NNUE load failed: " + error);
    if (!Eval::load_graph_model(graphPath, error))
        throw std::runtime_error("GraphEvaluator load failed: " + error);
    if (!Eval::set_graph_residual_enabled(true, error))
        throw std::runtime_error("GraphEvaluator enable failed: " + error);

    const std::vector<std::string> fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bq1rk1/ppp2ppp/2npbn2/4p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 8",
        "8/8/2k5/8/3K4/4P3/8/8 w - - 0 1",
        "r3k2r/8/8/3pP3/8/8/8/R3K2R w Kq d6 0 1",
    };
    std::vector<Position> positions;
    positions.reserve(fens.size());
    for (const std::string& fen : fens)
        positions.push_back(Position::from_fen(fen));

    GNN::GraphEncoder encoder;
    std::vector<GNN::GraphFeatures> graphFeatures(positions.size());
    bool encoderFeaturesIdentical = true;
    volatile f32 sink = 0.0f;
    const BenchmarkResult encoding = benchmark(iterations, [&](ui64 index) {
        const std::size_t positionIndex = index % positions.size();
        encoder.encode(positions[positionIndex], graphFeatures[positionIndex]);
        sink += graphFeatures[positionIndex].squares[positionIndex][13];
    });
    report("graph encoding", encoding);
    for (std::size_t positionIndex = 0; positionIndex < positions.size(); ++positionIndex) {
        encoder.encode(positions[positionIndex], graphFeatures[positionIndex]);
        GNN::GraphFeatures verificationFeatures;
        encoder.encode(positions[positionIndex], verificationFeatures);
        encoderFeaturesIdentical = encoderFeaturesIdentical
            && same_features(graphFeatures[positionIndex], verificationFeatures);
    }
    std::cout << "encoder features identical across paths: "
              << (encoderFeaturesIdentical ? "yes" : "no") << '\n';

    GNN::GraphEvaluator graphModel;
    if (!graphModel.load(graphPath, error))
        throw std::runtime_error("GraphEvaluator reload failed: " + error);
    const PairedBenchmarkResult inference = benchmark_paired(
        iterations,
        [&](ui64 index) {
            sink += graphModel.evaluate(graphFeatures[index % graphFeatures.size()]);
        },
        [&](ui64 index) {
            sink += graphModel.evaluate_reordered(graphFeatures[index % graphFeatures.size()]);
    });
    report("reference graph inference", inference.reference);
    report("reordered graph inference", inference.reordered);

    GNN::GraphEvaluatorProfile referenceProfile;
    GNN::GraphEvaluatorProfile reorderedProfile;
    benchmark_paired(
        iterations,
        [&](ui64 index) {
            GNN::GraphEvaluatorProfile profile;
            sink += graphModel.evaluate_profiled(graphFeatures[index % graphFeatures.size()], profile);
            accumulate(referenceProfile, profile);
        },
        [&](ui64 index) {
            GNN::GraphEvaluatorProfile profile;
            sink += graphModel.evaluate_reordered_profiled(
                graphFeatures[index % graphFeatures.size()], profile);
            accumulate(reorderedProfile, profile);
        });
    report_profile("reference", referenceProfile, iterations);
    report_profile("reordered", reorderedProfile, iterations);

    OutputErrorStats outputErrors;
    for (const GNN::GraphFeatures& features : graphFeatures) {
        outputErrors.add(graphModel.evaluate(features), graphModel.evaluate_reordered(features));
    }
    std::cout << "output error over " << outputErrors.count
              << " positions: max_abs=" << outputErrors.maximum_absolute
              << ", max_relative=" << outputErrors.maximum_relative
              << ", mean_abs=" << outputErrors.mean_absolute()
              << " (relative denominator max(|reference|, float epsilon))\n";

    const auto searchResults = benchmark_search_pair(positions, searchDepth, searchRuns);
    report_search("reference full search", searchResults.first);
    report_search("reordered full search", searchResults.second);

    const ui64 completedSearches = searchRuns * positions.size();
    std::cout << "search settings: depth " << searchDepth << ", "
              << completedSearches << " searches per implementation\n";
    std::cout << "benchmark sink: " << sink << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 6) {
        std::cerr << "usage: graph_eval_benchmark <baseline.nnue> <model.bmgnn> "
                     "[iterations=1000] [search-depth=8] [search-runs=2]\n";
        return 2;
    }
    try {
        const ui64 iterations = argc > 3 ? std::stoull(argv[3]) : 1000;
        const i32 depth = argc > 4 ? std::stoi(argv[4]) : 8;
        const ui64 searchRuns = argc > 5 ? std::stoull(argv[5]) : 2;
        if (iterations == 0 || depth < 1 || searchRuns == 0)
            throw std::invalid_argument("iterations, depth and search runs must be positive");
        return run(argv[1], argv[2], iterations, depth, searchRuns);
    } catch (const std::exception& exception) {
        std::cerr << "graph benchmark error: " << exception.what() << '\n';
        return 1;
    }
}
