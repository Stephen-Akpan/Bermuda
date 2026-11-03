#include "catch_amalgamated.hpp"

#include <bermuda_tuner/checkpoint.h>
#include <bermuda_tuner/dataset.h>
#include <bermuda_tuner/objective.h>
#include <bermuda_tuner/optimizer.h>
#include <bermuda_tuner/parameter_space.h>
#include <bermuda_tuner/report.h>

#include "bitboard.h"
#include "zobrist.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

namespace {

void initialize_tuner_tables() {
    static const bool initialized = [] {
        BB::init();
        Zobrist::init();
        return true;
    }();
    (void)initialized;
}

std::string temporary_path(const std::string& suffix) {
    const auto ticks = std::chrono::steady_clock::now()
                           .time_since_epoch().count();
    return (std::filesystem::temp_directory_path()
            / ("bermuda-tuner-" + std::to_string(ticks) + suffix)).string();
}

} // namespace

TEST_CASE("Tuner parameter descriptors and conversion preserve schema order",
          "[tuner][parameters]") {
    using namespace Bermuda::Tuner;
    initialize_tuner_tables();

    const auto& descriptors = parameter_descriptors();
    REQUIRE(descriptors.size() == ParameterCount);
    CHECK(descriptors[0].name == "material");
    CHECK(descriptors[8].name == "tempo");
    CHECK(descriptors[9].name == "pawn_mg");
    CHECK(descriptors[18].name == "queen_eg");

    ParameterVector values{};
    for (std::size_t i = 0; i < values.size(); ++i)
        values[i] = descriptors[i].minimum
                  + (descriptors[i].maximum - descriptors[i].minimum)
                    * (static_cast<double>(i + 1) / 20.0);
    const auto round_trip = to_vector(to_parameters(values));
    CHECK(round_trip == values);

    CHECK(parameter_enabled(0, "material"));
    CHECK_FALSE(parameter_enabled(1, "material"));
    CHECK_FALSE(parameter_enabled(0, "piece-values"));
    CHECK(parameter_enabled(9, "piece-values"));
    CHECK(parameter_enabled(18, "piece-values"));
    CHECK(parameter_enabled(1, "positional"));
    CHECK_FALSE(parameter_enabled(0, "positional"));
    CHECK(parameter_enabled(9, "piece-values-mg"));
    CHECK_FALSE(parameter_enabled(14, "piece-values-mg"));
    CHECK(parameter_enabled(14, "piece-values-eg"));
    CHECK(parameter_enabled(4, "mobility"));
    CHECK_FALSE(parameter_enabled(4, "king-safety"));
    CHECK(parameter_enabled(7, "king-safety"));
    CHECK(parameter_enabled(12, "rook-mg"));
    CHECK(parameter_family_supported("pawn_structure"));
    CHECK_FALSE(parameter_enabled(19, "all"));

    values[0] = 4.0;
    values[9] = 0.1;
    clamp_parameters(values);
    CHECK(values[0] == 3.0);
    CHECK(values[9] == 0.7);
    std::string error;
    CHECK(validate_parameters(values, error));
    values[0] = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(validate_parameters(values, error));
    CHECK(error.find("material") != std::string::npos);
}

TEST_CASE("PGN importer retains pre-move positions and game results",
          "[tuner][pgn]") {
    using namespace Bermuda::Tuner;
    initialize_tuner_tables();
    const std::string path = temporary_path(".pgn");
    {
        std::ofstream output(path);
        REQUIRE(output.good());
        output << "[Event \"one\"]\n[White \"Alice \\\"A\\\"\"]\n"
               << "[TimeControl \"600+0\"]\n[Result \"1-0\"]\n\n"
               << "1. e4 e5 2. Nf3 Nc6 1-0\n"
               << "[Event \"two\"]\n[White \"Bob\"]\n"
               << "[TimeControl \"600+0\"]\n[Result \"0-1\"]\n\n"
               << "1. d4 d5 2. c4 e6 0-1\n"
               << "[Event \"three\"]\n[White \"Carol\"]\n"
               << "[TimeControl \"300+0\"]\n[Result \"1-0\"]\n\n"
               << "1. e4 d5 2. exd5 Qxd5 1-0\n";
    }

    Dataset dataset;
    ImportStatistics statistics;
    std::string error;
    const bool success = load_pgn_dataset(
        path, dataset, statistics, {{"TimeControl", "600+0"}}, error);
    std::remove(path.c_str());

    REQUIRE(success);
    CHECK(dataset.game_ids.size() == 2);
    REQUIRE(dataset.samples.size() == 8);
    CHECK(statistics.games_encountered == 3);
    CHECK(statistics.games_accepted == 2);
    CHECK(statistics.games_skipped == 1);
    CHECK(statistics.skip_reasons.at("metadata_filter") == 1);
    CHECK(statistics.positions_generated == 8);
    CHECK(statistics.minimum_positions_per_game == 4);
    CHECK(statistics.median_positions_per_game == 4.0);
    CHECK(statistics.maximum_positions_per_game == 4);
    CHECK(dataset.games[0].white == "Alice \"A\"");
    CHECK(dataset.games[0].tags.at("TimeControl") == "600+0");
    CHECK(dataset.games[0].source == path);
    CHECK(dataset.samples[0].result == 1.0);
    CHECK(dataset.samples[3].result == 1.0);
    CHECK(dataset.samples[4].result == 0.0);
    CHECK(dataset.samples[7].result == 0.0);
    CHECK(dataset.samples[0].game != dataset.samples[4].game);
    CHECK(dataset.samples[0].position.to_fen()
          == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

TEST_CASE("Prepared dataset loader rejects conflicting game results",
          "[tuner][dataset]") {
    using namespace Bermuda::Tuner;
    initialize_tuner_tables();
    const std::string path = temporary_path(".data");
    {
        std::ofstream output(path);
        REQUIRE(output.good());
        output << "same 1-0 rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n"
               << "same 0-1 rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1\n"
               << "other 1-0 rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n";
    }

    Dataset dataset;
    std::string error;
    CHECK_FALSE(load_dataset(path, dataset, error));
    CHECK_FALSE(error.empty());
    std::remove(path.c_str());

    Dataset valid;
    valid.game_ids = {"one"};
    valid.samples.push_back({
        Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        0.5, 0
    });
    CHECK(valid.validate(error));
    valid.game_ids.push_back("empty");
    CHECK_FALSE(valid.validate(error));
    CHECK(error.find("no positions") != std::string::npos);
    valid.game_ids.pop_back();
    valid.samples[0].result = 0.25;
    CHECK_FALSE(valid.validate(error));
}

TEST_CASE("Binary datasets preserve metadata and reject corruption",
          "[tuner][dataset][binary]") {
    using namespace Bermuda::Tuner;
    initialize_tuner_tables();
    Dataset dataset;
    dataset.source_description = "fixture source";
    dataset.game_ids = {"first", "second"};
    GameMetadata first;
    first.id = "first";
    first.event = "Event A";
    first.tags["Event"] = first.event;
    dataset.games.push_back(first);
    GameMetadata second;
    second.id = "second";
    second.white = "White B";
    second.tags["White"] = second.white;
    dataset.games.push_back(second);
    const std::string fen =
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    dataset.samples.push_back({Position::from_fen(fen), 1.0, 0});
    dataset.samples.push_back({Position::from_fen(fen), 0.0, 1});

    const std::string path = temporary_path(".btd");
    std::string error;
    REQUIRE(save_binary_dataset(path, dataset, error));
    Dataset loaded;
    REQUIRE(load_dataset(path, loaded, error));
    CHECK(loaded.source_description == dataset.source_description);
    CHECK(loaded.games[0].tags.at("Event") == "Event A");
    CHECK(loaded.games[1].white == "White B");
    CHECK(dataset_fingerprint(loaded) == dataset_fingerprint(dataset));

    {
        std::ofstream corrupt(path, std::ios::binary | std::ios::app);
        REQUIRE(corrupt.good());
        corrupt.put('x');
    }
    CHECK_FALSE(load_binary_dataset(path, loaded, error));
    CHECK(error.find("length") != std::string::npos);
    std::remove(path.c_str());
    std::remove((path + ".bak").c_str());
    std::remove((path + ".tmp").c_str());
}

TEST_CASE("Prepared text splits retain tags, source, and game membership",
          "[tuner][dataset][metadata]") {
    using namespace Bermuda::Tuner;
    initialize_tuner_tables();
    Dataset dataset;
    dataset.source_description = "source with spaces.pgn";
    const Position position = Position::from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    for (std::size_t i = 0; i < 4; ++i) {
        GameMetadata metadata;
        metadata.id = "game-" + std::to_string(i);
        metadata.white = "Player \"Q\" " + std::to_string(i);
        metadata.result = i % 2 == 0 ? "1-0" : "0-1";
        metadata.source = dataset.source_description;
        metadata.tags["Event"] = "Source\tCohort";
        metadata.tags["White"] = metadata.white;
        metadata.tags["Result"] = metadata.result;
        dataset.game_ids.push_back(metadata.id);
        dataset.games.push_back(metadata);
        dataset.samples.push_back({
            position, i % 2 == 0 ? 1.0 : 0.0, i
        });
    }

    Config config;
    config.train_output_path = temporary_path(".train.txt");
    config.validation_output_path = temporary_path(".validation.txt");
    config.validation_percent = 50;
    config.split_seed = 7;
    std::string error;
    REQUIRE(write_prepared_datasets(config, dataset, error));

    Dataset training;
    Dataset validation;
    REQUIRE(load_dataset(config.train_output_path, training, error));
    REQUIRE(load_dataset(config.validation_output_path, validation, error));
    CHECK(dataset_fingerprint(training) != dataset_fingerprint(validation));
    CHECK(training.source_description == dataset.source_description);
    CHECK(validation.source_description == dataset.source_description);
    REQUIRE(training.games.size() == training.game_ids.size());
    REQUIRE(validation.games.size() == validation.game_ids.size());
    CHECK(training.games[0].tags.at("Event") == "Source\tCohort");

    ImportStatistics statistics;
    REQUIRE(filter_dataset_metadata(training, {{"Event", "Source\tCohort"}},
                                    statistics, error));
    CHECK(statistics.games_accepted == training.game_ids.size());
    REQUIRE(filter_dataset_metadata(validation, {{"Event", "Source\tCohort"}},
                                    statistics, error));
    CHECK(statistics.games_accepted == validation.game_ids.size());

    std::remove(config.train_output_path.c_str());
    std::remove(config.validation_output_path.c_str());
    std::remove((config.train_output_path + ".manifest.json").c_str());
    std::remove((config.train_output_path + ".manifest.json.tmp").c_str());
}

TEST_CASE("Game-weighted objective gives each game equal influence",
          "[tuner][objective]") {
    using namespace Bermuda::Tuner;
    initialize_tuner_tables();
    Dataset dataset;
    dataset.game_ids = {"long", "short"};
    const Position initial = Position::from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    const Position black_to_move = Position::from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    dataset.samples = {
        {initial, 1.0, 0}, {initial, 1.0, 0},
        {black_to_move, 0.5, 1}
    };
    const Eval::Parameters parameters;
    const double long_error = std::pow(
        predicted_score(dataset.samples[0], parameters, 1.13) - 1.0, 2.0);
    const double short_error = std::pow(
        predicted_score(dataset.samples[2], parameters, 1.13) - 0.5, 2.0);
    double position_loss = 0.0;
    double game_loss = 0.0;
    std::string error;
    REQUIRE(objective(dataset, {0, 1, 2}, parameters, 1.13,
                      WeightingMode::Position, position_loss, error));
    REQUIRE(objective(dataset, {0, 1, 2}, parameters, 1.13,
                      WeightingMode::Game, game_loss, error));
    CHECK(position_loss == Catch::Approx((2.0 * long_error + short_error) / 3.0));
    CHECK(game_loss == Catch::Approx((long_error + short_error) / 2.0));
    CHECK(game_loss != Catch::Approx(position_loss));
}

TEST_CASE("Experiment reports and CSV progress are repeatable",
          "[tuner][report]") {
    using namespace Bermuda::Tuner;
    initialize_tuner_tables();
    Dataset dataset;
    dataset.source_description = "report fixture";
    dataset.game_ids = {"train", "validation"};
    for (const std::string& id : dataset.game_ids) {
        GameMetadata metadata;
        metadata.id = id;
        metadata.tags["Event"] = "fixture";
        dataset.games.push_back(metadata);
    }
    const Position position = Position::from_fen(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    dataset.samples = {{position, 1.0, 0}, {position, 0.0, 1}};
    Config config;
    config.report_directory = temporary_path(".reports");
    ImportStatistics statistics;
    statistics.games_encountered = 2;
    statistics.games_accepted = 2;
    statistics.positions_generated = 2;
    statistics.mean_positions_per_game = 1.0;
    OptimizerState state;
    state.completed_iterations = 3;
    state.best_iteration = 2;
    state.initial_training = 0.4;
    state.current_training = 0.3;
    state.initial_validation = 0.5;
    state.current_validation = 0.45;
    state.best_validation = 0.42;
    state.initial_weights = to_vector(Eval::Parameters{});
    state.weights = state.initial_weights;
    state.best_weights = state.weights;

    std::string error;
    REQUIRE(write_experiment_config(config, dataset, statistics, {0}, {1},
                                    state.initial_weights, error));
    REQUIRE(write_progress_csv(config, state, state.current_validation, error));
    REQUIRE(write_progress_csv(config, state, state.current_validation, error));
    REQUIRE(write_experiment_report(config, dataset, statistics, state, 1.5,
                                    error));
    std::ifstream config_input(
        std::filesystem::path(config.report_directory) / "config.json");
    std::ifstream report_input(
        std::filesystem::path(config.report_directory) / "report.json");
    std::ifstream progress_input(
        std::filesystem::path(config.report_directory) / "progress.csv");
    const std::string config_json((std::istreambuf_iterator<char>(config_input)),
                                  std::istreambuf_iterator<char>());
    const std::string report_json((std::istreambuf_iterator<char>(report_input)),
                                  std::istreambuf_iterator<char>());
    const std::string progress((std::istreambuf_iterator<char>(progress_input)),
                               std::istreambuf_iterator<char>());
    config_input.close();
    report_input.close();
    progress_input.close();
    CHECK(config_json.find("\"weighting\": \"position\"") != std::string::npos);
    CHECK(report_json.find("\"best_iteration\": 2") != std::string::npos);
    CHECK(report_json.find("\"initial_training_loss\": 0.4") != std::string::npos);
    CHECK(std::count(progress.begin(), progress.end(), '\n') == 2);
    std::filesystem::remove_all(config.report_directory);
}

TEST_CASE("Texel objective validates index sets and handles both sides",
          "[tuner][objective]") {
    using namespace Bermuda::Tuner;
    initialize_tuner_tables();
    Dataset dataset;
    dataset.samples.push_back({
        Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        1.0, 0
    });
    dataset.samples.push_back({
        Position::from_fen(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1"),
        0.0, 1
    });
    dataset.game_ids = {"white", "black"};

    double loss = 0.0;
    std::string error;
    const std::vector<std::size_t> indices = {0, 1};
    REQUIRE(objective(dataset, indices, Eval::Parameters{}, 1.13, loss, error));
    const double white = predicted_score(dataset.samples[0],
                                         Eval::Parameters{}, 1.13);
    const double black = predicted_score(dataset.samples[1],
                                         Eval::Parameters{}, 1.13);
    CHECK(white + black == Catch::Approx(1.0).margin(1e-12));
    CHECK(loss == Catch::Approx((std::pow(white - 1.0, 2.0)
                               + std::pow(black, 2.0)) / 2.0).margin(1e-12));
    CHECK_FALSE(objective(dataset, {}, Eval::Parameters{}, 1.13, loss, error));
    CHECK_FALSE(objective(dataset, {2}, Eval::Parameters{}, 1.13, loss, error));
    CHECK_FALSE(objective(dataset, indices, Eval::Parameters{}, 0.0, loss, error));
}

TEST_CASE("SPSA is deterministic and leaves inactive parameters unchanged",
          "[tuner][optimizer]") {
    using namespace Bermuda::Tuner;
    initialize_tuner_tables();
    Dataset dataset;
    dataset.samples.push_back({
        Position::from_fen(
            "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
        0.0, 0
    });
    dataset.samples.push_back({
        Position::from_fen(
            "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
        0.0, 1
    });
    Config config;
    config.iterations = 10;
    config.batch_size = 1;
    config.family = "material";
    config.optimizer_seed = 19;

    OptimizerState first;
    first.random.seed(config.optimizer_seed);
    first.weights = to_vector(Eval::Parameters{});
    first.best_weights = first.weights;
    first.best_validation = 1.0;
    OptimizerState second = first;
    second.random.seed(config.optimizer_seed);
    std::string error;
    const ProgressCallback no_progress;
    REQUIRE(run_spsa(dataset, {0}, {1}, config, first, no_progress, error));
    REQUIRE(run_spsa(dataset, {0}, {1}, config, second, no_progress, error));
    CHECK(first.weights == second.weights);
    CHECK(first.best_weights == second.best_weights);
    CHECK(first.weights[0] != 1.0);
    CHECK(std::equal(first.weights.begin() + 1, first.weights.end(),
                     second.weights.begin() + 1));
    for (std::size_t i = 1; i < ParameterCount; ++i)
        CHECK(first.weights[i] == 1.0);
    CHECK(first.completed_iterations == config.iterations);
}

TEST_CASE("Optimizer checkpoints preserve state and reject mismatched seeds",
          "[tuner][checkpoint]") {
    using namespace Bermuda::Tuner;
    Config config;
    config.checkpoint_path = temporary_path(".checkpoint");
    config.iterations = 10;
    config.batch_size = 32;
    config.optimizer_seed = 19;

    OptimizerState state;
    state.initial_weights = to_vector(Eval::Parameters{});
    state.weights = state.initial_weights;
    state.best_weights = state.initial_weights;
    state.completed_iterations = 4;
    state.weights[0] = 0.8;
    state.best_weights[0] = 0.7;
    state.best_iteration = 2;
    state.best_validation = 0.2;
    state.initial_validation = 0.3;
    state.current_training = 0.1;
    state.random.seed(123);
    (void)state.random();
    std::string error;
    REQUIRE(save_checkpoint(config, 12345, state, error));

    OptimizerState loaded;
    loaded.initial_weights = state.initial_weights;
    REQUIRE(load_checkpoint(config, 12345, loaded, error));
    CHECK(loaded.completed_iterations == state.completed_iterations);
    CHECK(loaded.best_iteration == state.best_iteration);
    CHECK(loaded.initial_weights == state.initial_weights);
    CHECK(loaded.weights == state.weights);
    CHECK(loaded.best_weights == state.best_weights);
    CHECK(loaded.best_validation == state.best_validation);
    CHECK(loaded.random() == state.random());

    config.optimizer_seed += 1;
    CHECK_FALSE(load_checkpoint(config, 12345, loaded, error));

    std::remove(config.checkpoint_path.c_str());
    std::remove((config.checkpoint_path + ".bak").c_str());
    std::remove((config.checkpoint_path + ".tmp").c_str());
}
