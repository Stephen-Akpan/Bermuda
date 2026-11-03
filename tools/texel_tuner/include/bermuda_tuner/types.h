#pragma once

#include "position.h"

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace Bermuda::Tuner {

struct Sample {
    Position position;
    double result = 0.5;
    std::size_t game = 0;
};

struct GameMetadata {
    std::string id;
    std::string event;
    std::string site;
    std::string date;
    std::string round;
    std::string white;
    std::string black;
    std::string result;
    std::string eco;
    std::string opening;
    std::string time_control;
    std::string termination;
    std::string variant;
    std::string source;
    std::map<std::string, std::string> tags;
};

struct ImportStatistics {
    std::size_t games_encountered = 0;
    std::size_t games_accepted = 0;
    std::size_t games_skipped = 0;
    std::size_t games_rejected = 0;
    std::size_t positions_generated = 0;
    std::map<std::string, std::size_t> missing_metadata;
    std::map<std::string, std::size_t> skip_reasons;
    std::size_t minimum_positions_per_game = 0;
    double median_positions_per_game = 0.0;
    std::size_t maximum_positions_per_game = 0;
    double mean_positions_per_game = 0.0;
    std::vector<std::string> warnings;
};

struct Dataset {
    std::vector<Sample> samples;
    std::vector<std::string> game_ids;
    std::vector<GameMetadata> games;
    std::string source_description;

    bool validate(std::string& error) const;
};

} // namespace Bermuda::Tuner
