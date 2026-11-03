#pragma once

#include <bermuda_tuner/config.h>
#include <bermuda_tuner/types.h>

#include <cstdint>
#include <string>
#include <vector>

namespace Bermuda::Tuner {

bool load_dataset(const std::string& path, Dataset& dataset, std::string& error);
bool load_dataset(const std::string& path, Dataset& dataset, ImportStatistics& statistics,
                  std::string& error);
bool load_pgn_dataset(const std::string& path, Dataset& dataset,
                      std::string& error);
bool load_pgn_dataset(
    const std::string& path, Dataset& dataset, ImportStatistics& statistics,
    const std::vector<std::pair<std::string, std::string>>& metadata_filters,
    std::string& error);
bool filter_dataset_metadata(
    Dataset& dataset,
    const std::vector<std::pair<std::string, std::string>>& metadata_filters,
    ImportStatistics& statistics, std::string& error);
bool write_prepared_datasets(const Config& config, const Dataset& dataset,
                             std::string& error);
std::uint64_t dataset_fingerprint(const Dataset& dataset);
bool save_binary_dataset(const std::string& path, const Dataset& dataset,
                         std::string& error);
bool load_binary_dataset(const std::string& path, Dataset& dataset,
                         std::string& error);

} // namespace Bermuda::Tuner
