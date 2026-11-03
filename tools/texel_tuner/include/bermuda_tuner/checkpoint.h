#pragma once

#include <bermuda_tuner/optimizer.h>
#include <bermuda_tuner/config.h>

#include <cstdint>
#include <string>

namespace Bermuda::Tuner {

bool save_checkpoint(const Config& config, std::uint64_t fingerprint,
                     const OptimizerState& state, std::string& error);
bool load_checkpoint(const Config& config, std::uint64_t fingerprint,
                     OptimizerState& state, std::string& error);

} // namespace Bermuda::Tuner
