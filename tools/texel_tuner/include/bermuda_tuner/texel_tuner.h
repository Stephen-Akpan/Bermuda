#pragma once

#include <bermuda_tuner/config.h>

#include <functional>
#include <string>

namespace Bermuda::Tuner {

int run(const Config& config);
int run(const Config& config, const std::function<void(const std::string&)>& log);

} // namespace Bermuda::Tuner
