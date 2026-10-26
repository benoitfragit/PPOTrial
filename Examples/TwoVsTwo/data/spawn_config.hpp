#pragma once

#include "spawn.hpp"

#include <nlohmann/json_fwd.hpp>
#include <optional>

namespace ppo::examples::twovstwo {

/**
 * @brief Parses and validates the twoVsTwo spawn configuration object.
 * @param settings JSON object stored under the scenario's twoVsTwo key.
 * @return Typed settings, or std::nullopt when a required value is invalid.
 */
std::optional<SpawnSettings> ParseSpawnSettings(
    nlohmann::json const& settings);

} // namespace ppo::examples::twovstwo
