#pragma once

#include "spawn.hpp"

#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>

namespace ppo::examples::bluedefense {

/**
 * @brief Parses and validates the blueDefense spawn configuration object.
 * @param settings JSON object stored under the scenario's blueDefense key.
 * @param diagnostic Optional destination for a precise validation error.
 * @return Typed settings, or std::nullopt when a required value is invalid.
 */
std::optional<SpawnSettings> ParseSpawnSettings(
    nlohmann::json const& settings,
    std::string* diagnostic = nullptr);

} // namespace ppo::examples::bluedefense
