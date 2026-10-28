#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

namespace ppo::examples::twovstwo::defenders {

class PatrolPolicy;

/**
 * @brief Loads one compatible blue MAPPO Actor checkpoint.
 * @param checkpoint Version-4 Actor checkpoint path.
 * @return Shared deterministic inference policy.
 */
std::shared_ptr<PatrolPolicy>
LoadPatrolPolicy(std::filesystem::path const& checkpoint);

/**
 * @brief Creates an untrained blue MAPPO Actor with deterministic weights.
 * @param seed RLtools initialization seed.
 * @return Shared deterministic inference policy.
 */
std::shared_ptr<PatrolPolicy> CreateUntrainedPatrolPolicy(std::uint32_t seed);

} // namespace ppo::examples::twovstwo::defenders
