#pragma once
#include "flight_config.hpp"

namespace ppo::examples::flight {
inline constexpr std::size_t RewardTuningCount=RewardCount;

/** @brief Read an existing channel weight or approach-shaping coefficient. */
double RewardTuningValue(RewardSettings const& settings,std::size_t index) noexcept;
/**
 * @brief Validate a single adjustment and update the checkpoint contract.
 * @param current Active or pending configuration.
 * @param index Existing channel or approach coefficient index.
 * @param value Requested value.
 * @return Updated validated configuration.
 */
Config WithRewardTuning(Config const& current,std::size_t index,double value);
/**
 * @brief Atomically save an existing weight to the scenario JSON.
 * @param path Scenario file to update.
 * @param index Existing channel or approach coefficient index.
 * @param value Validated value.
 */
void SaveRewardTuning(std::filesystem::path const& path,std::size_t index,double value);
} // namespace ppo::examples::flight
