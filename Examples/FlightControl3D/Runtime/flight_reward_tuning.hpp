/**
 * @file flight_reward_tuning.hpp
 * @brief Desktop editing of the existing reward channel weights.
 *
 * Role in the chain: desktop reward sliders -> **tuning** -> session -> learner and
 * environment. An edit is validated against the full configuration, written atomically to
 * the scenario JSON and recorded in the checkpoint signature; the session applies it at the
 * next PPO rollout boundary. Only the existing weights can be edited: no channel, formula
 * or normalization is added.
 *
 * Read next: flight_reward_tuning.cpp, flight_session.cpp (SetRewardTuning).
 */
#pragma once

#include "flight_config.hpp"

namespace ppo::examples::flight {

/** @brief Number of editable settings: one weight per reward channel, in Reward order. */
inline constexpr std::size_t RewardTuningCount = RewardCount;

/** @brief Read an existing channel weight (zero for an unknown index). */
double RewardTuningValue(RewardSettings const& settings, std::size_t index) noexcept;
/**
 * @brief Validate a single adjustment and update the checkpoint contract.
 * @param current Active or pending configuration.
 * @param index Existing channel index.
 * @param value Requested value.
 * @return Updated validated configuration.
 * @throws std::invalid_argument for an unknown index, a non-finite value or an invalid configuration.
 */
Config WithRewardTuning(Config const& current, std::size_t index, double value);
/**
 * @brief Atomically save an existing weight to the scenario JSON.
 * @param path Scenario file to update.
 * @param index Existing channel index.
 * @param value Validated value.
 * @throws std::runtime_error when the file cannot be read or replaced; the original is then preserved.
 */
void SaveRewardTuning(std::filesystem::path const& path, std::size_t index, double value);

} // namespace ppo::examples::flight
