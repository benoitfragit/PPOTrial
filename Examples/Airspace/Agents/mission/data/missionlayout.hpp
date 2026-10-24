/**
* @file Agents/mission/data/missionlayout.hpp
* @brief Shared layout constants for mission observations and actions.
* @author PPOTrial
* @date 2026-03-30
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include <cstddef>

namespace rl {

/**
* @brief Central source of truth for the mission observation layout.
*
* The mission state is flattened before it is passed to PPO. These constants
* define how many per-track features are emitted, how many tracks are exposed,
* how many global scalar features are appended, and how many action values are
* expected back from the policy.
*/
struct MissionObservationLayout
{
    /**
    * @brief Number of scalar features exported for each ranked track.
    */
    static constexpr std::size_t kTrackFeatureCount{ 9U };
    /**
    * @brief Maximum number of ranked tracks exposed to the policy.
    */
    static constexpr std::size_t kMaxTracks{ 10U };
    /**
    * @brief Number of ammunition-related scalar features.
    */
    static constexpr std::size_t kAmmoFeatureCount{ 1U };
    /**
    * @brief Number of pseudo-lidar bins appended after the ammo feature.
    */
    static constexpr std::size_t kPseudoLidarBinCount{ 61U };
    /**
    * @brief Number of global scalar features appended after the track block.
    */
    static constexpr std::size_t kGlobalFeatureCount{
        kAmmoFeatureCount + kPseudoLidarBinCount
    };
    /**
    * @brief Number of action values expected back from the mission policy.
    */
    static constexpr std::size_t kActionCount{ 3U };
    /**
    * @brief Flattened mission observation size consumed by PPO.
    */
    static constexpr std::size_t kStateDimension{
        kMaxTracks * kTrackFeatureCount + kGlobalFeatureCount
    };
};

} /* rl */
