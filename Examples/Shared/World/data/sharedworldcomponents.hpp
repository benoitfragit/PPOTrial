/**
* @file sharedworldcomponents.hpp
* @brief Shared 2D world states; field layout and defaults preserved for legacy models.
*/
#pragma once

#include <array>
#include <cstddef>

#include "entt/entt.hpp"
#include "entitytypes.hpp"

namespace ppo::model {

/**
* @brief Situation awareness and engagement state for guidance logic.
*/
struct sSNAState
{
    /**
    * @brief Current target entity identifier.
    */
    EntityId m_target{ entt::null };
    /**
    * @brief Estimated kill probability against the target.
    */
    float m_pk{ 0.F };
    /**
    * @brief Target north position in meters.
    */
    float m_target_x{ 0.F };
    /**
    * @brief Target east position in meters.
    */
    float m_target_y{ 0.F };
    /**
    * @brief Target north velocity in meters per second.
    */
    float m_target_dx{ 0.F };
    /**
    * @brief Target east velocity in meters per second.
    */
    float m_target_dy{ 0.F };
    /**
    * @brief Closing speed relative to the target in meters per second.
    */
    float m_closingspeed{ 0.F };
    /**
    * @brief Current line-of-sight angle in radians.
    */
    float m_target_los{ 0.F };
    /**
    * @brief Line-of-sight rate in radians per second.
    */
    float m_target_los_rate{ 0.F };
    /**
    * @brief Maximum no-escape-zone reference for the platform.
    */
    float m_maxNoescapeZone{ 8000.F };
    /**
    * @brief Escape-zone range in meters.
    */
    float m_escapezone{ 0.F };
    /**
    * @brief No-escape-zone range in meters.
    */
    float m_noescapezone{ 0.F };
    /**
    * @brief Maximum launch angle in radians.
    */
    float m_maxShootAngle{ 0.F };
    /**
    * @brief Remaining missile count.
    */
    unsigned short m_nbmissiles{ 0 };
    /**
    * @brief Maximum missile count carried by the platform.
    */
    unsigned short m_maxnbmissiles{ 0 };
};

/**
* @brief Manual commands applied to the guidance and flight controller.
*/
struct sSNAControlState
{
    /**
    * @brief Longitudinal acceleration command.
    */
    float m_accCmd{ 0.F };
    /**
    * @brief Heading-rate command.
    */
    float m_rotSpeedCmd{ 0.F };
};

/**
* @brief Pseudo-lidar collision sketch used by mission logic and rendering.
*/
struct sPseudoLidarState
{
    /**
    * @brief Normalized collision ranges for each lidar bin.
    */
    std::array<float, 61> m_nearestCollisionRanges{};
    /**
    * @brief Raw collision distances for each lidar bin.
    */
    std::array<float, 61> m_nearestCollisionDistances{};
    /**
    * @brief Angular span covered by the pseudo-lidar in radians.
    */
    float m_angleSpanRadians{ 6.2831855F };
};

/**
* @brief Simplified flight model state.
*/
struct sFlightModelState
{
    /**
    * @brief Current longitudinal acceleration.
    */
    float m_acc{ 0.F };
    /**
    * @brief Current turn acceleration.
    */
    float m_accpsi{ 0.F };
    /**
    * @brief Maximum longitudinal acceleration.
    */
    float m_maxAcc{ 0.F };
    /**
    * @brief Maximum turn acceleration.
    */
    float m_maxAccRot{ 0.F };
    /**
    * @brief Maximum allowed speed.
    */
    float m_maxSpeed{ 0.F };
    /**
    * @brief Minimum allowed speed.
    */
    float m_minSpeed{ 0.F };
    /**
    * @brief Remaining fuel quantity.
    */
    float m_fuel{ 0.F };
    /**
    * @brief Instantaneous fuel consumption.
    */
    float m_fuelconsumption{ 0.F };
    /**
    * @brief Maximum fuel quantity.
    */
    float m_maxfuel{ 0.F };
};

// Rendering metadata derived from the current world bounds (sEnvironmentState).
// Snapshot-only: no independent geometry or collision behavior is imposed.
struct sDefendedArea
{
    float m_north{0.F};
    float m_east{0.F};
    float m_halfNorth{0.F};
    float m_halfEast{0.F};
};

/**
 * @brief Snapshot-only samples used to draw the analytical patrol curves.
 *
 * These points have no navigation or training semantics: the controller uses
 * the continuous geometry directly. The fixed-size storage avoids allocations
 * while copying and rendering snapshots.
 */
struct sPatrolTrackOverlay final
{
    /** @brief Number of samples reserved per closed defender curve. */
    static constexpr std::size_t SampleCapacity = 97U;

    /** @brief North coordinates, indexed by defender role then sample. */
    std::array<std::array<float, SampleCapacity>, 2> m_north{};
    /** @brief East coordinates, indexed by defender role then sample. */
    std::array<std::array<float, SampleCapacity>, 2> m_east{};
    /** @brief Number of valid entries in each coordinate array. */
    std::size_t m_sampleCount{};
};

} /* ppo::model */
