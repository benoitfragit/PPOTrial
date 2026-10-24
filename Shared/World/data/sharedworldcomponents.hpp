/**
* @file sharedworldcomponents.hpp
* @brief Shared 2D world states; field layout and defaults preserved for legacy models.
*/
#pragma once

#include <array>
#include <vector>

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

/**
* @brief One persistent explosion event published by the environment.
*/
struct sExplosionEvent
{
    /**
    * @brief Unique event identifier.
    */
    unsigned long long m_eventId{ 0 };
    /**
    * @brief North position in meters.
    */
    float m_x{ 0.F };
    /**
    * @brief East position in meters.
    */
    float m_y{ 0.F };
    /**
    * @brief Down position in meters.
    */
    float m_z{ 0.F };
    /**
    * @brief Destroyed entity type used to scale the visual effect.
    */
    eEntityType m_type{ eEntityType::None };
    /**
    * @brief Event start time expressed in steady-clock seconds.
    */
    double m_startTime{ 0.0 };
    /**
    * @brief Base radius used to size the visual effect.
    */
    float m_baseRadius{ 0.9F };
    /**
    * @brief Vertical scaling factor used to make the plume taller.
    */
    float m_heightScale{ 1.F };
};

/**
* @brief Buffer of active explosion events published through the snapshot context.
*/
struct sExplosionEventBuffer
{
    /**
    * @brief Active explosion events.
    */
    std::vector<sExplosionEvent> m_events{};
};

} /* ppo::model */
