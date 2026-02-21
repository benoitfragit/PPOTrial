/**
* @file Entities/data/components.hpp
* @brief Declarations for components.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include <array>
#include <string>
#include <vector>

#include "entt/entt.hpp"

#include "entitytypes.hpp"

namespace ppo::model {

/**
* @brief Global environment parameters shared through the registry context.
*/
struct sEnvironmentState
{
    /**
    * @brief North-axis world size in meters.
    */
    float m_maxNorth{ 112500.F };
    /**
    * @brief East-axis world size in meters.
    */
    float m_maxEast{ 150000.F };
    /**
    * @brief Fixed simulation time step in seconds.
    */
    float m_dt{ 0.02F };
};

/**
* @brief Core kinematic and ownership state of one entity.
*/
struct sEntityState
{
    /**
    * @brief Unique ECS identifier of the entity.
    */
    EntityId m_entityId{ entt::null };
    /**
    * @brief North position in meters.
    */
    float m_x{ 0.F };
    /**
    * @brief East position in meters.
    */
    float m_y{ 0.F };
    /**
    * @brief North velocity component in meters per second.
    */
    float m_dx{ 0.F };
    /**
    * @brief East velocity component in meters per second.
    */
    float m_dy{ 0.F };
    /**
    * @brief Scalar speed in meters per second.
    */
    float m_speed{ 0.F };
    /**
    * @brief Heading angle in radians.
    */
    float m_rotation{ 0.F };
    /**
    * @brief Team side owning the entity.
    */
    eEntitySide m_side{ eEntitySide::Unknown };
    /**
    * @brief Functional type of the entity.
    */
    eEntityType m_type{ eEntityType::Objective };
    /**
    * @brief Destroyed state captured on the previous simulation step.
    */
    bool m_prevDestroyed{ false };
    /**
    * @brief Current destroyed state.
    */
    bool m_destroyed{ false };
    /**
    * @brief Identifier of the destroying entity when known.
    */
    EntityId m_destroyedById{ entt::null };
};

/**
* @brief Human-readable metadata used by tooling and rendering.
*/
struct sEntityMetadata
{
    /**
    * @brief Scenario-level entity name.
    */
    std::string m_name{};
    /**
    * @brief Archetype or platform name used to instantiate the entity.
    */
    std::string m_archetype{};
};

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
* @brief State snapshot published by the asynchronous evaluator.
*/
struct sEvaluatorState
{
    /**
    * @brief Number of steps executed per test.
    */
    unsigned int m_nbstep{ 0 };
    /**
    * @brief Total number of evaluation tests.
    */
    unsigned int m_nbtests{ 0 };
    /**
    * @brief Current evaluation score.
    */
    float m_notation{ 0.F };
    /**
    * @brief Progress ratio in the range [0, 1].
    */
    float m_progress{ 0.F };
    /**
    * @brief True while an evaluation is running.
    */
    bool m_inprogress{ false };
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
