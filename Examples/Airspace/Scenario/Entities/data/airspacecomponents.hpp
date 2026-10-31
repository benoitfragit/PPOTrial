/**
 * @file Examples/Airspace/Scenario/Entities/data/airspacecomponents.hpp
 * @brief Construction, données et arbitrage du scénario airspacecomponents.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once
// Historic Airspace visuals remain local to this example; shared ECS states
// keep their exact field layouts in SharedWorld.
#include "sharedworldcomponents.hpp"
#include <vector>

namespace ppo::model {

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
