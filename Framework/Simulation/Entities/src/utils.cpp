/**
* @file Entities/src/utils.cpp
* @brief Implementation for utils.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "utils.hpp"
#include "components.hpp"

#include <math.h>
#include <random>

namespace ppo::model {

/**
* @brief Wraps one angle in radians to the [-pi, pi] interval.
* @param az Input angle in radians.
* @return Wrapped angle in radians.
*/
double
wrapMinusPi2Pi(double az)
{
    az = fmod(az, TWOPI);
    if (az > PI)
        az -= TWOPI;
    else if (az < -PI)
        az += TWOPI;

    return az;
}

/**
* @brief The registry's initialization engine, created from the system's entropy if absent.
*/
std::mt19937_64&
InitializationEngine(entt::registry& registry)
{
    if (!registry.ctx().contains<sInitializationRandom>())
    {
        std::random_device entropy;
        std::seed_seq seed{ entropy(), entropy(), entropy(), entropy() };
        registry.ctx().emplace<sInitializationRandom>().m_engine.seed(seed);
    }
    return registry.ctx().get<sInitializationRandom>().m_engine;
}

}
