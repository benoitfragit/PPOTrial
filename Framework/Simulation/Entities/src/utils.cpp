/**
* @file Entities/src/utils.cpp
* @brief Implementation for utils.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "utils.hpp"

#include <math.h>

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
}
