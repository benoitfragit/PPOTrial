#include "utils.hpp"

#include <math.h>

namespace ppo::model {

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