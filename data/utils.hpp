#pragma once

namespace ppo::model {

constexpr double PI                      = 3.1415926536;
constexpr double TWOPI                   = 2.0 * PI;
constexpr double HALFPI                  = PI / 2.0;

constexpr double DEG_TO_RAD              = PI /180.0;

constexpr float  KT_TO_MS                = 0.51444444F;
constexpr float  MS_TO_KT                = 1.94384F;

constexpr double EULER                   = 2.7182818284;

constexpr float  MIN_VELOCITY            = 100.F * KT_TO_MS;
constexpr float  MIN_CONSUMPTION         = 0.5F; // Kg.s-1
constexpr float  LINEAR_CONSUMPTION      = 0.005F; // Kg.s-1/m.s-1
constexpr float  QUADRA_CONSUMPTION      = 0.0004F;
constexpr double EPS                     = 1e-8;

double wrapMinusPi2Pi(double az);

}