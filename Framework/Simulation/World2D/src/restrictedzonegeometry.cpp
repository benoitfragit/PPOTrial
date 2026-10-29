#include "restrictedzonegeometry.hpp"

#include <algorithm>
#include <cmath>

namespace ppo::model {
namespace {

bool Valid(PlanarPosition position) noexcept
{
    return std::isfinite(position.north) && std::isfinite(position.east);
}

} // namespace

float RestrictedZoneWarningRadius(float lethalRadius) noexcept
{
    return std::isfinite(lethalRadius) && lethalRadius > 0.F
        ? kRestrictedZoneWarningRadiusScale * lethalRadius : 0.F;
}

bool IsInsideRestrictedZone(PlanarPosition point,
                            PlanarPosition centre,
                            float lethalRadius) noexcept
{
    if (!Valid(point) || !Valid(centre) || !std::isfinite(lethalRadius)
        || lethalRadius <= 0.F)
        return false;
    float const north = point.north - centre.north;
    float const east = point.east - centre.east;
    return north * north + east * east <= lethalRadius * lethalRadius;
}

bool CrossesRestrictedZone(PlanarPosition previous,
                           PlanarPosition current,
                           PlanarPosition centre,
                           float lethalRadius) noexcept
{
    if (!Valid(previous) || !Valid(current) || !Valid(centre)
        || !std::isfinite(lethalRadius) || lethalRadius <= 0.F)
        return false;
    if (IsInsideRestrictedZone(previous, centre, lethalRadius)
        || IsInsideRestrictedZone(current, centre, lethalRadius))
        return true;

    float const segmentNorth = current.north - previous.north;
    float const segmentEast = current.east - previous.east;
    float const lengthSquared = segmentNorth * segmentNorth
                              + segmentEast * segmentEast;
    if (lengthSquared <= 1.e-8F)
        return false;
    float const fromStartNorth = centre.north - previous.north;
    float const fromStartEast = centre.east - previous.east;
    float const fraction = std::clamp(
        (fromStartNorth * segmentNorth + fromStartEast * segmentEast)
            / lengthSquared,
        0.F, 1.F);
    PlanarPosition const closest{
        previous.north + fraction * segmentNorth,
        previous.east + fraction * segmentEast};
    return IsInsideRestrictedZone(closest, centre, lethalRadius);
}

} // namespace ppo::model
