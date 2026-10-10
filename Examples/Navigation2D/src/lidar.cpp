#include "lidar.hpp"
#include "navigation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace navigation
{
namespace
{
constexpr double pi = 3.14159265358979323846;
constexpr double epsilon = 1e-9;

// Always normalize the physically nearest intersection. Comparing normalized
// distances across different reference scales would make a farther boundary
// override a closer zone even when the ray actually hits the zone first.
void accumulate_hit(double distance, double scale, double& closest, float& normalized)
{
    if (distance < 0 || !std::isfinite(distance)) return;
    if (distance > closest) return;
    const auto candidate = static_cast<float>(distance / (distance + scale));
    if (distance < closest)
    {
        closest = distance;
        normalized = candidate;
    }
    else
    {
        normalized = std::min(normalized, candidate);
    }
}

double boundary_hit(Point position, double dx, double dy, double limit)
{
    double best = std::numeric_limits<double>::infinity();
    const auto try_hit = [&](double t, double x, double y)
    {
        if (t >= 0 && std::abs(x) <= limit + epsilon && std::abs(y) <= limit + epsilon)
            best = std::min(best, t);
    };
    if (std::abs(dx) > epsilon)
    {
        const double t1 = (limit - position.x) / dx;
        const double t2 = (-limit - position.x) / dx;
        try_hit(t1, limit, position.y + t1 * dy);
        try_hit(t2, -limit, position.y + t2 * dy);
    }
    if (std::abs(dy) > epsilon)
    {
        const double t1 = (limit - position.y) / dy;
        const double t2 = (-limit - position.y) / dy;
        try_hit(t1, position.x + t1 * dx, limit);
        try_hit(t2, position.x + t2 * dx, -limit);
    }
    return best;
}
}

LidarScan scan_lidar(const State& own, double half_extent,
                     const std::vector<ZoneDefinition>& zones)
{
    LidarScan scan;
    scan.normalized.fill(1.F);
    scan.distances.fill(std::numeric_limits<double>::infinity());
    const double boundary_scale = std::max(std::sqrt(2.0) * half_extent, 1.0);
    for (std::size_t i = 0; i < kLidarBinCount; ++i)
    {
        const double angle = own.heading - pi
            + static_cast<double>(i) * (2.0 * pi / static_cast<double>(kLidarBinCount - 1));
        const double dx = std::cos(angle);
        const double dy = std::sin(angle);
        auto& closest = scan.distances[i];
        auto& normalized = scan.normalized[i];
        accumulate_hit(boundary_hit(own.position, dx, dy, half_extent),
                  boundary_scale, closest, normalized);
        for (const auto& zone : zones)
        {
            const double mx = own.position.x - zone.center.x;
            const double my = own.position.y - zone.center.y;
            const double c = mx * mx + my * my - zone.radius * zone.radius;
            if (c <= 0)
            {
                closest = 0;
                normalized = 0;
                continue;
            }
            const double b = mx * dx + my * dy;
            const double delta = b * b - c;
            if (delta < 0) continue;
            const double root = std::sqrt(delta);
            accumulate_hit(-b - root, std::max(zone.radius, 1.0), closest, normalized);
            accumulate_hit(-b + root, std::max(zone.radius, 1.0), closest, normalized);
        }
    }
    return scan;
}
} // namespace navigation
