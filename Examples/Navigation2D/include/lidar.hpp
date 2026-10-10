#pragma once

#include "components.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace navigation
{
struct ZoneDefinition;

/// Head-relative 360-degree scan: 61 samples at six-degree intervals.
/// The first and last directions intentionally coincide (legacy layout).
constexpr std::size_t kLidarBinCount = 61;

struct LidarScan
{
    /// The nearest ray hit, normalized as d / (d + reference), where reference
    /// is the hit circle's radius (at least 1) or the world half-diagonal.
    /// 1 means no obstacle; 0 means contact.
    std::array<float, kLidarBinCount> normalized{};
    /// Physical distance to the nearest intersection along each ray.
    /// Infinity means there is no intersection.
    std::array<double, kLidarBinCount> distances{};
};

/// Pure, deterministic geometry; independent of EnTT, PPO and rendering.
/// The square work-area boundary and circular forbidden zones are both sensed.
LidarScan scan_lidar(const State& own, double half_extent,
                     const std::vector<ZoneDefinition>& zones);

} // namespace navigation
