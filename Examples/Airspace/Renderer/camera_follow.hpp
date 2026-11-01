/**
 * @file camera_follow.hpp
 * @brief Frame-rate-independent camera-follow helpers.
 */
#pragma once

#include <algorithm>
#include <cmath>

namespace ppo::renderer {

/** @brief Minimal 3D point used by the renderer-independent camera contract. */
struct CameraFollowPoint final
{
    float x{};
    float y{};
    float z{};
};

/** @brief Camera position and look-at target updated as one coherent pose. */
struct CameraFollowPose final
{
    CameraFollowPoint position{};
    CameraFollowPoint target{};
};

/**
 * @brief Computes an exponential smoothing factor independent of frame rate.
 * @param deltaSeconds Elapsed render time in seconds.
 * @param responsePerSecond Positive follow response rate.
 * @return Interpolation factor in the [0,1] interval.
 */
inline float CameraFollowBlend(float const deltaSeconds,
                               float const responsePerSecond) noexcept
{
    float const safeDelta = std::clamp(deltaSeconds, 0.F, 0.25F);
    float const safeResponse = std::max(responsePerSecond, 0.F);
    return 1.F - std::exp(-safeResponse * safeDelta);
}

/**
 * @brief Advances a camera towards a desired pose, snapping after discontinuities.
 * @param current Current camera pose.
 * @param desired Desired follow pose.
 * @param deltaSeconds Elapsed render time in seconds.
 * @param responsePerSecond Positive follow response rate.
 * @param forceSnap True after a tracked-entity change.
 * @param snapDistance Maximum continuous target displacement.
 * @return New camera pose.
 */
inline CameraFollowPose AdvanceCameraFollow(
    CameraFollowPose const& current, CameraFollowPose const& desired,
    float const deltaSeconds, float const responsePerSecond,
    bool const forceSnap, float const snapDistance) noexcept
{
    auto const distanceSquared = [](CameraFollowPoint const& first,
                                    CameraFollowPoint const& second) noexcept {
        float const dx = first.x - second.x;
        float const dy = first.y - second.y;
        float const dz = first.z - second.z;
        return dx * dx + dy * dy + dz * dz;
    };
    float const safeSnapDistance = std::max(snapDistance, 0.F);
    if (forceSnap
        || distanceSquared(current.target, desired.target)
            > safeSnapDistance * safeSnapDistance)
    {
        return desired;
    }

    float const blend = CameraFollowBlend(deltaSeconds, responsePerSecond);
    auto const interpolate = [blend](CameraFollowPoint const& first,
                                     CameraFollowPoint const& second) noexcept {
        return CameraFollowPoint{
            first.x + (second.x - first.x) * blend,
            first.y + (second.y - first.y) * blend,
            first.z + (second.z - first.z) * blend};
    };
    return {interpolate(current.position, desired.position),
            interpolate(current.target, desired.target)};
}

} // namespace ppo::renderer
