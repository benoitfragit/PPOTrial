/**
 * @file tactical_view_layout_tests.cpp
 * @brief Regressions for tactical layouts, role badges, and camera following.
 */
#include "camera_follow.hpp"
#include "tactical_role.hpp"
#include "tactical_view_layout.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace {

/**
 * @brief Requires one exact grid shape.
 * @param count Requested blue-camera count.
 * @param columns Expected columns.
 * @param rows Expected rows.
 */
void RequireShape(std::size_t const count, std::size_t const columns,
                  std::size_t const rows)
{
    auto const grid = ppo::renderer::MakeTacticalViewGrid(
        count, 1200, 800);
    if (!grid.Valid() || grid.viewCount != count
        || grid.columns != columns || grid.rows != rows
        || grid.cellWidth < 32 || grid.cellHeight < 32)
    {
        throw std::runtime_error("Invalid tactical camera grid shape");
    }
}

/** @brief Requires two floating-point values to remain approximately equal. */
void RequireNear(float const actual, float const expected,
                 float const tolerance, char const* const message)
{
    if (std::fabs(actual - expected) > tolerance)
        throw std::runtime_error(message);
}

/** @brief Verifies canonical leader and teammate markers without allocations. */
void CheckRoleMarkers()
{
    using ppo::renderer::ResolveTacticalRole;
    using ppo::renderer::TacticalRole;
    if (ResolveTacticalRole("blue_3", 4U, 2U) != TacticalRole::Leader
        || ResolveTacticalRole("blue_1", 4U, 2U) != TacticalRole::Teammate
        || ResolveTacticalRole("red_1", 4U, 2U) != TacticalRole::None
        || ResolveTacticalRole("blue_5", 4U, 2U) != TacticalRole::None
        || ResolveTacticalRole("blue_10", 9U, 2U) != TacticalRole::None
        || ResolveTacticalRole("blue_1", 0U, 0U) != TacticalRole::None)
    {
        throw std::runtime_error("Invalid tactical role marker mapping");
    }
}

/** @brief Verifies snapping and frame-rate-independent camera smoothing. */
void CheckCameraFollow()
{
    using ppo::renderer::AdvanceCameraFollow;
    using ppo::renderer::CameraFollowPose;
    CameraFollowPose const start{{0.F, 10.F, 10.F}, {0.F, 0.F, 0.F}};
    CameraFollowPose const desired{{10.F, 10.F, 10.F}, {10.F, 0.F, 0.F}};

    auto const snapped = AdvanceCameraFollow(
        start, desired, 1.F / 60.F, 8.F, true, 20.F);
    RequireNear(snapped.target.x, 10.F, 1.0e-5F,
                "camera did not snap after target change");

    auto const teleport = AdvanceCameraFollow(
        start, desired, 1.F / 60.F, 8.F, false, 5.F);
    RequireNear(teleport.position.x, 10.F, 1.0e-5F,
                "camera did not snap after target teleport");

    auto const oneFrame = AdvanceCameraFollow(
        start, desired, 1.F / 60.F, 8.F, false, 20.F);
    auto halfFrame = AdvanceCameraFollow(
        start, desired, 1.F / 120.F, 8.F, false, 20.F);
    halfFrame = AdvanceCameraFollow(
        halfFrame, desired, 1.F / 120.F, 8.F, false, 20.F);
    RequireNear(oneFrame.position.x, halfFrame.position.x, 1.0e-4F,
                "camera smoothing depends on frame rate");
    if (!(oneFrame.position.x > start.position.x
          && oneFrame.position.x < desired.position.x))
    {
        throw std::runtime_error("continuous camera follow did not interpolate");
    }
}

} // namespace

/** @brief Verifies balanced layouts and invalid camera counts. */
int main()
{
    RequireShape(1U, 1U, 1U);
    RequireShape(2U, 2U, 1U);
    RequireShape(3U, 2U, 2U);
    RequireShape(4U, 2U, 2U);
    RequireShape(5U, 3U, 2U);
    RequireShape(6U, 3U, 2U);
    RequireShape(7U, 3U, 3U);
    RequireShape(8U, 3U, 3U);
    RequireShape(9U, 3U, 3U);
    if (ppo::renderer::MakeTacticalViewGrid(0U, 1200, 800).Valid()
        || ppo::renderer::MakeTacticalViewGrid(10U, 1200, 800).Valid())
    {
        throw std::runtime_error("Invalid tactical camera count accepted");
    }
    CheckRoleMarkers();
    CheckCameraFollow();
    return 0;
}
