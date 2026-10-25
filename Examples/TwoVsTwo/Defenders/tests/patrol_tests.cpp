#include "patrol_core.hpp"
#include "patrol_reference_controller.hpp"
#include "ppo/core/team_episode.hpp"

#include <array>
#include <cmath>
#include <iostream>

namespace blue = ppo::examples::twovstwo::defenders;

int main()
{
    static_assert(blue::ObservationLayout::Size == 124U);
    static_assert(blue::ObservationLayout::ActionSize == 2U);
    blue::Parameters const p{};
    if (!blue::Valid(p))
        return 1;
    float const length = blue::CircuitLength(p);
    // Entire orbit stays within world, starts/ends on the same tangent.
    for (int side : {-1, 1})
        for (std::size_t role = 0; role < 2; ++role)
        {
            auto const first = blue::PatrolReference(p, side, role, 0.F);
            auto const last = blue::PatrolReference(p, side, role, length);
            if (std::abs(first.north - last.north) > 0.01F
                || std::abs(first.east - last.east) > 0.01F
                || std::abs(std::sin(first.heading - last.heading)) > 0.001F)
                return 2;
            for (unsigned i = 0; i < 2000U; ++i)
            {
                auto const at = blue::PatrolReference(p, side, role,
                                   length * static_cast<float>(i) / 2000.F);
                if (std::abs(at.north) >= p.halfNorth
                    || std::abs(at.east) >= p.halfEast)
                    return 3;
            }
        }
    auto state = blue::InitialState(p, 1, 0U, 0.F);
    auto const r0 = blue::PatrolReference(p, 1, 0, 0.F);
    auto const r1 = blue::PatrolReference(p, 1, 1, 0.F);
    if (r0.north <= r1.north || r0.east == r1.east)
        return 4; // Leader ahead, wingman offset inward.
    for (std::size_t i = 0; i < 61; ++i)
    {
        auto const obs = blue::Observe(p, state);
        if (obs[blue::ObservationLayout::Own + i] < 0.F
            || obs[blue::ObservationLayout::Own + i] > 1.F)
            return 5;
    }
    auto const observation = blue::Observe(p, state);
    for (std::size_t i = blue::ObservationLayout::RedOffset;
         i < blue::ObservationLayout::FriendOffset; ++i)
        if (observation[i] != 0.F)
            return 6;
    if (observation[blue::ObservationLayout::FriendOffset + 8] != 1.F
        || observation[blue::ObservationLayout::FriendOffset + 17] != 1.F)
        return 7;
    auto const next = blue::Advance(p, state, blue::Action{0.F, 0.F});
    if (next.step != state.step + 1
        || next.guards[0].north <= state.guards[0].north
        || !std::isfinite(blue::TeamReward(p, state, {0.F, 0.F}, next)))
        return 8;
    // A stopped agent must not score the same progression as the reference.
    auto stopped = next;
    stopped.guards[0].speed = 0.F;
    if (!(blue::TrackingReward(p, stopped, 0U)
        < blue::TrackingReward(p, next, 0U)))
        return 9;
    auto out = state;
    out.guards[0].north = p.halfNorth + 1.F;
    auto const failed = blue::Advance(p, out, {0.F, 0.F});
    if (!failed.failed || blue::TeamReward(p, out, {0.F, 0.F}, failed) != -100.F)
        return 10;
    // Forced timeout has penalty but is not a victory or termination.
    auto timeout = next;
    timeout.step = p.episodeSteps;
    if (timeout.failed ||
        !(blue::TeamReward(p, state, {0.F, 0.F}, timeout)
          < blue::TeamReward(p, state, {0.F, 0.F}, next) - 14.F))
        return 11;
    // Reference controllers use the same normalized SNA actions/physics as
    // the production ECS; both guards must move and turn without PPO weights.
    for (int side : {-1, 1})
    {
        auto patrol = blue::InitialState(p, side, 0U, 0.F);
        float maxError = 0.F;
        float accumulatedTurn = 0.F;
        for (std::size_t i = 0; i < 3000U; ++i)
        {
            std::array<blue::Action, 2> commands{};
            for (std::size_t role = 0; role < commands.size(); ++role)
                commands[role] = blue::PatrolReferenceAction(
                    p, patrol.guards[role], side, role, patrol.arclength);
            auto const previous = patrol;
            patrol = blue::AdvanceBoth(p, patrol, commands);
            if (patrol.failed || patrol.step != i + 1U)
                return 12;
            for (std::size_t role = 0; role < patrol.guards.size(); ++role)
            {
                auto const reference = blue::PatrolReference(
                    p, side, role, patrol.arclength);
                auto const& car = patrol.guards[role];
                maxError = std::max(maxError,
                    std::hypot(car.north - reference.north,
                               car.east - reference.east));
                accumulatedTurn += std::abs(car.turnRate * p.dt);
            }
            if (i == 0U
                && (std::abs(patrol.guards[0].north - previous.guards[0].north)
                    + std::abs(patrol.guards[0].east - previous.guards[0].east)) < 1.F)
                return 13;
        }
        if (maxError > 300.F || accumulatedTurn < blue::kTwoPi)
            return 14;
    }
    return 0;
}
