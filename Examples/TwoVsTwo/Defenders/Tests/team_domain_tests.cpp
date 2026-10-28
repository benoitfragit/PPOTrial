#include "blue_team_roles.hpp"
#include "team_capacity.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

namespace patrol = ppo::examples::twovstwo::defenders;

namespace {

bool Near(float const first, float const second) noexcept
{
    return std::abs(first - second) <= 1.0e-5F;
}

bool ValidAssignmentForEveryTeamSize() noexcept
{
    for (std::size_t blueCount = patrol::TeamCapacity::MinimumBlueMobiles;
         blueCount <= patrol::TeamCapacity::MaximumBlueMobiles; ++blueCount)
    {
        for (std::size_t leader = 0U; leader < blueCount; ++leader)
        {
            auto const roles = patrol::BlueTeamRoles::Create(blueCount, leader);
            if (!roles || !roles->IsValid()
                || roles->ActiveBlueCount() != blueCount
                || roles->LeaderPhysicalIndex() != leader)
            {
                return false;
            }

            std::array<bool,
                       patrol::TeamCapacity::MaximumBlueMobiles - 1U> slots{};
            std::size_t observedLeaders = 0U;
            for (std::size_t physical = 0U; physical < blueCount; ++physical)
            {
                auto const assignment = roles->Assignment(physical);
                if (!assignment)
                    return false;
                if (assignment->role == patrol::BlueRole::Leader)
                {
                    ++observedLeaders;
                    if (assignment->formationSlot
                        != patrol::BlueRoleAssignment::NoFormationSlot)
                    {
                        return false;
                    }
                    continue;
                }

                if (assignment->formationSlot >= blueCount - 1U
                    || slots[assignment->formationSlot])
                {
                    return false;
                }
                slots[assignment->formationSlot] = true;
            }
            if (observedLeaders != 1U || roles->Assignment(blueCount))
                return false;
        }
    }
    return true;
}

bool DistinctFormationGeometryForEveryTeamSize() noexcept
{
    constexpr float distance = 6000.F;
    constexpr float angle = 0.25F;
    for (std::size_t blueCount = 2U;
         blueCount <= patrol::TeamCapacity::MaximumBlueMobiles; ++blueCount)
    {
        std::array<patrol::FormationOffset,
                   patrol::TeamCapacity::MaximumBlueMobiles - 1U> offsets{};
        for (std::size_t slot = 0U; slot < blueCount - 1U; ++slot)
        {
            offsets[slot] = patrol::EvaluateFormationOffset(
                slot, distance, angle);
            for (std::size_t previous = 0U; previous < slot; ++previous)
            {
                if (Near(offsets[slot].longitudinalMeters,
                         offsets[previous].longitudinalMeters)
                    && Near(offsets[slot].inwardMeters,
                            offsets[previous].inwardMeters))
                {
                    return false;
                }
            }
        }
    }

    auto const inside = patrol::EvaluateFormationOffset(0U, distance, angle);
    auto const outside = patrol::EvaluateFormationOffset(1U, distance, angle);
    return inside.longitudinalMeters < 0.F && inside.inwardMeters > 0.F
        && Near(inside.longitudinalMeters, outside.longitudinalMeters)
        && Near(inside.inwardMeters, -outside.inwardMeters);
}

} // namespace

int main()
{
    using patrol::ActiveTeamCounts;
    if (!patrol::Valid(ActiveTeamCounts{1U, 0U})
        || !patrol::Valid(ActiveTeamCounts{9U, 10U})
        || patrol::Valid(ActiveTeamCounts{0U, 0U})
        || patrol::Valid(ActiveTeamCounts{10U, 0U})
        || patrol::Valid(ActiveTeamCounts{1U, 11U}))
    {
        return 1;
    }

    if (patrol::BlueTeamRoles::Create(0U, 0U)
        || patrol::BlueTeamRoles::Create(10U, 0U)
        || patrol::BlueTeamRoles::Create(2U, 2U))
    {
        return 2;
    }

    if (!ValidAssignmentForEveryTeamSize())
        return 3;

    if (!DistinctFormationGeometryForEveryTeamSize())
        return 4;

    std::mt19937 first{12345U};
    std::mt19937 repeated{12345U};
    auto const firstRoles = patrol::BlueTeamRoles::Sample(9U, first);
    auto const repeatedRoles = patrol::BlueTeamRoles::Sample(9U, repeated);
    if (!firstRoles || !repeatedRoles
        || firstRoles->LeaderPhysicalIndex()
            != repeatedRoles->LeaderPhysicalIndex())
    {
        return 5;
    }

    auto const singleton = patrol::BlueTeamRoles::Create(1U, 0U);
    if (!singleton || !singleton->IsValid()
        || singleton->Assignment(0U)->role != patrol::BlueRole::Leader)
    {
        return 6;
    }
    return 0;
}
