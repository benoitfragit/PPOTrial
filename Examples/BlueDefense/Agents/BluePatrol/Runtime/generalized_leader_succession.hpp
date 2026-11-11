/** @file
 * @brief Deterministic leader relief without changing physical episode indices.
 */
#pragma once

#include "generalized_patrol_waypoint_sequencer.hpp"
#include "generalized_reward_shaping.hpp"

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Relieves a lost combat leader with the first surviving physical Blue.
 *
 * Episode membership and physical trajectory indices stay fixed. Rebuilding
 * the assignment keeps unique slots, including masked destroyed participants.
 * Call after capturing the old roles for the resolved step's rewards, before
 * encoding the next observation. Pvs0 keeps its leader-loss failure contract.
 * @return True only when a new leader was assigned and its progress reset.
 */
inline bool RelieveGeneralizedLeader(
    GeneralizedRuntimeState const& state, BlueTeamRoles& roles,
    GeneralizedPatrolWaypointSequencer& sequencer,
    GeneralizedPatrolLapProgress& lap) noexcept
{
    if (!Valid(state) || !roles.IsValid()
        || roles.ActiveBlueCount() != state.counts.blueMobiles
        || state.counts.redMobiles == 0U
        || state.blues[roles.LeaderPhysicalIndex()].available)
    {
        return false;
    }
    for (std::size_t index = 0U; index < state.counts.blueMobiles; ++index)
    {
        if (!state.blues[index].available)
            continue;
        roles = *BlueTeamRoles::Create(state.counts.blueMobiles, index);
        sequencer.Reset();
        lap.Reset();
        return true;
    }
    return false;
}

} // namespace ppo::examples::bluedefense::defenders
