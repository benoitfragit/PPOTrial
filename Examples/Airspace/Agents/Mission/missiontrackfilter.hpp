/**
 * @file Examples/Airspace/Agents/Mission/missiontrackfilter.hpp
 * @brief Interface ou implémentation du composant missiontrackfilter.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "components.hpp"

namespace rl::detail {

/**
 * @brief Tests whether one entity may consume a ranked RLMission track slot.
 * @param ownState State of the controlled mission platform.
 * @param otherState State of the candidate entity.
 * @param ownEntity Identifier of the controlled mission platform.
 * @return True only for a living opposing mobile or objective.
 *
 * Restricted zones remain observable through the dedicated pseudo-LiDAR and
 * SAM threat assessment. They must not consume one of the ten ranked slots.
 */
inline bool IsMissionTrackCandidate(
    ppo::model::sEntityState const& ownState,
    ppo::model::sEntityState const& otherState,
    entt::entity ownEntity) noexcept
{
    return !otherState.m_destroyed
        && otherState.m_entityId != ownEntity
        && otherState.m_side != ownState.m_side
        && (otherState.m_type == ppo::model::eEntityType::Mobile
            || otherState.m_type == ppo::model::eEntityType::Objective);
}

} // namespace rl::detail
