/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Training/generalized_critic_observation.hpp
 * @brief Entrées conditionnées du critic centralisé pour chaque participant.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "patrol_core.hpp"

#include <array>
#include <cstddef>

namespace ppo::examples::bluedefense::defenders {

/**
 * @brief Agent-conditioned value input with fixed size for 1..9 blue mobiles.
 *
 * The evaluated agent contributes its attention-encoded local state and role.
 * Mean and max pool the alive team without depending on slot order.
 * Living and initial team fractions distinguish losses from smaller teams.
 * Dead participants have a zero own block and retain the living pooled context.
 */
struct GeneralizedCriticLayout final
{
    static constexpr std::size_t Compact = ObservationLayout::Size;
    static constexpr std::size_t OwnOffset = 0U;
    static constexpr std::size_t MeanOffset = Compact;
    static constexpr std::size_t MaxOffset = 2U * Compact;
    static constexpr std::size_t LivingFractionOffset = 3U * Compact;
    static constexpr std::size_t TeamFractionOffset = LivingFractionOffset + 1U;
    static constexpr std::size_t Size = TeamFractionOffset + 1U;
};

using GeneralizedCriticObservation =
    std::array<float, GeneralizedCriticLayout::Size>;

} // namespace ppo::examples::bluedefense::defenders
