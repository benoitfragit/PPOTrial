#pragma once

#include "patrol_core.hpp"

#include <array>
#include <cstddef>

namespace ppo::examples::twovstwo::defenders {

/**
 * @brief Agent-conditioned value input with fixed size for 1..9 blue mobiles.
 *
 * The evaluated agent contributes its attention-encoded local state and role.
 * Mean and max pool the alive team without depending on slot order.
 * Living and initial team fractions distinguish losses from smaller teams.
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

} // namespace ppo::examples::twovstwo::defenders
