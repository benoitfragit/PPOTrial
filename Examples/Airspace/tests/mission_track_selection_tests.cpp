#include "missionlayout.hpp"
#include "missiontrackfilter.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>

namespace {

using ppo::model::eEntitySide;
using ppo::model::eEntityType;
using ppo::model::sEntityState;

/**
 * @brief Creates one candidate state for the focused selection contract test.
 * @param id Stable synthetic entity identifier.
 * @param type Runtime entity category.
 * @return Living friendly candidate state.
 */
sEntityState Candidate(unsigned int id, eEntityType type) noexcept
{
    sEntityState state{};
    state.m_entityId = static_cast<entt::entity>(id);
    state.m_side = eEntitySide::Friend;
    state.m_type = type;
    return state;
}

} // namespace

int main()
{
    static_assert(rl::MissionObservationLayout::kMaxTracks == 10U,
                  "RLMission must keep exactly ten ranked tracks");
    static_assert(rl::MissionObservationLayout::kTrackFeatureCount == 9U,
                  "The persisted RLMission track layout must not change");
    static_assert(rl::MissionObservationLayout::kPseudoLidarBinCount == 61U,
                  "The persisted RLMission pseudo-LiDAR must not change");
    static_assert(rl::MissionObservationLayout::kStateDimension == 152U,
                  "RLMission checkpoints require 152 observation values");
    static_assert(eEntityType::SAM == eEntityType::RestrictedZone,
                  "The legacy SAM alias must use the RestrictedZone filter");

    sEntityState own{};
    own.m_entityId = static_cast<entt::entity>(100U);
    own.m_side = eEntitySide::Hostile;
    own.m_type = eEntityType::Mobile;

    // Zones deliberately come first: if accepted, they would consume the
    // fixed capacity before one or more valid mobiles/objectives are seen.
    std::array<sEntityState, 13U> const candidates{
        Candidate(1U, eEntityType::RestrictedZone),
        Candidate(2U, eEntityType::SAM),
        Candidate(3U, eEntityType::RestrictedZone),
        Candidate(10U, eEntityType::Mobile),
        Candidate(11U, eEntityType::Mobile),
        Candidate(12U, eEntityType::Mobile),
        Candidate(13U, eEntityType::Mobile),
        Candidate(14U, eEntityType::Mobile),
        Candidate(15U, eEntityType::Mobile),
        Candidate(16U, eEntityType::Mobile),
        Candidate(17U, eEntityType::Mobile),
        Candidate(18U, eEntityType::Mobile),
        Candidate(19U, eEntityType::Objective),
    };

    std::array<entt::entity, rl::MissionObservationLayout::kMaxTracks>
        selected{};
    std::size_t selectedCount = 0U;
    for (auto const& candidate : candidates)
    {
        if (!rl::detail::IsMissionTrackCandidate(
                own, candidate, own.m_entityId))
        {
            continue;
        }

        if (selectedCount >= selected.size())
            break;
        selected[selectedCount++] = candidate.m_entityId;
    }

    if (selectedCount != selected.size())
        throw std::runtime_error("RLMission did not select exactly ten tracks");

    for (std::size_t index = 0U; index < selected.size(); ++index)
    {
        entt::entity const expected =
            static_cast<entt::entity>(10U + static_cast<unsigned int>(index));
        if (selected[index] != expected)
            throw std::runtime_error("A SAM displaced a mobile/objective track");
    }

    return 0;
}
