#pragma once

#include "patrol_core.hpp"
#include "baseentity.hpp"
#include "iagent.hpp"
#include <cstddef>

namespace ppo::examples::twovstwo::defenders {

/**
 * @brief Drives one guard with the same ECS dynamics as the MAPPO actors.
 *
 * The episode context determines whether the physical guard follows the
 * racetrack as leader or follows the leader's wedge as wingman.
 */
class PatrolReferenceAgent final : public ppo::model::IAgent
{
public:
    /**
     * @brief Creates one physical reference driver with episode-random roles.
     * @param own Physical blue guard controlled by this agent.
     * @param teammate The other physical blue guard.
     * @param guardIndex Stable physical index: zero or one.
     */
    PatrolReferenceAgent(ppo::model::BaseEntity& own,
                         ppo::model::BaseEntity& teammate,
                         std::size_t guardIndex);

    void Init() override;
    void ChooseAction() override;
    bool SetNewState() override;
    void ProcessCommand(ppo::model::sAgentRealTimeCommand const&) noexcept override {}
    ppo::model::sAgentStateSnaphot GetSnapshot() const noexcept override;

private:
    Parameters ReadParameters();
    /** @brief Returns this physical guard's semantic role for the episode. */
    std::size_t Role() const noexcept;

    entt::registry* registry_{};
    ppo::model::BaseEntity own_;
    ppo::model::BaseEntity teammate_;
    std::size_t guardIndex_ = 0U;
    std::size_t step_ = 0U;
    unsigned long long id_ = ppo::model::GetNewUniqId();
    bool done_ = false;
};

} // namespace ppo::examples::twovstwo::defenders
