#pragma once

#include "patrol_core.hpp"
#include "baseentity.hpp"
#include "iagent.hpp"
#include <cstddef>

namespace ppo::examples::twovstwo::defenders {

// Drives a guard along the reference path using the same ECS vehicle dynamics
// as a PPO actor. This controller is NOT a trained or trainable PPO policy.
class PatrolReferenceAgent final : public ppo::model::IAgent
{
public:
    PatrolReferenceAgent(ppo::model::BaseEntity& own, std::size_t role);

    void Init() override;
    void ChooseAction() override;
    bool SetNewState() override;
    void ProcessCommand(ppo::model::sAgentRealTimeCommand const&) noexcept override {}
    ppo::model::sAgentStateSnaphot GetSnapshot() const noexcept override;

private:
    Parameters ReadParameters();

    ppo::model::BaseEntity own_;
    std::size_t role_ = 0U;
    std::size_t step_ = 0U;
    unsigned long long id_ = ppo::model::GetNewUniqId();
    bool done_ = false;
};

} // namespace ppo::examples::twovstwo::defenders
