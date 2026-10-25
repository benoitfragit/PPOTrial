#pragma once
#include "patrol_policy.hpp"
#include "baseentity.hpp"
#include "iagent.hpp"

#include <cstddef>
#include <memory>

namespace ppo::examples::twovstwo::defenders {

// Runtime inference adapter only: both agents use one shared RLtools actor;
// the environment already sequences both ChooseAction calls before physics.
class BlueProdAgent final : public ppo::model::IAgent
{
public:
    BlueProdAgent(ppo::model::BaseEntity& own, ppo::model::BaseEntity& teammate,
                  std::size_t role, std::shared_ptr<BluePolicy> actor);

    void Init() override;
    void ChooseAction() override;
    bool SetNewState() override;
    void ProcessCommand(ppo::model::sAgentRealTimeCommand const&) noexcept override {}
    ppo::model::sAgentStateSnaphot GetSnapshot() const noexcept override;

private:
    State ReadState(Parameters const& parameters);
    Parameters ReadParameters();

    ppo::model::BaseEntity own_;
    ppo::model::BaseEntity teammate_;
    std::size_t role_{};
    std::shared_ptr<BluePolicy> actor_;
    std::size_t step_ = 0U;
    unsigned long long id_ = ppo::model::GetNewUniqId();
    bool done_ = false;
};
} // namespace ppo::examples::twovstwo::defenders
