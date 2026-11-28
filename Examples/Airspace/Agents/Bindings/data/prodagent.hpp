/**
 * @file Examples/Airspace/Agents/Bindings/data/prodagent.hpp
 * @brief Production agent: a fixed policy drives a domain model.
 */
#pragma once

#include "ppo/core/agent_contract.hpp"
#include "ppo/core/buffers.hpp"
#include "ppo/core/environment.hpp"
#include "ppo/runtime/iagent.hpp"

#include <memory>
#include <string>

namespace ppo::model {

/**
 * @brief Runs a production policy on a domain model (states in, actions out), without training.
 * @tparam Model Domain model with STATES, ACTIONS, SetNewState and SetNextAction.
 * @details The policy comes from the derived agent (a converted legacy policy or a PPOKit
 *          graph); without one, the agent stays idle, as when its model file is missing.
 */
template<typename Model>
class ProdAgent : public IAgent
{
public:
    using StateVector = typename Model::STATES;
    using ActionVector = typename Model::ACTIONS;

    /** @brief Optional descriptor of the observation and action contract. */
    static constexpr ppo::core::AgentContract Contract() noexcept
    {
        return ppo::core::AgentContractOf<Model>();
    }

    /**
     * @param model Domain model, owned by its entity.
     * @param name Agent name.
     * @param policy Production policy, or null to stay idle.
     */
    ProdAgent(Model* const model, std::string const& name, std::unique_ptr<ppo::core::IPolicy> policy)
        : p_model{model}, m_name{name}, p_policy{std::move(policy)}
    {
    }

    void Init(void) override
    {
        if (p_model != nullptr)
            p_model->SetNewState(m_stateVector);
    }

    void ChooseAction(void) override
    {
        if (p_model == nullptr || p_policy == nullptr)
            return;
        ppo::core::CopyAction(m_actionVector, p_policy->ChooseAction(ppo::core::CopyObservation(m_stateVector)));
        p_model->SetNextAction(m_actionVector);
    }

    /** @brief The observation the next action is chosen on (after Init or SetNewState). */
    StateVector const& State(void) const noexcept { return m_stateVector; }

    bool SetNewState(void) override
    {
        if (p_model != nullptr)
            p_model->SetNewState(m_stateVector);
        return false;
    }

private:
    Model* p_model{nullptr};
    StateVector m_stateVector{};
    ActionVector m_actionVector{};
    std::string m_name{};
    std::unique_ptr<ppo::core::IPolicy> p_policy{nullptr};
};

} // namespace ppo::model
