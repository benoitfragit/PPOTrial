/**
* @file PPO/data/prodagent.hpp
* @brief Declarations for prodagent.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once
#include <array>
#include <memory>
#include <filesystem>

#include "torchproductionpolicy.hpp"
#include "ppo/core/buffers.hpp"
#include "ppo/core/agent_contract.hpp"
#include "iagent.hpp"
#include <torch/torch.h>

namespace ppo::model {

/**
* @brief Generic inference-only agent wrapper.
* @tparam Policy Production policy network type.
* @tparam Model Domain model exposing state and action buffers.
*/
template<typename Policy, typename Model>
/** @brief Déclare le type ProdAgent, utilisé par Interface ou implémentation du composant prodagent. */
class ProdAgent :
    public IAgent
{
public:
    /**
    * @brief Alias to the model-defined state vector type.
    */
    using StateVector = typename Model::STATES;
    /**
    * @brief Alias to the model-defined action vector type.
    */
    using ActionVector = typename Model::ACTIONS;

    // Optional descriptor; no change to the PPO execution path or agent ABI.
/**
 * @brief Réalise l'opération Contract; voir les préconditions et effets dans son corps C++.
 * @details Utilise des fichiers ou chemins.
 */
    static constexpr ppo::core::AgentContract Contract() noexcept
    {
        return ppo::core::AgentContractOf<Model>();
    }

    /**
    * @brief Creates a production agent bound to one model.
    * @param model Controlled domain model.
    * @param name Logical agent name.
    * @param modelpath File path to the serialized policy.
    */
    ProdAgent(Model* const model,
        std::string const& name,
        std::filesystem::path const& modelpath);

    /**
    * @brief Initializes the internal state vector from the model.
    */
    void Init(void) override;

    /**
    * @brief Computes one action and applies it to the model.
    */
    void ChooseAction(void) override;
    /**
    * @brief Returns whether the production agent is ready for inference.
    * @return Always true for production agents.
    */
    bool IsTrained(void) const noexcept;
    /**
    * @brief Refreshes the state vector after one environment step.
    * @return Always false because no training rollout ends here.
    */
    bool SetNewState(void) override;

    /**
    * @brief Ignores real-time commands because production agents are read-only.
    */
    void ProcessCommand(sAgentRealTimeCommand const&) noexcept override { return; };

    /**
    * @brief Returns an empty snapshot for production agents.
    * @return Immutable empty snapshot.
    */
    sAgentStateSnaphot GetSnapshot(void) const noexcept override;

private:
    /**
    * @brief Controlled domain model.
    */
    Model* p_model{ nullptr };
    /**
    * @brief Current state buffer.
    */
    StateVector m_stateVector{};
    /**
    * @brief Current action buffer.
    */
    ActionVector m_actionVector{};
    /**
    * @brief Logical agent name.
    */
    std::string m_name{ "" };
    /**
    * @brief Path to the serialized production model.
    */
    std::filesystem::path m_modelpath{ "" };
    /**
    * @brief Production policy network.
    */
    std::unique_ptr<ppo::core::IPolicy> p_policy{ nullptr };
    /**
    * @brief Unique runtime identifier.
    */
    unsigned long long m_id{ GetNewUniqId() };
};

} /* ppo::model */

#include "prodagent_def.hpp"
