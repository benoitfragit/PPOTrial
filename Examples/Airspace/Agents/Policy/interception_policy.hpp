/**
 * @file Examples/Airspace/Agents/Policy/interception_policy.hpp
 * @brief The red RLSNA production policy (two-layer MLP), computed without LibTorch.
 */
#pragma once

#include "legacy_policy_file.hpp"

#include "ppo/core/environment.hpp"

#include <filesystem>
#include <vector>

namespace rl {

/**
 * @brief Deterministic actor of the RLSNA network: mu_head(relu(fc2(relu(fc1(state))))).
 * @details Same computation as the LibTorch ActorCriticNet::forward_mu it replaces; the sizes
 *          come from the agent (observation and action vectors).
 */
class InterceptionPolicy final : public ppo::core::IPolicy
{
public:
    /** @brief Hidden width of the legacy network. */
    static constexpr std::size_t Hidden = 64;

    /**
     * @brief Loads an "interception_mlp" legacy policy file.
     * @param path Legacy policy file.
     * @param observationSize Observation width of the agent.
     * @param actionSize Action width of the agent.
     * @throw std::runtime_error when it cannot be read or does not fit these sizes.
     */
    InterceptionPolicy(std::filesystem::path const& path, std::size_t observationSize, std::size_t actionSize);

    ppo::core::Action ChooseAction(ppo::core::Observation const& observation) override;

private:
    LegacyDense layer1_;
    LegacyDense layer2_;
    LegacyDense mean_;
    std::vector<float> hidden1_;
    std::vector<float> hidden2_;
};

} // namespace rl
