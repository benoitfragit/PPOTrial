/**
 * @file Framework/RL/Core/tests/core_tests.cpp
 * @brief Tests et assertions déterministes pour core tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "ppo/core/agent_contract.hpp"
#include "ppo/core/buffers.hpp"
#include "ppo/core/multi_agent.hpp"

#include <array>
#include <stdexcept>
#include <memory>
#include <vector>

namespace {
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 * @details Peut signaler un état invalide par exception.
 */
void Check(bool condition) {
    if (!condition) throw std::runtime_error("core regression test failed");
}

} // namespace


void TestMultiAgentPhases() {
/** @brief Déclare le type Agent, utilisé par Tests et assertions déterministes pour core tests. */
    struct Agent { int id; };
    auto a = std::make_shared<Agent>(Agent{1});
    auto b = std::make_shared<Agent>(Agent{2});
    std::vector<std::shared_ptr<Agent>> agents{a, nullptr, b};
    std::vector<int> events{};

    ppo::core::ResetAgents(agents, [&](Agent& agent) { events.push_back(10 + agent.id); });
    Check(events == std::vector<int>({11, 12}));
    events.clear();

    bool const reset = ppo::core::StepAgents(
        agents,
        [&](Agent& agent) { events.push_back(10 + agent.id); },
        [&]() { events.push_back(20); },
        [&]() { events.push_back(30); },
        [&](Agent& agent) {
            events.push_back(40 + agent.id);
            return agent.id == 1;
        });
    Check(reset);
    Check(events == std::vector<int>({11, 12, 20, 30, 41, 42}));

    std::vector<std::shared_ptr<Agent>> empty{};
    events.clear();
    bool const noReset = ppo::core::StepAgents(
        empty,
        [&](Agent&) { events.push_back(-1); },
        [&]() { events.push_back(20); },
        [&]() { events.push_back(30); },
        [&](Agent&) { events.push_back(-2); return true; });
    Check(!noReset);
    Check(events == std::vector<int>({20, 30}));
}

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
void TestAgentContracts()
{
    using ppo::core::AgentContract;
    using ppo::core::MakeAgentContract;
    using ppo::core::IsValid;
    using ppo::core::SameContract;

    // An unrelated new agent type can declare its own contract without
    // modifying the generic core or existing agent implementations.
    constexpr AgentContract example = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "Navigation2D", "navigation2d.observation.v1", "navigation2d.action.v1");
    constexpr AgentContract same = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "Navigation2D", "navigation2d.observation.v1", "navigation2d.action.v1");
    constexpr AgentContract reordered = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "Navigation2D", "navigation2d.observation.v2", "navigation2d.action.v1");
    constexpr AgentContract differentAction = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "Navigation2D", "navigation2d.observation.v1", "navigation2d.action.v2");
    constexpr AgentContract differentSize = MakeAgentContract<
        std::array<float, 5>, std::array<float, 2>>(
            "Navigation2D", "navigation2d.observation.v1", "navigation2d.action.v1");
    constexpr AgentContract differentProfile = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "DifferentAgent", "navigation2d.observation.v1", "navigation2d.action.v1");
    constexpr AgentContract invalid = MakeAgentContract<
        std::array<float, 4>, std::array<float, 2>>(
            "", "navigation2d.observation.v1", "navigation2d.action.v1");
    constexpr AgentContract noObservationLayout = { "Navigation2D", "", 4U,
                                                     "navigation2d.action.v1", 2U };
    constexpr AgentContract noActionLayout = { "Navigation2D", "navigation2d.observation.v1",
                                                4U, "", 2U };
    constexpr AgentContract zeroObservation = { "Navigation2D", "navigation2d.observation.v1",
                                                 0U, "navigation2d.action.v1", 2U };
    constexpr AgentContract zeroAction = { "Navigation2D", "navigation2d.observation.v1",
                                            4U, "navigation2d.action.v1", 0U };



    static_assert(IsValid(example), "Example contract must be valid");
    static_assert(SameContract(example, same), "Identical layouts must match");
    static_assert(!SameContract(example, reordered), "Ordering is part of compatibility");
    static_assert(!SameContract(example, differentAction), "Action semantics must match");
    static_assert(!SameContract(example, differentSize), "Dimensions must match");
    static_assert(!SameContract(example, differentProfile), "Profile identity must match");
    static_assert(!IsValid(invalid), "Unnamed agent contracts are invalid");
    static_assert(!IsValid(noObservationLayout), "Missing observation ID is invalid");
    static_assert(!IsValid(noActionLayout), "Missing action ID is invalid");
    static_assert(!IsValid(zeroObservation), "Zero-length observation is invalid");
    static_assert(!IsValid(zeroAction), "Zero-length action is invalid");
    static_assert(!SameContract(invalid, invalid), "Invalid contracts cannot match");
    Check(SameContract(example, same));
    Check(!SameContract(example, reordered));
}

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
int main() {
    // Existing fixed-size domain buffers can be bridged without exposing
    // their lifetime or changing their numerical ordering.
    std::array<float, 3> domainObservation{0.25F, -1.F, 2.F};
    Check(ppo::core::CopyObservation(domainObservation) ==
          ppo::core::Observation({0.25F, -1.F, 2.F}));
    std::array<float, 2> domainAction{};
    ppo::core::CopyAction(domainAction, ppo::core::Action({0.5F, -0.75F}));
    Check(domainAction == std::array<float, 2>{0.5F, -0.75F});
    bool copyRejected = false;
    try { ppo::core::CopyAction(domainAction, ppo::core::Action{1.F}); }
    catch (std::invalid_argument const&) { copyRejected = true; }
    Check(copyRejected);
    Check(domainAction == std::array<float, 2>{0.5F, -0.75F});
    TestAgentContracts();
    TestMultiAgentPhases();
}
