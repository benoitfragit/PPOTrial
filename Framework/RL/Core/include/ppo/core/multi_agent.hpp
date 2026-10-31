/**
 * @file Framework/RL/Core/include/ppo/core/multi_agent.hpp
 * @brief Interface ou implémentation du composant multi agent.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include <utility>

namespace ppo::core {

// Agent ranges contain nullable pointer-like values, e.g. shared_ptr<IAgent>.
// Traversal order is the caller's order, never re-sorted by the runner.
template<typename Agents, typename Function>
/**
 * @brief Réalise l'opération ForEachAgent; voir les préconditions et effets dans son corps C++.
 */
void ForEachAgent(Agents& agents, Function&& function)
{
    for (auto& agent : agents)
    {
        if (agent)
            function(*agent);
    }
}

// World initialization must finish before any agent reads its first observation.
template<typename Agents, typename Reset>
/**
 * @brief Initialise ou rétablit l'état selon le contexte courant.
 */
void ResetAgents(Agents& agents, Reset&& reset)
{
    ForEachAgent(agents, std::forward<Reset>(reset));
}

// Common multi-agent step sequence. All agents select their action before any
// world object advances. Feedback is delivered only after outcomes are resolved.
// Every agent receives feedback, even when an earlier agent requests a reset.
template<typename Agents, typename Choose, typename Advance, typename Resolve, typename Feedback>
bool StepAgents(Agents& agents,
                Choose&& choose,
                Advance&& advance,
                Resolve&& resolve,
                Feedback&& feedback)
{
    ForEachAgent(agents, std::forward<Choose>(choose));
    std::forward<Advance>(advance)();
    std::forward<Resolve>(resolve)();

    bool resetRequested = false;
    ForEachAgent(agents, [&](auto& agent)
    {
        resetRequested |= static_cast<bool>(feedback(agent));
    });
    return resetRequested;
}

} // namespace ppo::core
