#pragma once

#include <utility>

namespace ppo::core {

// Agent ranges contain nullable pointer-like values, e.g. shared_ptr<IAgent>.
// Traversal order is the caller's order, never re-sorted by the runner.
template<typename Agents, typename Function>
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
