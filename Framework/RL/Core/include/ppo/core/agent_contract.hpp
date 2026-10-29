#pragma once

#include <cstddef>
#include <string_view>
#include <tuple>

namespace ppo::core {

// Read-only identity of an agent's neural input/output contract. The IDs
// describe meaning and ordering, not merely vector width. This type is
// independent of neural backends, simulation entities and serialization.
// Identifiers must have static lifetime (normally string literals):
// the descriptor does not own string_view storage.
struct AgentContract
{
    std::string_view profile;
    std::string_view observationLayout;
    std::size_t observationSize;
    std::string_view actionLayout;
    std::size_t actionSize;
};

constexpr bool IsValid(AgentContract const& contract) noexcept
{
    return !contract.profile.empty()
        && !contract.observationLayout.empty()
        && contract.observationSize != 0U
        && !contract.actionLayout.empty()
        && contract.actionSize != 0U;
}

// Exact compatibility is intentional: equal dimensions alone do not make
// feature normalizations, ordering or action interpretations compatible.
constexpr bool SameContract(AgentContract const& lhs,
                            AgentContract const& rhs) noexcept
{
    return IsValid(lhs) && IsValid(rhs)
        && lhs.profile == rhs.profile
        && lhs.observationLayout == rhs.observationLayout
        && lhs.observationSize == rhs.observationSize
        && lhs.actionLayout == rhs.actionLayout
        && lhs.actionSize == rhs.actionSize;
}

// The widths come from the domain model's existing fixed-size vectors.
// The generic core does not know any specific agent names or dimensions.
template<typename ObservationVector, typename ActionVector>
constexpr AgentContract MakeAgentContract(
    std::string_view profile,
    std::string_view observationLayout,
    std::string_view actionLayout) noexcept
{
    return {
        profile,
        observationLayout,
        std::tuple_size<ObservationVector>::value,
        actionLayout,
        std::tuple_size<ActionVector>::value
    };
}

// Shared validation for both training and inference bindings. Future domain
// adapters describe themselves without adding cases to the core.
template<typename Model>
constexpr AgentContract AgentContractOf() noexcept
{
    constexpr AgentContract contract = Model::kAgentContract;
    static_assert(IsValid(contract), "Agent contract must have non-empty IDs and dimensions");
    static_assert(contract.observationSize == std::tuple_size<typename Model::STATES>::value,
                  "Agent observation dimension does not match its vector");
    static_assert(contract.actionSize == std::tuple_size<typename Model::ACTIONS>::value,
                  "Agent action dimension does not match its vector");
    return contract;
}

} // namespace ppo::core
