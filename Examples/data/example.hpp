#pragma once

#include <filesystem>
#include <string_view>

namespace ppo::model {
class Environment;
}

namespace ppo::app {

// A compiled example provides its startup without coupling the host to its
// entity types, agent types or scenario semantics.
struct ExampleDefinition
{
    std::string_view id;
    std::string_view windowTitle;
    std::filesystem::path defaultScenario;
    void (*registerTypes)() = nullptr;
    // Reports whether every declared entity and agent was instantiated.
    // The desktop retains its historical permissive startup behavior.
    bool (*loadScenario)(std::filesystem::path const&, ppo::model::Environment&) = nullptr;
};

inline bool IsValid(ExampleDefinition const& example) noexcept
{
    return !example.id.empty()
        && !example.windowTitle.empty()
        && !example.defaultScenario.empty()
        && example.registerTypes != nullptr
        && example.loadScenario != nullptr;
}

} // namespace ppo::app
