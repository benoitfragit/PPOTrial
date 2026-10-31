/**
 * @file Examples/data/example.hpp
 * @brief Interface ou implémentation du composant example.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include <filesystem>
#include <string_view>

namespace ppo::model {
/** @brief Déclare le type Environment, utilisé par Interface ou implémentation du composant example. */
class Environment;
}

namespace ppo::app {

// A compiled example provides its startup without coupling the host to its
// entity types, agent types or scenario semantics.
/** @brief Déclare le type ExampleDefinition, utilisé par Interface ou implémentation du composant example. */
struct ExampleDefinition
{
    std::string_view id;
    std::string_view windowTitle;
    std::filesystem::path defaultScenario;
    // Registers this example's entity and agent factories and reflected types.
    void (*registerTypes)() = nullptr;
    // Reports whether every declared entity and agent was instantiated.
    // The desktop retains its historical permissive startup behavior.
    bool (*loadScenario)(std::filesystem::path const&, ppo::model::Environment&) = nullptr;
    // Optional description displayed by the desktop example selector.
    std::string_view description{};
};

/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
inline bool IsValid(ExampleDefinition const& example) noexcept
{
    return !example.id.empty()
        && !example.windowTitle.empty()
        && !example.defaultScenario.empty()
        && example.registerTypes != nullptr
        && example.loadScenario != nullptr;
}

} // namespace ppo::app
