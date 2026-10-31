/**
 * @file Main/data/exampleselector.hpp
 * @brief Interface ou implémentation du composant exampleselector.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

#include "examplecatalog.hpp"

#include <optional>
#include <string>

namespace ppo::app {

// A command-line example or a headless check must never create a GUI.
/**
 * @brief Contrôle les invariants, masques, bornes ou autres conditions spécifiés par cette opération.
 */
constexpr bool ShouldShowExampleSelector(bool explicitExample, bool headless) noexcept
{
    return !explicitExample && !headless;
}

// Displays the compiled example catalog before creating the simulation.
// Closing the window without choosing an example returns std::nullopt.
/**
 * @brief Réalise l'opération SelectExample; voir les préconditions et effets dans son corps C++.
 */
std::optional<std::string> SelectExample(ExampleCatalog const& catalog);

} // namespace ppo::app
