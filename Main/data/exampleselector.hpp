#pragma once

#include "examplecatalog.hpp"

#include <optional>
#include <string>

namespace ppo::app {

// A command-line example or a headless check must never create a GUI.
constexpr bool ShouldShowExampleSelector(bool explicitExample, bool headless) noexcept
{
    return !explicitExample && !headless;
}

// Displays the compiled example catalog before creating the simulation.
// Closing the window without choosing an example returns std::nullopt.
std::optional<std::string> SelectExample(ExampleCatalog const& catalog);

} // namespace ppo::app
