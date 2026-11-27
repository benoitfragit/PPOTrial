/** @file launch_settings.hpp
 * @brief Per-launch choices for the generic runner, independent of the persisted project defaults.
 */
#pragma once

#include "ppokit/schema/policy_reference.hpp"
#include "ppokit/schema/project.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ppokit::studio {
struct LaunchSettings final
{
    enum class Start { Project, Resume, Fresh };
    enum class Budget { Project, Additional, Absolute };
    Start start = Start::Project;
    Budget budget = Budget::Project;
    std::uint64_t updates = 1;
    struct PolicyChoice
    {
        std::string role;
        std::array<char, 1024> customFile{};
    };
    PolicyChoice evaluation, inference;

    /** @brief Apply only explicit choices, preserving manifest arguments unrelated to this launch. */
    std::vector<std::string> Arguments(schema::ProjectCommand command, std::vector<std::string> args) const;
    /**
     * @brief Draw the launch-settings button and its scrollable panel.
     * @param project Project supplying command defaults and checkpoint paths.
     * @param references Named policy references available to the project.
     */
    void Draw(schema::Project const& project, std::vector<schema::PolicyReference> const& references);
};
} // namespace ppokit::studio
