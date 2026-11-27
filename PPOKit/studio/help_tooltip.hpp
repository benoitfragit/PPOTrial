/**
 * @file PPOKit/studio/help_tooltip.hpp
 * @brief The help tooltip of the studio views: title, what it does, typical values, advice.
 */
#pragma once

#include "ppokit/editor/ppo_help.hpp"

#include <string>

namespace ppokit::studio {

/**
 * @brief Tooltip of the last item, when hovered.
 * @param footer Extra line under the help (a default and a range), empty for none.
 */
void HelpTooltip(editor::HelpEntry const& help, std::string const& footer = {});

/** @brief A "(?)" marker after the last item, with the help of `name` when it has one. */
void HelpMarker(std::string const& name);

} // namespace ppokit::studio
