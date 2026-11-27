/**
 * @file PPOKit/include/ppokit/editor/ppo_help.hpp
 * @brief Help texts of the PPO hyper-parameters and training metrics, for the studio.
 * @details One entry per name used in train.json and in the run journal (gamma,
 * approx_kl...): a title, what it does, a typical range and what to look for. The studio
 * shows them as tooltips; they do not depend on ImGui.
 */
#pragma once

#include <string_view>
#include <vector>

namespace ppokit::editor {

/** @brief Help of one hyper-parameter or metric. */
struct HelpEntry final
{
    /** @brief Name in train.json or in the journal, e.g. "approx_kl". */
    char const* name;
    /** @brief Short title, e.g. "Approximate KL". */
    char const* title;
    /** @brief What it is and what it changes. */
    char const* what;
    /** @brief Usual values, or a healthy range for a metric. */
    char const* typical;
    /** @brief Symptoms and what to try. */
    char const* advice;
};

/** @brief Hyper-parameters, in the order of the settings panel. */
std::vector<HelpEntry> const& PpoParameterHelp();

/** @brief Training metrics written by the learners, in the order of the dashboard. */
std::vector<HelpEntry> const& PpoMetricHelp();

/** @brief The four ways the dashboard plots a reward channel. */
std::vector<HelpEntry> const& RewardViewHelp();

/** @brief The statistics of the network probes (Probes tab of the dashboard). */
std::vector<HelpEntry> const& ProbeHelp();

/** @brief Entry called `name` in any of the lists, or null. */
HelpEntry const* FindHelp(std::string_view name);

} // namespace ppokit::editor
