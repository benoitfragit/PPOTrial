/**
 * @file PPOKit/include/ppokit/editor/run_series.hpp
 * @brief Curves of a run journal for the studio dashboard: one point per update.
 * @details Pure data, no ImGui: the dashboard plots these series and the tests check them.
 * Missing values (a NaN statistic, a window without completed episode) stay NaN so that
 * the plot leaves a gap instead of drawing a false zero.
 */
#pragma once

#include "ppokit/runtime/run_log.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ppokit::editor {

/** @brief How a reward channel is plotted (see RewardViewHelp). */
enum class RewardView : std::uint8_t
{
    WeightedStepMean,
    EpisodeMean,
    Cumulative,
    StepMean,
};

/** @brief Journal key of a reward view, e.g. "episode_mean" (also its help name). */
char const* RewardViewName(RewardView view) noexcept;

/** @brief x (update number) and y values of one curve, same length. */
struct Series final
{
    std::string name;
    std::string label;
    std::vector<double> updates;
    std::vector<double> values;
};

/** @brief One curve per reward channel, in the order of the journal header. */
std::vector<Series> RewardSeries(runtime::RunLog const& log, RewardView view);

/**
 * @brief Mean return per completed episode, update by update.
 * @details NaN for an update window in which no episode ended.
 */
Series ReturnSeries(runtime::RunLog const& log);

/** @brief Names of the PPO statistics found in the journal, in first-seen order. */
std::vector<std::string> MetricNames(runtime::RunLog const& log);

/** @brief One PPO statistic, update by update (NaN where an update lacks it). */
Series MetricSeries(runtime::RunLog const& log, std::string const& name);

} // namespace ppokit::editor
