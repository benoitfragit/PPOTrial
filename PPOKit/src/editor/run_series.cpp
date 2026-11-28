/**
 * @file PPOKit/src/editor/run_series.cpp
 * @brief Extraction of dashboard curves from a run journal.
 */
#include "ppokit/editor/run_series.hpp"

#include "ppokit/editor/ppo_help.hpp"

#include <algorithm>
#include <limits>

namespace ppokit::editor {
namespace {

constexpr double Missing = std::numeric_limits<double>::quiet_NaN();

double ChannelValue(runtime::ChannelSummary const& summary, RewardView view) noexcept
{
    switch (view)
    {
    case RewardView::WeightedStepMean:
        return summary.weightedStepMean;
    case RewardView::EpisodeMean:
        return summary.episodeMean;
    case RewardView::Cumulative:
        return summary.cumulative;
    case RewardView::StepMean:
        break;
    }
    return summary.stepMean;
}

/** @brief Per-episode values are undefined in a window without completed episode. */
bool Defined(runtime::UpdateRecord const& record, RewardView view) noexcept
{
    return view != RewardView::EpisodeMean || record.rewards.episodes != 0;
}

} // namespace

char const* RewardViewName(RewardView view) noexcept
{
    switch (view)
    {
    case RewardView::WeightedStepMean:
        return "weighted_step_mean";
    case RewardView::EpisodeMean:
        return "episode_mean";
    case RewardView::Cumulative:
        return "cumulative";
    case RewardView::StepMean:
        break;
    }
    return "step_mean";
}

std::vector<Series> RewardSeries(runtime::RunLog const& log, RewardView view)
{
    std::vector<Series> series;
    for (std::size_t channel = 0; channel < log.header.channels.size(); ++channel)
    {
        Series curve;
        curve.name = log.header.channels[channel].name;
        curve.label = log.header.channels[channel].label;
        for (runtime::UpdateRecord const& record : log.updates)
        {
            curve.updates.push_back(static_cast<double>(record.update));
            curve.values.push_back(Defined(record, view) && channel < record.rewards.channels.size()
                    ? ChannelValue(record.rewards.channels[channel], view)
                    : Missing);
        }
        series.push_back(std::move(curve));
    }
    return series;
}

Series ReturnSeries(runtime::RunLog const& log)
{
    Series curve;
    curve.name = "episode_return_mean";
    curve.label = "Episode return";
    for (runtime::UpdateRecord const& record : log.updates)
    {
        curve.updates.push_back(static_cast<double>(record.update));
        curve.values.push_back(record.rewards.episodes != 0 ? record.rewards.episodeReturnMean : Missing);
    }
    return curve;
}

std::vector<std::string> MetricNames(runtime::RunLog const& log)
{
    std::vector<std::string> names;
    for (runtime::UpdateRecord const& record : log.updates)
    {
        for (auto const& [name, value] : record.ppo)
        {
            if (std::find(names.begin(), names.end(), name) == names.end())
            {
                names.push_back(name);
            }
        }
    }
    return names;
}

Series MetricSeries(runtime::RunLog const& log, std::string const& name)
{
    Series curve;
    curve.name = name;
    HelpEntry const* const help = FindHelp(name);
    curve.label = help != nullptr ? help->title : name;
    for (runtime::UpdateRecord const& record : log.updates)
    {
        auto const value = std::find_if(record.ppo.begin(), record.ppo.end(),
            [&name](auto const& entry) { return entry.first == name; });
        curve.updates.push_back(static_cast<double>(record.update));
        curve.values.push_back(value != record.ppo.end() ? value->second : Missing);
    }
    return curve;
}

} // namespace ppokit::editor
