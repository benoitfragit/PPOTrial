/**
 * @file PPOKit/tests/run_series_tests.cpp
 * @brief Dashboard curves from a journal (gaps for missing values) and the help texts.
 */
#include "ppokit/editor/ppo_help.hpp"
#include "ppokit/editor/run_series.hpp"

#include <cmath>
#include <iostream>
#include <string>

namespace {

using namespace ppokit;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

runtime::RunLog MakeLog()
{
    runtime::RunLog log;
    log.header.task = "corridor";
    log.header.channels = {{"progress", "Progress", 1.0}, {"time", "Time penalty", 0.1}};
    for (std::uint64_t update = 1; update <= 3; ++update)
    {
        runtime::UpdateRecord record;
        record.update = update;
        record.rewards.steps = 10;
        record.rewards.episodes = update == 2 ? 0 : 2; // no episode ended during update 2
        record.rewards.episodeReturnMean = static_cast<double>(update);
        record.rewards.channels = {{1.0, 1.0, 5.0, 10.0 * static_cast<double>(update)},
            {-1.0, -0.1, -1.0, -static_cast<double>(update)}};
        record.ppo = {{"policy_loss", -0.1 * static_cast<double>(update)}};
        if (update == 3)
        {
            record.ppo.emplace_back("approx_kl", 0.01);
        }
        log.updates.push_back(record);
    }
    return log;
}

void Series()
{
    runtime::RunLog const log = MakeLog();
    auto const cumulative = editor::RewardSeries(log, editor::RewardView::Cumulative);
    Check(cumulative.size() == 2 && cumulative[0].label == "Progress" && cumulative[0].values[2] == 30.0
            && cumulative[0].updates[2] == 3.0,
        "series: one cumulative curve per channel");
    auto const episodes = editor::RewardSeries(log, editor::RewardView::EpisodeMean);
    Check(episodes[1].values[0] == -1.0 && std::isnan(episodes[1].values[1]),
        "series: no completed episode leaves a gap, not a zero");
    Check(editor::RewardSeries(log, editor::RewardView::StepMean)[1].values[0] == -1.0
            && editor::RewardSeries(log, editor::RewardView::WeightedStepMean)[1].values[0] == -0.1,
        "series: raw and weighted step means");
    auto const returns = editor::ReturnSeries(log);
    Check(returns.values[0] == 1.0 && std::isnan(returns.values[1]), "series: episode return with gaps");
    auto const names = editor::MetricNames(log);
    Check(names.size() == 2 && names[0] == "policy_loss" && names[1] == "approx_kl", "series: metrics in seen order");
    auto const kl = editor::MetricSeries(log, "approx_kl");
    Check(std::isnan(kl.values[0]) && kl.values[2] == 0.01 && kl.label == "Approximate KL",
        "series: a metric missing in early updates is a gap; its label comes from the help");
}

void Help()
{
    bool documented = true;
    for (auto const* list :
        {&editor::PpoParameterHelp(), &editor::PpoMetricHelp(), &editor::RewardViewHelp(), &editor::ProbeHelp()})
    {
        for (editor::HelpEntry const& entry : *list)
        {
            documented = documented && std::string(entry.what).size() > 30 && std::string(entry.typical).size() > 5
                && std::string(entry.advice).size() > 20;
        }
    }
    Check(documented, "help: every entry says what, typical values and advice");
    // Every name the journal writes has a help entry.
    runtime::TeamUpdateStatistics const team;
    ppokit::core::PpoSettings const settings;
    bool covered = true;
    for (auto const& [name, value] : runtime::PpoMetrics(team))
    {
        covered = covered && editor::FindHelp(name) != nullptr;
    }
    for (auto const& [name, value] : runtime::PpoSettingsValues(settings))
    {
        covered = covered && editor::FindHelp(name) != nullptr;
    }
    for (editor::RewardView view : {editor::RewardView::WeightedStepMean, editor::RewardView::EpisodeMean,
             editor::RewardView::Cumulative, editor::RewardView::StepMean})
    {
        covered = covered && editor::FindHelp(editor::RewardViewName(view)) != nullptr;
    }
    Check(covered, "help: every metric, setting and reward view written by PPOKit is documented");
}

} // namespace

int main()
{
    Series();
    Help();
    if (failures != 0)
    {
        std::cerr << failures << " run series check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit run series tests passed\n";
    return 0;
}
