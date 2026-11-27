/**
 * @file PPOKit/tests/run_health_tests.cpp
 * @brief Health cards of a run journal: verdicts, trends, missing data, durations.
 */
#include "ppokit/editor/run_health.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
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

/** @brief A journal of `updates` updates where `metric(update)` gives each PPO value. */
template<typename Metric>
runtime::RunLog Journal(std::size_t updates, Metric metric, double targetKl = 0.02)
{
    runtime::RunLog log;
    log.header.task = "health";
    log.header.settings = {{"target_kl", targetKl}};
    for (std::size_t update = 1; update <= updates; ++update)
    {
        runtime::UpdateRecord record;
        record.update = update;
        record.rewards.episodes = 3;
        metric(update, record);
        log.updates.push_back(record);
    }
    return log;
}

editor::HealthCard const* Card(std::vector<editor::HealthCard> const& cards, std::string const& title)
{
    for (editor::HealthCard const& card : cards)
    {
        if (card.title == title)
        {
            return &card;
        }
    }
    return nullptr;
}

void HealthyRun()
{
    auto const log = Journal(30, [](std::size_t update, runtime::UpdateRecord& record) {
        record.rewards.episodeReturnMean = static_cast<double>(update);
        record.ppo = {{"explained_variance", 0.8}, {"approx_kl", 0.01}, {"clip_fraction", 0.1},
            {"actor_kl_skips", 0.0}};
    });
    auto const cards = editor::RunHealth(log);
    Check(cards.size() == 5, "a run with every statistic has five cards");
    editor::HealthCard const* const reward = Card(cards, "Episode return");
    Check(reward != nullptr && reward->trend == editor::Trend::Up && reward->health == editor::Health::Good,
        "a rising return is good");
    Check(reward != nullptr && reward->value == "28", "the return shown is the mean of the last 5 updates");
    Check(editor::OverallHealth(cards) == editor::Health::Good, "a healthy run is good overall");
    editor::HealthCard const* const kl = Card(cards, "Policy change");
    Check(kl != nullptr && kl->message.find("target KL (0.02)") != std::string::npos, "the KL verdict names the target");
}

void ProblemRun()
{
    auto const log = Journal(30, [](std::size_t update, runtime::UpdateRecord& record) {
        record.rewards.episodeReturnMean = 100.0 - static_cast<double>(update);
        record.ppo = {{"explained_variance", -0.5}, {"approx_kl", 0.09}, {"clip_fraction", 0.45},
            {"actor_kl_skips", 3.0}};
    });
    auto const cards = editor::RunHealth(log);
    Check(Card(cards, "Episode return")->health == editor::Health::Watch, "a falling return is worth a look");
    Check(Card(cards, "Critic accuracy")->health == editor::Health::Problem,
        "a negative explained variance is a problem");
    Check(Card(cards, "Policy change")->health == editor::Health::Problem, "a KL above twice the target is a problem");
    Check(Card(cards, "Clipped samples")->health == editor::Health::Watch, "heavy clipping is worth a look");
    Check(Card(cards, "Skipped actor steps")->health == editor::Health::Watch, "skipped actor steps are worth a look");
    Check(editor::OverallHealth(cards) == editor::Health::Problem, "the worst card decides overall");
}

void EdgeCases()
{
    Check(editor::RunHealth(runtime::RunLog{}).empty(), "no update: no card");

    auto noEpisodes = Journal(8, [](std::size_t, runtime::UpdateRecord& record) {
        record.rewards.episodes = 0;
        record.ppo = {{"max_approx_kl", 0.001}};
    });
    auto const cards = editor::RunHealth(noEpisodes);
    Check(cards.size() == 2, "only the quantities the run journals get a card");
    Check(Card(cards, "Episode return")->health == editor::Health::Watch, "no finished episode is worth a look");
    Check(Card(cards, "Policy change") != nullptr && Card(cards, "Policy change")->metric == "max_approx_kl",
        "Flight's largest KL stands in for the mean KL");

    auto const young = Journal(4, [](std::size_t update, runtime::UpdateRecord& record) {
        record.rewards.episodeReturnMean = static_cast<double>(update);
    });
    Check(editor::RunHealth(young).front().trend == editor::Trend::Unknown, "a trend needs enough updates");

    auto const diverged = Journal(6, [](std::size_t, runtime::UpdateRecord& record) {
        record.ppo = {{"explained_variance", std::numeric_limits<double>::quiet_NaN()}};
    });
    Check(Card(editor::RunHealth(diverged), "Critic accuracy")->health == editor::Health::Problem,
        "a non-finite value is a problem");

    auto const noTarget = Journal(6, [](std::size_t, runtime::UpdateRecord& record) {
        record.ppo = {{"approx_kl", 0.04}};
    }, 0.0);
    Check(Card(editor::RunHealth(noTarget), "Policy change")->health == editor::Health::Good,
        "without a target, 0.05 is the reference");
}

void Durations()
{
    Check(editor::FormatDuration(42.4) == "42 s", "seconds");
    Check(editor::FormatDuration(125.0) == "2 min", "minutes");
    Check(editor::FormatDuration(3.0 * 3600.0 + 5.0 * 60.0) == "3 h 05", "hours and minutes");
    Check(editor::FormatDuration(2.0 * 86400.0 + 4.0 * 3600.0) == "2 d 4 h", "days and hours");
    Check(editor::FormatDuration(-1.0) == "-", "a negative duration is not shown");
}

} // namespace

int main()
{
    HealthyRun();
    ProblemRun();
    EdgeCases();
    Durations();
    if (failures != 0)
    {
        std::cerr << failures << " run health check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "run health checks passed\n";
    return EXIT_SUCCESS;
}
