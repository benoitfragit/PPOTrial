/**
 * @file PPOKit/tests/evaluation_tests.cpp
 * @brief Evaluation suites: file validation, independent seeds, metrics, gates and report.
 */
#include "ppokit/eval/evaluator.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>
#include <iostream>
#include <string>

namespace {

using ppokit::eval::Evaluate;
using ppokit::eval::EvaluationSuite;
using ppokit::eval::ParseEvaluationSuite;
using ppokit::runtime::StepEnd;
using ppokit::runtime::StepOutcome;

int failures = 0;

void Check(bool condition, char const* description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

void ExpectError(std::string const& text, std::string const& expected, char const* description)
{
    try
    {
        ParseEvaluationSuite(text, "smoke.json");
    }
    catch (ppokit::schema::SchemaError const& error)
    {
        bool const matches = std::string(error.what()).find(expected) != std::string::npos;
        if (!matches)
        {
            std::cerr << "  message was: " << error.what() << '\n';
        }
        Check(matches, description);
        return;
    }
    Check(false, description);
}

/** @brief Corridor whose goal distance depends on the seed: 3 + seed % 2. */
class SeededCorridor final
{
public:
    explicit SeededCorridor(std::uint32_t seed) noexcept : goal_(3.F + static_cast<float>(seed % 2)) {}

    std::array<float, 1> Reset()
    {
        position_ = 0.F;
        steps_ = 0;
        return {position_};
    }

    StepOutcome<double> Step(std::array<float, 1> const& action, std::array<float, 1>& next)
    {
        position_ += action[0];
        ++steps_;
        next = {position_};
        if (position_ >= goal_)
        {
            return {1.0, StepEnd::Terminated};
        }
        return {-0.1, steps_ == 10 ? StepEnd::Truncated : StepEnd::Running};
    }

private:
    float goal_;
    float position_ = 0.F;
    int steps_ = 0;
};

SeededCorridor MakeCorridor(std::uint32_t seed)
{
    return SeededCorridor(seed);
}

/** @brief Deterministic policy taking a fixed step. */
struct FixedPolicy final
{
    float step;
    std::array<float, 1> Mean(std::array<float, 1> const&) const { return {step}; }
};

/** @brief Task that never ends its episodes, to exercise the safety bound. */
struct EndlessTask final
{
    std::array<float, 1> Reset() { return {0.F}; }
    StepOutcome<double> Step(std::array<float, 1> const&, std::array<float, 1>& next)
    {
        next = {0.F};
        return {0.0, StepEnd::Running};
    }
};

EndlessTask MakeEndless(std::uint32_t)
{
    return {};
}

EvaluationSuite Suite()
{
    return ParseEvaluationSuite(R"({"schema": "ppokit.eval/1", "suite": "smoke", "seeds": [10, 11],
        "episodes": 2, "policy": "deterministic",
        "gates": {"success_rate": ">= 1.0", "mean_length": "<= 3.5"}})", "smoke.json");
}

void ValidatesFiles()
{
    auto const suite = Suite();
    Check(suite.seeds.size() == 2 && suite.episodesPerSeed == 2 && suite.gates.size() == 2, "suite content");
    ExpectError(R"({"schema": "ppokit.eval/1", "suite": "s", "seeds": [1, 1], "episodes": 1})",
        "seeds[1]: duplicate seed 1", "duplicate seed");
    ExpectError(R"({"schema": "ppokit.eval/1", "suite": "s", "seeds": [1], "episodes": 1,
        "gates": {"succes_rate": ">= 1"}})", "gates.succes_rate: unknown metric", "typo in a metric name");
    ExpectError(R"({"schema": "ppokit.eval/1", "suite": "s", "seeds": [1], "episodes": 1,
        "gates": {"success_rate": "> 1"}})", "expected \">= <number>\"", "unsupported comparison");
    ExpectError(R"({"schema": "ppokit.eval/1", "suite": "s", "seeds": [1], "episodes": 1, "policy": "stochastic"})",
        "policy: expected \"deterministic\"", "stochastic evaluation not available yet");
}

void RejectsTrainingSeeds()
{
    bool rejected = false;
    try
    {
        ppokit::eval::RequireDisjointSeeds(Suite(), {7, 11});
    }
    catch (std::invalid_argument const& error)
    {
        rejected = std::string(error.what()).find("reuses training seed 11") != std::string::npos;
    }
    Check(rejected, "an evaluation seed used by training is rejected");
}

void MeasuresAndGates()
{
    FixedPolicy fast{1.F};
    auto campaign = Suite();
    campaign.campaignEnvironment = true;
    bool rejectedScope = false;
    try { Evaluate(campaign, MakeCorridor, fast, 100); }
    catch (std::invalid_argument const&) { rejectedScope = true; }
    Check(rejectedScope, "a seed-only factory cannot silently run a campaign-scoped suite");
    auto const report = Evaluate(Suite(), MakeCorridor, fast, 100);
    Check(report.episodes.size() == 4, "seeds x episodes");
    Check(report.successRate == 1.0 && report.truncationRate == 0.0, "every episode reaches its goal");
    Check(report.meanLength == 3.5, "goals at 3 and 4 steps");
    Check(report.Passed(), "gates pass");
    Check(report.ToJson().find("\"passed\": true") != std::string::npos, "JSON report");

    FixedPolicy slow{0.1F};
    auto const failed = Evaluate(Suite(), MakeCorridor, slow, 100);
    Check(failed.truncationRate == 1.0 && !failed.Passed(), "a policy that never arrives fails its gates");
    Check(!failed.gates[0].passed && failed.gates[0].value == 0.0, "gate verdict carries the measured value");
}

void BoundsEndlessEpisodes()
{
    FixedPolicy policy{0.F};
    bool rejected = false;
    try
    {
        Evaluate(Suite(), MakeEndless, policy, 50);
    }
    catch (std::runtime_error const&)
    {
        rejected = true;
    }
    Check(rejected, "an episode that never ends is reported instead of looping forever");
}

} // namespace

/** @brief Episode metrics of a landing task: a flag and a number. */
std::vector<ppokit::schema::EpisodeMetric> LandingMetrics()
{
    std::string const task = R"({ "schema": "ppokit.task/1", "name": "landing",
      "observation": { "state": { "fields": ["x"] } },
      "actions": { "move": { "head": "gaussian", "names": ["dx"] } },
      "rewards": { "R": {} },
      "metrics": { "stopped": { "kind": "flag", "label": "Stopped" },
                   "stop_m": { "kind": "number", "unit": "m" } } })";
    return ppokit::schema::ParseTaskSchema(task, "task.json").metrics;
}

std::string const LandingSuite = R"({ "schema": "ppokit.eval/1", "suite": "validation",
  "seeds": { "first": 4342, "count": 4 }, "episodes": 1,
  "gates": { "stopped.rate": ">= 0.75" },
  "during_training": { "every_updates": 10 },
  "goal": { "stop_m.mean": "<= 450" },
  "best": [ { "metric": "stopped.rate", "prefer": "higher" }, { "metric": "stop_m.mean", "prefer": "lower" } ] })";

/** @brief A finished campaign of the landing suite: `stops` stops, at `meters` each. */
ppokit::eval::EvaluationReport Campaign(EvaluationSuite const& suite, std::size_t stops, double meters)
{
    ppokit::eval::EvaluationReport report;
    report.suite = suite.name;
    double const nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t index = 0; index < suite.Episodes(); ++index)
    {
        ppokit::eval::EpisodeResult episode;
        episode.seed = suite.seeds[index];
        episode.length = 10;
        episode.terminated = true;
        bool const stopped = index < stops;
        episode.metrics = {stopped ? 1.0 : 0.0, stopped ? meters : nan};
        report.episodes.push_back(episode);
    }
    ppokit::eval::Summarize(suite, report);
    return report;
}

/** @brief task.json metrics, suites on them: range seeds, gates, training rhythm and BEST ranking. */
void EpisodeMetrics()
{
    auto const metrics = LandingMetrics();
    Check(metrics.size() == 2 && metrics[0].kind == ppokit::schema::MetricKind::Flag
            && metrics[1].unit == "m" && metrics[0].label == "Stopped",
        "task.json metrics keep their kind, label and unit");
    EvaluationSuite const suite = ParseEvaluationSuite(LandingSuite, "validation.json", metrics);
    Check(suite.seeds == std::vector<std::uint32_t>{4342, 4343, 4344, 4345}, "a seed range lists its seeds");
    Check(suite.everyUpdates == 10 && suite.best.size() == 2 && !suite.best[1].higher,
        "the suite runs during training and ranks the BEST");
    auto const names = ppokit::eval::MetricNames(metrics);
    Check(names.back() == "stop_m.mean" && names[names.size() - 2] == "stopped.rate",
        "a flag gives m.rate, a number m.mean");

    auto const good = Campaign(suite, 3, 400.0);
    Check(good.Metric("stopped.rate") == 0.75 && good.Metric("stop_m.mean") == 400.0,
        "rates count the flags, means skip the episodes where the value does not apply");
    Check(good.Passed(), "a gate on an episode metric passes");
    Check(good.GoalReached() && !Campaign(suite, 3, 500.0).GoalReached() && Campaign(suite, 3, 500.0).Passed(),
        "the goal adds its conditions to the gates");
    Check(!Campaign(suite, 2, 400.0).Passed(), "and fails below its threshold");
    auto const none = Campaign(suite, 0, 0.0);
    Check(std::isnan(none.Metric("stop_m.mean")), "a mean with no value is NaN");

    Check(ppokit::eval::RanksAbove(Campaign(suite, 4, 500.0), good, suite.best), "a higher stop rate wins");
    Check(ppokit::eval::RanksAbove(Campaign(suite, 3, 300.0), good, suite.best),
        "at equal rate, a shorter stop wins");
    Check(!ppokit::eval::RanksAbove(Campaign(suite, 3, 400.0), good, suite.best), "an equal campaign does not win");
    Check(ppokit::eval::RanksAbove(good, none, suite.best) && !ppokit::eval::RanksAbove(none, good, suite.best),
        "a NaN value loses");

    auto const restored = ppokit::eval::ReportFromJson(good.ToJson(), "best.json");
    Check(restored.Metric("stopped.rate") == 0.75 && restored.Metric("stop_m.mean") == 400.0 && restored.Passed()
            && restored.GoalReached() && restored.goal.size() == 1,
        "a report reads back from its JSON");
    Check(std::isnan(ppokit::eval::ReportFromJson(none.ToJson(), "best.json").Metric("stop_m.mean")),
        "NaN is written as null and read back as NaN");

    EvaluationSuite const again =
        ParseEvaluationSuite(ppokit::eval::SuiteToJson(suite), "validation.json", metrics);
    Check(again.seeds == suite.seeds && again.gates.size() == 1 && again.gates[0].threshold == 0.75
            && again.everyUpdates == 10 && again.best.size() == 2 && again.best[0].metric == "stopped.rate"
            && again.goal.size() == 1 && again.goal[0].comparison == ppokit::eval::Comparison::LessOrEqual,
        "a suite written by SuiteToJson reads back the same");

    ExpectError(R"({ "schema": "ppokit.eval/1", "suite": "s", "seeds": [1], "episodes": 1,
        "gates": { "landed.rate": ">= 1" } })",
        "unknown metric 'landed.rate'", "a gate on an undeclared metric is refused");
    ExpectError(R"({ "schema": "ppokit.eval/1", "suite": "s", "seeds": { "first": 1 }, "episodes": 1 })",
        "needs", "a range without count is refused");
}

int main()
{
    ValidatesFiles();
    RejectsTrainingSeeds();
    MeasuresAndGates();
    BoundsEndlessEpisodes();
    EpisodeMetrics();
    if (failures != 0)
    {
        std::cerr << failures << " evaluation check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit evaluation tests passed\n";
    return EXIT_SUCCESS;
}
