/**
 * @file PPOKit/tests/evaluation_tests.cpp
 * @brief Evaluation suites: file validation, independent seeds, metrics, gates and report.
 */
#include "ppokit/eval/evaluator.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <cstdlib>
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

int main()
{
    ValidatesFiles();
    RejectsTrainingSeeds();
    MeasuresAndGates();
    BoundsEndlessEpisodes();
    if (failures != 0)
    {
        std::cerr << failures << " evaluation check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit evaluation tests passed\n";
    return EXIT_SUCCESS;
}
