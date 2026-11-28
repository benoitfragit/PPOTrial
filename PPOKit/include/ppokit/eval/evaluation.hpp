/**
 * @file PPOKit/include/ppokit/eval/evaluation.hpp
 * @brief Evaluation suites (files `eval/<suite>.json`), their results and gates.
 * @details An evaluation replays a frozen policy on seeds that training never uses,
 * and checks gates. AGENTS.md requires this independent-seed evaluation before any
 * learned behaviour is trusted: unit tests do not prove convergence.
 *
 * @code{.json}
 * {
 *   "schema": "ppokit.eval/1",
 *   "suite": "smoke",
 *   "seeds": [101, 202, 303],
 *   "episodes": 5,
 *   "policy": "deterministic",
 *   "gates": { "success_rate": ">= 0.9", "mean_return": ">= 3.0" }
 * }
 * @endcode
 *
 * Built-in metrics: `mean_return`, `success_rate` (episodes that terminated, e.g. a
 * landing or a reached goal), `truncation_rate` (time limit) and `mean_length`.
 *
 * The episode metrics of task.json ("metrics", reported by the environment) add, for each
 * metric `m`: `m.rate` (share of episodes where a flag is 1) and `m.mean` (mean over the
 * episodes where the value applies, i.e. is not NaN). A suite may also run during training
 * and rank the BEST policy:
 *
 * @code{.json}
 * {
 *   "schema": "ppokit.eval/1",
 *   "suite": "validation",
 *   "seeds": { "first": 4342, "count": 20 },
 *   "episodes": 1,
 *   "gates": { "stopped.rate": ">= 0.92" },
 *   "during_training": { "every_updates": 10 },
 *   "goal": { "difficulty.mean": ">= 1" },
 *   "best": [ { "metric": "difficulty.mean", "prefer": "higher" },
 *             { "metric": "stopped.rate", "prefer": "higher" },
 *             { "metric": "stop_m.mean", "prefer": "lower" } ]
 * }
 * @endcode
 *
 * A policy that passes every gate becomes the BEST when it beats the current one on the
 * first "best" criterion that differs (a NaN value always loses). The "goal" adds
 * conditions, in the gates' form, for the training to be finished (ppokit-run train
 * --until-passed): a campaign reaches the goal when it passes its gates and the goal's
 * conditions hold, e.g. a landing validated on the full task and not only on an easier
 * stage of the curriculum.
 */
#pragma once

#include "ppokit/schema/task_schema.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ppokit::eval {

/** @brief Comparison of a gate. */
enum class Comparison : std::uint8_t
{
    GreaterOrEqual,
    LessOrEqual,
};

/** @brief Pass condition on one metric, e.g. "success_rate >= 0.9". */
struct Gate final
{
    std::string metric;
    Comparison comparison = Comparison::GreaterOrEqual;
    double threshold = 0;
};

/** @brief One criterion of the BEST ranking. */
struct RankKey final
{
    std::string metric;
    /** @brief True when a higher value is better. */
    bool higher = true;
};

/** @brief Validated content of an evaluation file. */
struct EvaluationSuite final
{
    std::string name;
    /** @brief One task instance per seed; seeds must not be used by training. */
    std::vector<std::uint32_t> seeds;
    std::size_t episodesPerSeed = 1;
    /** @brief Keep one environment across the campaign, resetting it explicitly on each seed. */
    bool campaignEnvironment = false;
    std::vector<Gate> gates;
    /** @brief Further conditions for the training to be finished; empty: passing the gates is enough. */
    std::vector<Gate> goal;
    /** @brief Updates between two campaigns during training; 0 when the suite only runs on demand. */
    std::size_t everyUpdates = 0;
    /** @brief BEST ranking, most important first; empty when the suite keeps no BEST. */
    std::vector<RankKey> best;
    /** @brief Episode metrics of the task the suite was read for (task.json order). */
    std::vector<schema::EpisodeMetric> metrics;

    /** @brief Episodes of one campaign: seeds times episodes per seed. */
    std::size_t Episodes() const noexcept { return seeds.size() * episodesPerSeed; }
};

/** @brief Names of the built-in metrics. */
std::vector<std::string> const& BuiltInMetrics();

/** @brief Names a gate or a ranking may use: the built-in ones, then m.rate and m.mean of each episode metric. */
std::vector<std::string> MetricNames(std::vector<schema::EpisodeMetric> const& metrics);

/**
 * @brief Parses and validates an evaluation suite.
 * @param metrics Episode metrics of the task (task.json "metrics"); none by default.
 * @throws ppokit::schema::SchemaError naming the faulty key, e.g. "smoke.json: gates.succes_rate: unknown metric".
 */
EvaluationSuite ParseEvaluationSuite(std::string_view text, std::string_view sourceName,
    std::vector<schema::EpisodeMetric> const& metrics = {});

/** @brief Reads and parses an evaluation file. */
EvaluationSuite LoadEvaluationSuite(std::filesystem::path const& path,
    std::vector<schema::EpisodeMetric> const& metrics = {});

/** @brief The suite as JSON text (the format ParseEvaluationSuite reads), for the studio. */
std::string SuiteToJson(EvaluationSuite const& suite);

/**
 * @brief Rejects a suite that shares a seed with training.
 * @throws std::invalid_argument naming the first shared seed.
 */
void RequireDisjointSeeds(EvaluationSuite const& suite, std::vector<std::uint32_t> const& trainingSeeds);

/** @brief Outcome of one evaluation episode. */
struct EpisodeResult final
{
    std::uint32_t seed = 0;
    double episodeReturn = 0;
    std::size_t length = 0;
    bool terminated = false;
    /** @brief Episode metrics reported by the environment, in task.json order. */
    std::vector<double> metrics;
};

/** @brief An aggregate of the episode metrics, e.g. "stopped.rate". */
struct NamedValue final
{
    std::string name;
    double value = 0;
};

/** @brief Value of a gate after evaluation. */
struct GateResult final
{
    Gate gate;
    double value = 0;
    bool passed = false;
};

/** @brief All episodes of a suite, their aggregate metrics and the gate verdicts. */
struct EvaluationReport final
{
    std::string suite;
    std::vector<EpisodeResult> episodes;
    double meanReturn = 0;
    double successRate = 0;
    double truncationRate = 0;
    double meanLength = 0;
    /** @brief m.rate and m.mean of every episode metric (MetricNames order, built-ins excluded). */
    std::vector<NamedValue> aggregates;
    std::vector<GateResult> gates;
    /** @brief The goal's conditions, evaluated like the gates. */
    std::vector<GateResult> goal;

    /** @brief SHA-256 of the actor opened for this campaign. Required for a BEST promotion. */
    std::string policySha256;

    /** @brief True when every gate passed (and vacuously true without gates). */
    bool Passed() const noexcept;
    /** @brief Passed, and every condition of the goal holds. */
    bool GoalReached() const noexcept;
    /** @brief Value of a built-in metric by name. @throws std::invalid_argument for an unknown name. */
    double Metric(std::string const& name) const;
    /** @brief Indented JSON report: metrics, gates and episodes, for logs, CI and the studio. */
    std::string ToJson() const;
};

/** @brief Computes the aggregate metrics and the gate verdicts from `report.episodes`. */
void Summarize(EvaluationSuite const& suite, EvaluationReport& report);

/**
 * @brief Whether `candidate` ranks above `incumbent` by the suite's "best" criteria: the
 *        first criterion whose values differ decides; a NaN value loses to any number.
 * @return False when every criterion is equal, or when the suite has no ranking.
 */
bool RanksAbove(EvaluationReport const& candidate, EvaluationReport const& incumbent,
    std::vector<RankKey> const& ranking);

/** @brief Reads a report written by EvaluationReport::ToJson (metrics, aggregates and gates). */
EvaluationReport ReportFromJson(std::string_view text, std::string_view sourceName);

/** @brief Summary and optional actor attribution of an on-demand runner evaluation. */
struct SavedEvaluation final
{
    EvaluationReport report;
    std::size_t episodeCount{};
    std::string policyRole;
    std::optional<std::uint64_t> update;
};

/** @brief Path written by eval for this suite, or empty while an editor holds an invalid name. */
std::filesystem::path EvaluationReportPath(std::filesystem::path const& checkpoints, std::string const& suite);

/** @brief Read the completed evaluation once, keeping its aggregate results and episode count. */
SavedEvaluation ReadSavedEvaluation(std::filesystem::path const& path);

} // namespace ppokit::eval
