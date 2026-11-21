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
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
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

/** @brief Validated content of an evaluation file. */
struct EvaluationSuite final
{
    std::string name;
    /** @brief One task instance per seed; seeds must not be used by training. */
    std::vector<std::uint32_t> seeds;
    std::size_t episodesPerSeed = 1;
    std::vector<Gate> gates;
};

/** @brief Names of the metrics a gate may use. */
std::vector<std::string> const& BuiltInMetrics();

/**
 * @brief Parses and validates an evaluation suite.
 * @throws ppokit::schema::SchemaError naming the faulty key, e.g. "smoke.json: gates.succes_rate: unknown metric".
 */
EvaluationSuite ParseEvaluationSuite(std::string_view text, std::string_view sourceName);

/** @brief Reads and parses an evaluation file. */
EvaluationSuite LoadEvaluationSuite(std::filesystem::path const& path);

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
    std::vector<GateResult> gates;

    /** @brief True when every gate passed (and vacuously true without gates). */
    bool Passed() const noexcept;
    /** @brief Value of a built-in metric by name. @throws std::invalid_argument for an unknown name. */
    double Metric(std::string const& name) const;
    /** @brief Indented JSON report: metrics, gates and episodes, for logs, CI and the studio. */
    std::string ToJson() const;
};

/** @brief Computes the aggregate metrics and the gate verdicts from `report.episodes`. */
void Summarize(EvaluationSuite const& suite, EvaluationReport& report);

} // namespace ppokit::eval
