/**
 * @file PPOKit/src/eval/evaluation.cpp
 * @brief Evaluation suite parsing, metrics, gates and JSON report.
 */
#include "ppokit/eval/evaluation.hpp"

#include "../schema/json_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <sstream>
#include <stdexcept>

namespace ppokit::eval {
namespace {

using schema::detail::ErrorReporter;
using schema::detail::Json;
using schema::detail::RequireKnownKeys;
using schema::detail::RequireObject;
using schema::detail::RequirePositiveInteger;

/** @brief Format identifier accepted in the "schema" key. */
constexpr char const* EvaluationSchemaVersion = "ppokit.eval/1";

std::vector<std::uint32_t> ReadSeeds(Json const& value, ErrorReporter const& errors)
{
    if (!value.is_array() || value.empty())
    {
        errors.Fail("seeds", "expected a non-empty array of seeds");
    }
    std::vector<std::uint32_t> seeds;
    std::set<std::uint32_t> seen;
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        std::string const path = "seeds[" + std::to_string(index) + "]";
        if (!value[index].is_number_unsigned() || value[index].get<std::uint64_t>() > UINT32_MAX)
        {
            errors.Fail(path, "expected a 32-bit unsigned integer");
        }
        auto const seed = value[index].get<std::uint32_t>();
        if (!seen.insert(seed).second)
        {
            errors.Fail(path, "duplicate seed " + std::to_string(seed));
        }
        seeds.push_back(seed);
    }
    return seeds;
}

/** @brief Parses "<op> <number>" with op ">=" or "<=". */
Gate ReadGate(std::string const& metric, Json const& value, ErrorReporter const& errors)
{
    std::string const path = "gates." + metric;
    auto const& metrics = BuiltInMetrics();
    if (std::find(metrics.begin(), metrics.end(), metric) == metrics.end())
    {
        errors.Fail(path, "unknown metric (available: mean_return, success_rate, truncation_rate, mean_length)");
    }
    if (!value.is_string())
    {
        errors.Fail(path, "expected a condition such as \">= 0.9\"");
    }
    std::istringstream condition(value.get<std::string>());
    std::string comparison;
    double threshold = 0;
    condition >> comparison >> threshold;
    bool const complete = static_cast<bool>(condition) && (condition >> std::ws).eof();
    Gate gate;
    gate.metric = metric;
    gate.threshold = threshold;
    if (!complete || !std::isfinite(threshold) || (comparison != ">=" && comparison != "<="))
    {
        errors.Fail(path, "expected \">= <number>\" or \"<= <number>\"");
    }
    gate.comparison = comparison == ">=" ? Comparison::GreaterOrEqual : Comparison::LessOrEqual;
    return gate;
}

bool GatePasses(Gate const& gate, double value) noexcept
{
    if (gate.comparison == Comparison::GreaterOrEqual)
    {
        return value >= gate.threshold;
    }
    return value <= gate.threshold;
}

} // namespace

std::vector<std::string> const& BuiltInMetrics()
{
    static std::vector<std::string> const metrics{"mean_return", "success_rate", "truncation_rate", "mean_length"};
    return metrics;
}

EvaluationSuite ParseEvaluationSuite(std::string_view text, std::string_view sourceName)
{
    ErrorReporter const errors(sourceName);
    Json const root = schema::detail::ParseRootObject(text, errors);
    RequireKnownKeys(root, {"schema", "suite", "seeds", "episodes", "policy", "gates"}, "", errors);
    if (!root.contains("schema") || root.at("schema") != EvaluationSchemaVersion)
    {
        errors.Fail("schema", std::string("expected \"") + EvaluationSchemaVersion + "\"");
    }
    for (char const* required : {"suite", "seeds", "episodes"})
    {
        if (!root.contains(required))
        {
            errors.Fail(required, "missing required key");
        }
    }
    EvaluationSuite suite;
    suite.name = schema::detail::RequireIdentifier(root.at("suite"), "suite", errors);
    suite.seeds = ReadSeeds(root.at("seeds"), errors);
    suite.episodesPerSeed = RequirePositiveInteger(root.at("episodes"), "episodes", errors);
    if (root.contains("policy") && root.at("policy") != "deterministic")
    {
        errors.Fail("policy", "expected \"deterministic\" (the only mode of this version)");
    }
    if (root.contains("gates"))
    {
        Json const& gates = root.at("gates");
        RequireObject(gates, "gates", errors);
        for (auto const& item : gates.items())
        {
            suite.gates.push_back(ReadGate(item.key(), item.value(), errors));
        }
    }
    return suite;
}

EvaluationSuite LoadEvaluationSuite(std::filesystem::path const& path)
{
    return ParseEvaluationSuite(schema::detail::ReadTextFile(path), path.filename().string());
}

void RequireDisjointSeeds(EvaluationSuite const& suite, std::vector<std::uint32_t> const& trainingSeeds)
{
    for (std::uint32_t const seed : suite.seeds)
    {
        if (std::find(trainingSeeds.begin(), trainingSeeds.end(), seed) != trainingSeeds.end())
        {
            throw std::invalid_argument("PPOKit evaluation: suite '" + suite.name + "' reuses training seed "
                + std::to_string(seed) + "; evaluation seeds must be independent");
        }
    }
}

bool EvaluationReport::Passed() const noexcept
{
    for (auto const& result : gates)
    {
        if (!result.passed)
        {
            return false;
        }
    }
    return true;
}

double EvaluationReport::Metric(std::string const& name) const
{
    if (name == "mean_return")
    {
        return meanReturn;
    }
    if (name == "success_rate")
    {
        return successRate;
    }
    if (name == "truncation_rate")
    {
        return truncationRate;
    }
    if (name == "mean_length")
    {
        return meanLength;
    }
    throw std::invalid_argument("PPOKit evaluation: unknown metric '" + name + "'");
}

std::string EvaluationReport::ToJson() const
{
    Json json;
    json["suite"] = suite;
    json["passed"] = Passed();
    json["metrics"] = Json{{"mean_return", meanReturn}, {"success_rate", successRate},
        {"truncation_rate", truncationRate}, {"mean_length", meanLength}};
    json["gates"] = Json::array();
    for (auto const& result : gates)
    {
        char const* comparison = result.gate.comparison == Comparison::GreaterOrEqual ? ">=" : "<=";
        json["gates"].push_back(Json{{"metric", result.gate.metric}, {"comparison", comparison},
            {"threshold", result.gate.threshold}, {"value", result.value}, {"passed", result.passed}});
    }
    json["episodes"] = Json::array();
    for (auto const& episode : episodes)
    {
        json["episodes"].push_back(Json{{"seed", episode.seed}, {"return", episode.episodeReturn},
            {"length", episode.length}, {"terminated", episode.terminated}});
    }
    return json.dump(2);
}

void Summarize(EvaluationSuite const& suite, EvaluationReport& report)
{
    double returnSum = 0;
    double lengthSum = 0;
    std::size_t successes = 0;
    for (auto const& episode : report.episodes)
    {
        returnSum += episode.episodeReturn;
        lengthSum += static_cast<double>(episode.length);
        successes += episode.terminated ? 1 : 0;
    }
    if (!report.episodes.empty())
    {
        double const count = static_cast<double>(report.episodes.size());
        report.meanReturn = returnSum / count;
        report.meanLength = lengthSum / count;
        report.successRate = static_cast<double>(successes) / count;
        report.truncationRate = 1.0 - report.successRate;
    }
    report.gates.clear();
    for (auto const& gate : suite.gates)
    {
        double const value = report.Metric(gate.metric);
        report.gates.push_back({gate, value, GatePasses(gate, value)});
    }
}

} // namespace ppokit::eval
