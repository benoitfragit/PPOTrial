/**
 * @file PPOKit/src/eval/evaluation.cpp
 * @brief Evaluation suite parsing, metrics, gates and JSON report.
 */
#include "ppokit/eval/evaluation.hpp"

#include "../schema/json_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

/** @brief A 32-bit unsigned integer at `path`. */
std::uint32_t ReadSeed(Json const& value, std::string const& path, ErrorReporter const& errors)
{
    if (!value.is_number_unsigned() || value.get<std::uint64_t>() > UINT32_MAX)
    {
        errors.Fail(path, "expected a 32-bit unsigned integer");
    }
    return value.get<std::uint32_t>();
}

/** @brief "seeds": {"first": 4342, "count": 20}, i.e. 4342 to 4361. */
std::vector<std::uint32_t> ReadSeedRange(Json const& value, ErrorReporter const& errors)
{
    RequireKnownKeys(value, {"first", "count"}, "seeds", errors);
    if (!value.contains("first") || !value.contains("count"))
    {
        errors.Fail("seeds", "a range needs \"first\" and \"count\"");
    }
    std::uint32_t const first = ReadSeed(value.at("first"), "seeds.first", errors);
    std::size_t const count = RequirePositiveInteger(value.at("count"), "seeds.count", errors);
    if (count - 1 > static_cast<std::size_t>(UINT32_MAX - first))
    {
        errors.Fail("seeds.count", "the range goes past the largest 32-bit seed");
    }
    std::vector<std::uint32_t> seeds;
    for (std::size_t index = 0; index < count; ++index)
    {
        seeds.push_back(first + static_cast<std::uint32_t>(index));
    }
    return seeds;
}

/** @brief "seeds": an array of distinct seeds, or a range. */
std::vector<std::uint32_t> ReadSeeds(Json const& value, ErrorReporter const& errors)
{
    if (value.is_object())
    {
        return ReadSeedRange(value, errors);
    }
    if (!value.is_array() || value.empty())
    {
        errors.Fail("seeds", "expected a non-empty array of seeds, or {\"first\": seed, \"count\": n}");
    }
    std::vector<std::uint32_t> seeds;
    std::set<std::uint32_t> seen;
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        std::string const path = "seeds[" + std::to_string(index) + "]";
        std::uint32_t const seed = ReadSeed(value[index], path, errors);
        if (!seen.insert(seed).second)
        {
            errors.Fail(path, "duplicate seed " + std::to_string(seed));
        }
        seeds.push_back(seed);
    }
    return seeds;
}

/** @brief The names joined by ", ", for error messages. */
std::string Joined(std::vector<std::string> const& names)
{
    std::string text;
    for (std::string const& name : names)
    {
        if (!text.empty())
        {
            text += ", ";
        }
        text += name;
    }
    return text;
}

/** @brief Fails unless `metric` is one of `names`. */
void RequireMetric(std::string const& metric, std::vector<std::string> const& names, std::string const& path,
    ErrorReporter const& errors)
{
    if (std::find(names.begin(), names.end(), metric) == names.end())
    {
        errors.Fail(path, "unknown metric '" + metric + "' (available: " + Joined(names) + ")");
    }
}

/**
 * @brief Parses "<op> <number>" with op ">=" or "<=".
 * @param section "gates" or "goal", for the error path.
 */
Gate ReadGate(std::string const& section, std::string const& metric, Json const& value,
    std::vector<std::string> const& names, ErrorReporter const& errors)
{
    std::string const path = section + "." + metric;
    RequireMetric(metric, names, path, errors);
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

/** @brief "during_training": {"every_updates": 10}. */
std::size_t ReadDuringTraining(Json const& value, ErrorReporter const& errors)
{
    RequireObject(value, "during_training", errors);
    RequireKnownKeys(value, {"every_updates"}, "during_training", errors);
    if (!value.contains("every_updates"))
    {
        errors.Fail("during_training", "missing \"every_updates\"");
    }
    return RequirePositiveInteger(value.at("every_updates"), "during_training.every_updates", errors);
}

/** @brief "best": [{"metric": "stopped.rate", "prefer": "higher"}, ...]. */
std::vector<RankKey> ReadRanking(Json const& value, std::vector<std::string> const& names, ErrorReporter const& errors)
{
    if (!value.is_array() || value.empty())
    {
        errors.Fail("best", "expected a non-empty array of {\"metric\": name, \"prefer\": \"higher\" | \"lower\"}");
    }
    std::vector<RankKey> ranking;
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        std::string const path = "best[" + std::to_string(index) + "]";
        Json const& item = value[index];
        RequireObject(item, path, errors);
        RequireKnownKeys(item, {"metric", "prefer"}, path, errors);
        if (!item.contains("metric") || !item.at("metric").is_string())
        {
            errors.Fail(path + ".metric", "expected a metric name");
        }
        RankKey key;
        key.metric = item.at("metric").get<std::string>();
        RequireMetric(key.metric, names, path + ".metric", errors);
        std::string const prefer = item.value("prefer", std::string("higher"));
        if (prefer != "higher" && prefer != "lower")
        {
            errors.Fail(path + ".prefer", "expected \"higher\" or \"lower\"");
        }
        key.higher = prefer == "higher";
        ranking.push_back(std::move(key));
    }
    return ranking;
}

/** @brief Aggregate name of a metric: m.rate for a flag, m.mean for a number. */
std::string AggregateName(schema::EpisodeMetric const& metric)
{
    if (metric.kind == schema::MetricKind::Flag)
    {
        return metric.name + ".rate";
    }
    return metric.name + ".mean";
}

/** @brief Value of metric `index` in an episode, NaN when the episode did not report it. */
double EpisodeValue(EpisodeResult const& episode, std::size_t index) noexcept
{
    if (index >= episode.metrics.size())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return episode.metrics[index];
}

/**
 * @brief Share of episodes where a flag is 1, or mean of a number over the episodes where it
 *        applies; NaN when no episode gives a value.
 */
double Aggregate(std::vector<EpisodeResult> const& episodes, std::size_t index, schema::MetricKind kind) noexcept
{
    double sum = 0;
    std::size_t counted = 0;
    for (EpisodeResult const& episode : episodes)
    {
        double const value = EpisodeValue(episode, index);
        if (kind == schema::MetricKind::Flag)
        {
            sum += value == 1.0 ? 1.0 : 0.0;
            ++counted;
        }
        else if (std::isfinite(value))
        {
            sum += value;
            ++counted;
        }
    }
    if (counted == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / static_cast<double>(counted);
}

/** @brief A number for JSON: NaN (does not apply) becomes null. */
Json NumberOrNull(double value)
{
    if (std::isfinite(value))
    {
        return value;
    }
    return nullptr;
}

double NumberOrNaN(Json const& value)
{
    if (value.is_number())
    {
        return value.get<double>();
    }
    return std::numeric_limits<double>::quiet_NaN();
}

/** @brief Seeds as JSON: a range when they follow each other, else the array. */
Json SeedsToJson(std::vector<std::uint32_t> const& seeds)
{
    bool consecutive = !seeds.empty();
    for (std::size_t index = 1; consecutive && index < seeds.size(); ++index)
    {
        consecutive = seeds[index] == seeds[index - 1] + 1;
    }
    if (consecutive && seeds.size() > 1)
    {
        return Json{{"first", seeds.front()}, {"count", seeds.size()}};
    }
    return Json(seeds);
}

char const* ComparisonText(Comparison comparison) noexcept
{
    if (comparison == Comparison::GreaterOrEqual)
    {
        return ">=";
    }
    return "<=";
}

/** @brief {"metric": ">= threshold", ...}, the form of "gates" and "goal". */
Json ConditionsToJson(std::vector<Gate> const& conditions)
{
    Json json = Json::object();
    for (Gate const& gate : conditions)
    {
        std::ostringstream condition;
        condition << ComparisonText(gate.comparison) << ' ' << gate.threshold;
        json[gate.metric] = condition.str();
    }
    return json;
}

/** @brief Each condition with its value in the report. */
std::vector<GateResult> Verdicts(std::vector<Gate> const& conditions, EvaluationReport const& report)
{
    std::vector<GateResult> results;
    for (Gate const& gate : conditions)
    {
        double const value = report.Metric(gate.metric);
        results.push_back({gate, value, GatePasses(gate, value)});
    }
    return results;
}

/** @brief The conditions of a report as JSON. */
Json VerdictsToJson(std::vector<GateResult> const& results)
{
    Json json = Json::array();
    for (auto const& result : results)
    {
        json.push_back(Json{{"metric", result.gate.metric}, {"comparison", ComparisonText(result.gate.comparison)},
            {"threshold", result.gate.threshold}, {"value", NumberOrNull(result.value)}, {"passed", result.passed}});
    }
    return json;
}

/** @brief Conditions written by VerdictsToJson. */
std::vector<GateResult> VerdictsFromJson(Json const& array)
{
    std::vector<GateResult> results;
    for (Json const& gate : array)
    {
        GateResult result;
        result.gate.metric = gate.at("metric").get<std::string>();
        if (gate.at("comparison").get<std::string>() == "<=")
        {
            result.gate.comparison = Comparison::LessOrEqual;
        }
        result.gate.threshold = gate.at("threshold").get<double>();
        result.value = NumberOrNaN(gate.at("value"));
        result.passed = gate.at("passed").get<bool>();
        results.push_back(result);
    }
    return results;
}

} // namespace

std::vector<std::string> const& BuiltInMetrics()
{
    static std::vector<std::string> const metrics{"mean_return", "success_rate", "truncation_rate", "mean_length"};
    return metrics;
}

std::vector<std::string> MetricNames(std::vector<schema::EpisodeMetric> const& metrics)
{
    std::vector<std::string> names = BuiltInMetrics();
    for (schema::EpisodeMetric const& metric : metrics)
    {
        names.push_back(AggregateName(metric));
    }
    return names;
}

EvaluationSuite ParseEvaluationSuite(std::string_view text, std::string_view sourceName,
    std::vector<schema::EpisodeMetric> const& metrics)
{
    ErrorReporter const errors(sourceName);
    Json const root = schema::detail::ParseRootObject(text, errors);
    RequireKnownKeys(root,
        {"schema", "suite", "seeds", "episodes", "policy", "gates", "goal", "during_training", "best", "environment_scope"}, "", errors);
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
    suite.metrics = metrics;
    if (root.contains("environment_scope"))
    {
        auto const& scope = root.at("environment_scope");
        if (scope != "campaign" && scope != "seed")
            errors.Fail("environment_scope", "expected campaign or seed");
        suite.campaignEnvironment = scope == "campaign";
    }
    std::vector<std::string> const names = MetricNames(metrics);
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
            suite.gates.push_back(ReadGate("gates", item.key(), item.value(), names, errors));
        }
    }
    if (root.contains("goal"))
    {
        Json const& goal = root.at("goal");
        RequireObject(goal, "goal", errors);
        for (auto const& item : goal.items())
        {
            suite.goal.push_back(ReadGate("goal", item.key(), item.value(), names, errors));
        }
    }
    if (root.contains("during_training"))
    {
        suite.everyUpdates = ReadDuringTraining(root.at("during_training"), errors);
    }
    if (root.contains("best"))
    {
        suite.best = ReadRanking(root.at("best"), names, errors);
    }
    return suite;
}

EvaluationSuite LoadEvaluationSuite(std::filesystem::path const& path, std::vector<schema::EpisodeMetric> const& metrics)
{
    return ParseEvaluationSuite(schema::detail::ReadTextFile(path), path.filename().string(), metrics);
}

std::string SuiteToJson(EvaluationSuite const& suite)
{
    // Json (ordered) keeps the keys in the order a person reads them.
    Json json;
    json["schema"] = EvaluationSchemaVersion;
    json["suite"] = suite.name;
    json["seeds"] = SeedsToJson(suite.seeds);
    json["episodes"] = suite.episodesPerSeed;
    json["policy"] = "deterministic";
    if (suite.campaignEnvironment) json["environment_scope"] = "campaign";
    json["gates"] = ConditionsToJson(suite.gates);
    if (!suite.goal.empty())
    {
        json["goal"] = ConditionsToJson(suite.goal);
    }
    if (suite.everyUpdates != 0)
    {
        json["during_training"] = {{"every_updates", suite.everyUpdates}};
    }
    if (!suite.best.empty())
    {
        Json ranking = Json::array();
        for (RankKey const& key : suite.best)
        {
            ranking.push_back({{"metric", key.metric}, {"prefer", key.higher ? "higher" : "lower"}});
        }
        json["best"] = ranking;
    }
    return json.dump(2) + "\n";
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

bool EvaluationReport::GoalReached() const noexcept
{
    if (!Passed())
    {
        return false;
    }
    for (auto const& result : goal)
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
    for (NamedValue const& aggregate : aggregates)
    {
        if (aggregate.name == name)
        {
            return aggregate.value;
        }
    }
    throw std::invalid_argument("PPOKit evaluation: unknown metric '" + name + "'");
}

std::string EvaluationReport::ToJson() const
{
    Json json;
    json["suite"] = suite;
    if (!policySha256.empty()) json["policy_sha256"] = policySha256;
    json["passed"] = Passed();
    json["metrics"] = Json{{"mean_return", meanReturn}, {"success_rate", successRate},
        {"truncation_rate", truncationRate}, {"mean_length", meanLength}};
    json["aggregates"] = Json::object();
    for (NamedValue const& aggregate : aggregates)
    {
        json["aggregates"][aggregate.name] = NumberOrNull(aggregate.value);
    }
    json["gates"] = VerdictsToJson(gates);
    json["goal"] = VerdictsToJson(goal);
    json["goal_reached"] = GoalReached();
    json["episodes"] = Json::array();
    for (auto const& episode : episodes)
    {
        Json values = Json::array();
        for (double const value : episode.metrics)
        {
            values.push_back(NumberOrNull(value));
        }
        json["episodes"].push_back(Json{{"seed", episode.seed}, {"return", episode.episodeReturn},
            {"length", episode.length}, {"terminated", episode.terminated}, {"metrics", values}});
    }
    return json.dump(2);
}

namespace {
/** @brief Parse the summary shared by a BEST manifest and an on-demand evaluation. */
EvaluationReport ReadReportSummary(Json const& root, ErrorReporter const& errors)
{
    EvaluationReport report;
    try
    {
        report.suite = root.at("suite").get<std::string>();
        report.policySha256 = root.value("policy_sha256", std::string{});
        Json const& metrics = root.at("metrics");
        report.meanReturn = metrics.at("mean_return").get<double>();
        report.successRate = metrics.at("success_rate").get<double>();
        report.truncationRate = metrics.at("truncation_rate").get<double>();
        report.meanLength = metrics.at("mean_length").get<double>();
        // Named copies: a range-for over items() of a temporary would dangle.
        Json const aggregates = root.value("aggregates", Json::object());
        Json const gates = root.value("gates", Json::array());
        for (auto const& item : aggregates.items())
        {
            report.aggregates.push_back({item.key(), NumberOrNaN(item.value())});
        }
        report.gates = VerdictsFromJson(gates);
        report.goal = VerdictsFromJson(root.value("goal", Json::array()));
    }
    catch (Json::exception const& error)
    {
        errors.Fail("", std::string("not an evaluation report: ") + error.what());
    }
    return report;
}
} // namespace

EvaluationReport ReportFromJson(std::string_view text, std::string_view sourceName)
{
    ErrorReporter const errors(sourceName);
    return ReadReportSummary(schema::detail::ParseRootObject(text, errors), errors);
}

std::filesystem::path EvaluationReportPath(std::filesystem::path const& checkpoints, std::string const& suite)
{
    if (!schema::detail::IsIdentifier(suite)) return {};
    return checkpoints / ("evaluation_" + suite + ".json");
}

SavedEvaluation ReadSavedEvaluation(std::filesystem::path const& path)
{
    ErrorReporter const errors(path.string());
    auto const root = schema::detail::ParseRootObject(schema::detail::ReadTextFile(path), errors);
    SavedEvaluation result;
    result.report = ReadReportSummary(root, errors);
    if (!root.contains("episodes") || !root.at("episodes").is_array() || root.at("episodes").empty())
        errors.Fail("episodes", "expected at least one episode in a completed evaluation");
    result.episodeCount = root.at("episodes").size();
    if (root.contains("seeds") && root.contains("episodes_per_seed"))
    {
        auto const& repeats = root.at("episodes_per_seed");
        if (!root.at("seeds").is_array() || !repeats.is_number_unsigned() || repeats == 0)
            errors.Fail("seeds", "invalid evaluation schedule");
        auto const perSeed = repeats.get<std::uint64_t>();
        if (result.episodeCount % perSeed != 0 || result.episodeCount / perSeed != root.at("seeds").size())
            errors.Fail("episodes", "incomplete campaign for the recorded seed schedule");
    }
    if (root.contains("policy_role") && !root.at("policy_role").is_null())
    {
        if (!root.at("policy_role").is_string()) errors.Fail("policy_role", "expected an actor role");
        result.policyRole = root.at("policy_role").get<std::string>();
    }
    if (root.contains("update") && !root.at("update").is_null())
    {
        if (!root.at("update").is_number_unsigned()) errors.Fail("update", "expected a nonnegative update");
        result.update = root.at("update").get<std::uint64_t>();
    }
    return result;
}

bool RanksAbove(EvaluationReport const& candidate, EvaluationReport const& incumbent,
    std::vector<RankKey> const& ranking)
{
    for (RankKey const& key : ranking)
    {
        double const ours = candidate.Metric(key.metric);
        double const theirs = incumbent.Metric(key.metric);
        bool const oursKnown = std::isfinite(ours);
        bool const theirsKnown = std::isfinite(theirs);
        if (oursKnown != theirsKnown)
        {
            return oursKnown;
        }
        if (!oursKnown || ours == theirs)
        {
            continue;
        }
        if (key.higher)
        {
            return ours > theirs;
        }
        return ours < theirs;
    }
    return false;
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
    report.aggregates.clear();
    for (std::size_t index = 0; index < suite.metrics.size(); ++index)
    {
        schema::EpisodeMetric const& metric = suite.metrics[index];
        report.aggregates.push_back({AggregateName(metric), Aggregate(report.episodes, index, metric.kind)});
    }
    report.gates = Verdicts(suite.gates, report);
    report.goal = Verdicts(suite.goal, report);
}

} // namespace ppokit::eval
