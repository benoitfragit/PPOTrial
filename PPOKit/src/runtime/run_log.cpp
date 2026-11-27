/**
 * @file PPOKit/src/runtime/run_log.cpp
 * @brief Writing and reading of the run journal (ppokit.run/1, one JSON object per line).
 */
#include "ppokit/runtime/run_log.hpp"

#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace ppokit::runtime {
namespace {

using Json = nlohmann::ordered_json;

constexpr char const* RunLogSchemaVersion = "ppokit.run/1";

/** @brief JSON has no NaN or infinity: such values are written as null. */
Json Number(double value)
{
    return std::isfinite(value) ? Json(value) : Json(nullptr);
}

double ReadNumber(Json const& value)
{
    return value.is_number() ? value.get<double>() : std::numeric_limits<double>::quiet_NaN();
}

Json Values(NamedValues const& values)
{
    Json object = Json::object();
    for (auto const& [name, value] : values)
    {
        object[name] = Number(value);
    }
    return object;
}

NamedValues ReadValues(Json const& object)
{
    NamedValues values;
    for (auto const& [name, value] : object.items())
    {
        values.emplace_back(name, ReadNumber(value));
    }
    return values;
}

Json Numbers(std::vector<double> const& values)
{
    Json array = Json::array();
    for (double const value : values)
    {
        array.push_back(Number(value));
    }
    return array;
}

std::vector<double> ReadNumbers(Json const& array)
{
    std::vector<double> values;
    for (Json const& value : array)
    {
        values.push_back(ReadNumber(value));
    }
    return values;
}

/** @brief Gates or goal conditions of an `evaluation` event. */
std::vector<EvaluationGate> ReadConditions(Json const& array)
{
    std::vector<EvaluationGate> conditions;
    for (Json const& entry : array)
    {
        conditions.push_back({entry.at("metric").get<std::string>(), entry.at("comparison").get<std::string>(),
            ReadNumber(entry.at("threshold")), ReadNumber(entry.at("value")), entry.at("passed").get<bool>()});
    }
    return conditions;
}

Json Conditions(std::vector<EvaluationGate> const& conditions)
{
    Json array = Json::array();
    for (EvaluationGate const& gate : conditions)
    {
        array.push_back(Json{{"metric", gate.metric}, {"comparison", gate.comparison},
            {"threshold", Number(gate.threshold)}, {"value", Number(gate.value)}, {"passed", gate.passed}});
    }
    return array;
}

/** @brief Reads one event into the journal; throws nlohmann or std exceptions on bad content. */
void ReadEvent(Json const& event, RunLog& log, bool first)
{
    std::string const kind = event.at("event").get<std::string>();
    if (first != (kind == "start"))
    {
        throw std::invalid_argument(first ? "the journal must begin with a start event" : "a second start event");
    }
    if (kind == "start")
    {
        if (event.at("schema") != RunLogSchemaVersion)
        {
            throw std::invalid_argument(std::string("expected schema \"") + RunLogSchemaVersion + "\"");
        }
        log.header.task = event.at("task").get<std::string>();
        for (Json const& channel : event.at("channels"))
        {
            log.header.channels.push_back({channel.at("name").get<std::string>(),
                channel.at("label").get<std::string>(), ReadNumber(channel.at("weight"))});
        }
        log.header.settings = ReadValues(event.at("settings"));
        log.header.acceptsWeightRequests = event.value("weight_requests", false);
        return;
    }
    if (kind == "update")
    {
        UpdateRecord record;
        record.update = event.at("update").get<std::uint64_t>();
        record.elapsedSeconds = ReadNumber(event.at("elapsed_s"));
        record.ppo = ReadValues(event.at("ppo"));
        record.rewards.steps = event.at("steps").get<std::size_t>();
        record.rewards.episodes = event.at("episodes").get<std::size_t>();
        record.rewards.episodeReturnMean = ReadNumber(event.at("episode_return_mean"));
        Json const& rewards = event.at("rewards");
        for (RunChannel const& channel : log.header.channels)
        {
            Json const& entry = rewards.at(channel.name);
            record.rewards.channels.push_back({ReadNumber(entry.at("step_mean")),
                ReadNumber(entry.at("weighted_step_mean")), ReadNumber(entry.at("episode_mean")),
                ReadNumber(entry.at("cumulative"))});
        }
        log.updates.push_back(std::move(record));
        return;
    }
    if (kind == "weights")
    {
        WeightChange change;
        change.fromUpdate = event.at("from_update").get<std::uint64_t>();
        change.previous = ReadNumbers(event.at("previous"));
        change.weights = ReadNumbers(event.at("weights"));
        if (change.weights.size() != log.header.channels.size())
        {
            throw std::invalid_argument("one weight per reward channel is required");
        }
        log.weightChanges.push_back(std::move(change));
        return;
    }
    if (kind == "probes")
    {
        ProbeRecord record;
        record.update = event.at("update").get<std::uint64_t>();
        record.network = event.at("network").get<std::string>();
        for (Json const& entry : event.at("blocks"))
        {
            BlockProbe probe;
            probe.block = entry.at("block").get<std::string>();
            probe.type = entry.at("type").get<std::string>();
            probe.samples = entry.at("samples").get<std::size_t>();
            probe.mean = ReadNumber(entry.at("mean"));
            probe.deviation = ReadNumber(entry.at("std"));
            probe.minimum = ReadNumber(entry.at("min"));
            probe.maximum = ReadNumber(entry.at("max"));
            probe.deadShare = ReadNumber(entry.at("dead"));
            probe.saturatedShare = ReadNumber(entry.at("saturated"));
            probe.hasGradient = entry.contains("gradient_norm");
            probe.gradientNorm = probe.hasGradient ? ReadNumber(entry.at("gradient_norm")) : 0.0;
            record.blocks.push_back(std::move(probe));
        }
        log.probes.push_back(std::move(record));
        return;
    }
    if (kind == "evaluation")
    {
        EvaluationRecord record;
        record.update = event.at("update").get<std::uint64_t>();
        record.suite = event.at("suite").get<std::string>();
        record.episodes = event.at("episodes").get<std::size_t>();
        record.passed = event.at("passed").get<bool>();
        record.promoted = event.at("promoted").get<bool>();
        record.goalReached = event.value("goal_reached", record.passed);
        record.metrics = ReadValues(event.at("metrics"));
        record.gates = ReadConditions(event.at("gates"));
        record.goal = ReadConditions(event.value("goal", Json::array()));
        log.evaluations.push_back(std::move(record));
        return;
    }
    throw std::invalid_argument("unknown event '" + kind + "'");
}

} // namespace

NamedValues PpoMetrics(core::UpdateStatistics const& statistics)
{
    return {{"policy_loss", statistics.policyLoss}, {"value_loss", statistics.valueLoss},
        {"entropy", statistics.entropy}, {"approx_kl", statistics.approximateKl},
        {"clip_fraction", statistics.clipFraction}, {"explained_variance", statistics.explainedVariance},
        {"actor_kl_skips", static_cast<double>(statistics.actorKlSkips)}};
}

NamedValues PpoMetrics(TeamUpdateStatistics const& statistics)
{
    return {{"policy_loss", statistics.policyLoss}, {"value_loss", statistics.valueLoss},
        {"entropy", statistics.entropy}, {"approx_kl", statistics.approximateKl},
        {"clip_fraction", statistics.clipFraction}, {"explained_variance", statistics.explainedVariance},
        {"advantage_abs_mean", statistics.advantageAbsMean}, {"action_std_mean", statistics.actionStdMean},
        {"actor_kl_skips", static_cast<double>(statistics.actorKlSkips)},
        {"learnable_rows", static_cast<double>(statistics.learnableRows)},
        {"critic_rows", static_cast<double>(statistics.criticRows)}};
}

NamedValues PpoSettingsValues(core::PpoSettings const& settings)
{
    return {{"gamma", settings.gamma}, {"lambda", settings.lambda}, {"clip", settings.clip},
        {"actor_learning_rate", settings.actorLearningRate}, {"critic_learning_rate", settings.criticLearningRate},
        {"entropy_coefficient", settings.entropyCoefficient}, {"value_coefficient", settings.valueCoefficient},
        {"max_gradient_norm", settings.maxGradientNorm},
        {"epochs", static_cast<double>(settings.epochs)},
        {"minibatch", static_cast<double>(settings.minibatch)}, {"initial_log_std", settings.initialLogStd},
        {"minimum_log_std", settings.minimumLogStd}, {"maximum_log_std", settings.maximumLogStd},
        {"target_kl", settings.targetKl},
        {"seed", static_cast<double>(settings.seed)}};
}

NamedValues PpoSettingsValues(TeamPpoSettings const& settings)
{
    return {{"gamma", settings.gamma}, {"lambda", settings.lambda}, {"clip", settings.clip},
        {"entropy_coefficient", settings.entropyCoefficient}, {"epochs", static_cast<double>(settings.epochs)},
        {"minibatch", static_cast<double>(settings.minibatch)}};
}

RunLogWriter::RunLogWriter(std::filesystem::path path, RunHeader const& header) : path_(std::move(path))
{
    if (!std::ofstream(path_, std::ios::binary | std::ios::trunc))
    {
        throw std::runtime_error("PPOKit run journal: cannot write " + path_.string());
    }
    Json channels = Json::array();
    for (RunChannel const& channel : header.channels)
    {
        channels.push_back(Json{{"name", channel.name}, {"label", channel.label}, {"weight", Number(channel.weight)}});
    }
    Json event = Json::object();
    event["schema"] = RunLogSchemaVersion;
    event["event"] = "start";
    event["task"] = header.task;
    event["channels"] = std::move(channels);
    event["settings"] = Values(header.settings);
    if (header.acceptsWeightRequests)
    {
        event["weight_requests"] = true;
    }
    WriteLine(event.dump());
    channelNames_.reserve(header.channels.size());
    for (RunChannel const& channel : header.channels)
    {
        channelNames_.push_back(channel.name);
    }
}

RunLogWriter::RunLogWriter(std::filesystem::path path, std::vector<std::string> channelNames)
    : path_(std::move(path)), channelNames_(std::move(channelNames))
{
    if (!std::ofstream(path_, std::ios::binary | std::ios::app))
    {
        throw std::runtime_error("PPOKit run journal: cannot append to " + path_.string());
    }
}

namespace {

bool SameChannels(RunHeader const& a, RunHeader const& b)
{
    if (a.task != b.task || a.channels.size() != b.channels.size())
    {
        return false;
    }
    for (std::size_t channel = 0; channel < a.channels.size(); ++channel)
    {
        if (a.channels[channel].name != b.channels[channel].name)
        {
            return false;
        }
    }
    return true;
}

/** @brief Weights in force after update `update` of a journal. */
std::vector<double> WeightsAfter(RunLog const& log, std::uint64_t update)
{
    std::vector<double> weights;
    for (RunChannel const& channel : log.header.channels)
    {
        weights.push_back(channel.weight);
    }
    for (WeightChange const& change : log.weightChanges)
    {
        if (change.fromUpdate <= update + 1)
        {
            weights = change.weights;
        }
    }
    return weights;
}

/** @brief The lines of `text` describing updates up to `update`, start event first. */
std::string HistoryUpTo(std::string const& text, std::uint64_t update)
{
    std::string kept;
    std::size_t position = 0;
    while (position < text.size())
    {
        std::size_t const end = text.find('\n', position);
        if (end == std::string::npos)
        {
            break; // an unfinished event of the interrupted run
        }
        std::string const line = text.substr(position, end - position);
        position = end + 1;
        Json const event = Json::parse(line);
        std::string const kind = event.at("event").get<std::string>();
        bool const measured = kind == "update" || kind == "probes" || kind == "evaluation";
        bool const keep = kind == "start" || (measured && event.at("update").get<std::uint64_t>() <= update)
            || (kind == "weights" && event.at("from_update").get<std::uint64_t>() <= update + 1);
        if (keep)
        {
            kept += line + "\n";
        }
    }
    return kept;
}

} // namespace

RunLogWriter RunLogWriter::Resume(std::filesystem::path path, RunHeader const& header, std::uint64_t resumedUpdates,
    RewardLedger* ledger)
{
    std::optional<RunLog> previous;
    std::string text;
    try
    {
        previous = LoadRunLog(path);
        std::ifstream file(path, std::ios::binary);
        std::stringstream content;
        content << file.rdbuf();
        text = content.str();
    }
    catch (std::exception const&)
    {
        previous.reset(); // no journal, or not readable: start a new one
    }
    bool const sameHistory = previous && SameChannels(previous->header, header);
    if (ledger && resumedUpdates != 0)
    {
        RewardSummary const* checkpoint = nullptr;
        if (sameHistory)
        {
            for (auto const& update : previous->updates)
                if (update.update == resumedUpdates) checkpoint = &update.rewards;
        }
        ledger->RestoreCumulative(checkpoint);
    }
    if (!sameHistory)
    {
        return RunLogWriter(std::move(path), header);
    }
    std::string const history = HistoryUpTo(text, resumedUpdates);
    // Rewritten only when something is dropped: Windows cannot replace a file another
    // process (or a session not yet closed) has open.
    if (history != text)
    {
        artifacts::WriteFileAtomically(
            path, [&history](std::ostream& out) { out << history; }, artifacts::AtomicWriteOptions{},
            "PPOKit run journal");
    }
    std::vector<std::string> names;
    std::vector<double> weights;
    for (RunChannel const& channel : header.channels)
    {
        names.push_back(channel.name);
        weights.push_back(channel.weight);
    }
    RunLogWriter writer(std::move(path), std::move(names));
    for (UpdateRecord const& update : previous->updates)
    {
        if (update.update <= resumedUpdates && std::isfinite(update.elapsedSeconds))
        {
            writer.elapsedOffset_ = update.elapsedSeconds;
        }
    }
    std::vector<double> const inForce = WeightsAfter(*previous, resumedUpdates);
    if (inForce != weights)
    {
        writer.WriteWeights({resumedUpdates + 1, inForce, weights});
    }
    return writer;
}

void RunLogWriter::WriteUpdate(UpdateRecord const& record)
{
    if (record.rewards.channels.size() != channelNames_.size())
    {
        throw std::invalid_argument("PPOKit run journal: one reward summary per channel is required");
    }
    Json rewards = Json::object();
    for (std::size_t channel = 0; channel < channelNames_.size(); ++channel)
    {
        ChannelSummary const& summary = record.rewards.channels[channel];
        rewards[channelNames_[channel]] = Json{{"step_mean", Number(summary.stepMean)},
            {"weighted_step_mean", Number(summary.weightedStepMean)}, {"episode_mean", Number(summary.episodeMean)},
            {"cumulative", Number(summary.cumulative)}};
    }
    Json event = Json::object();
    event["event"] = "update";
    event["update"] = record.update;
    event["elapsed_s"] = Number(elapsedOffset_ + record.elapsedSeconds);
    event["steps"] = record.rewards.steps;
    event["episodes"] = record.rewards.episodes;
    event["episode_return_mean"] = Number(record.rewards.episodeReturnMean);
    event["ppo"] = Values(record.ppo);
    event["rewards"] = std::move(rewards);
    WriteLine(event.dump());
}

void RunLogWriter::WriteProbes(ProbeRecord const& record)
{
    Json blocks = Json::array();
    for (BlockProbe const& probe : record.blocks)
    {
        Json entry = Json{{"block", probe.block}, {"type", probe.type}, {"samples", probe.samples},
            {"mean", Number(probe.mean)}, {"std", Number(probe.deviation)}, {"min", Number(probe.minimum)},
            {"max", Number(probe.maximum)}, {"dead", Number(probe.deadShare)},
            {"saturated", Number(probe.saturatedShare)}};
        if (probe.hasGradient)
        {
            entry["gradient_norm"] = Number(probe.gradientNorm);
        }
        blocks.push_back(std::move(entry));
    }
    Json event = Json::object();
    event["event"] = "probes";
    event["update"] = record.update;
    event["network"] = record.network;
    event["blocks"] = std::move(blocks);
    WriteLine(event.dump());
}

void RunLogWriter::WriteEvaluation(EvaluationRecord const& record)
{
    Json event = Json::object();
    event["event"] = "evaluation";
    event["update"] = record.update;
    event["suite"] = record.suite;
    event["episodes"] = record.episodes;
    event["passed"] = record.passed;
    event["promoted"] = record.promoted;
    event["goal_reached"] = record.goalReached;
    event["metrics"] = Values(record.metrics);
    event["gates"] = Conditions(record.gates);
    event["goal"] = Conditions(record.goal);
    WriteLine(event.dump());
}

void RunLogWriter::WriteWeights(WeightChange const& change)
{
    Json event = Json::object();
    event["event"] = "weights";
    event["from_update"] = change.fromUpdate;
    event["previous"] = Numbers(change.previous);
    event["weights"] = Numbers(change.weights);
    WriteLine(event.dump());
}

void RunLogWriter::WriteLine(std::string const& line)
{
    // One write per event, and the file is closed between events: a reader never sees
    // half an event as complete, and nothing holds the file open while training runs.
    std::ofstream stream(path_, std::ios::binary | std::ios::app);
    stream << line << '\n';
    stream.close();
    if (!stream)
    {
        throw std::runtime_error("PPOKit run journal: write failed on " + path_.string());
    }
    if (sink_ != nullptr)
    {
        sink_->JournalLine(line);
    }
}

RunLog ParseRunLog(std::string_view text, std::string_view sourceName)
{
    RunLog log;
    std::size_t position = 0;
    std::size_t lineNumber = 0;
    while (position < text.size())
    {
        std::size_t const end = text.find('\n', position);
        bool const complete = end != std::string_view::npos;
        std::string_view const line = text.substr(position, complete ? end - position : std::string_view::npos);
        position = complete ? end + 1 : text.size();
        ++lineNumber;
        if (!complete)
        {
            break; // The run is still writing this event.
        }
        if (line.empty())
        {
            continue;
        }
        try
        {
            ReadEvent(Json::parse(line), log, lineNumber == 1);
        }
        catch (std::exception const& error)
        {
            throw schema::SchemaError(std::string(sourceName), "line " + std::to_string(lineNumber), error.what());
        }
    }
    if (lineNumber == 0 || log.header.task.empty())
    {
        throw schema::SchemaError(std::string(sourceName), "", "empty journal (no start event yet)");
    }
    return log;
}

void AppendRunLogLine(RunLog& log, std::string_view line)
{
    Json const event = Json::parse(line);
    if (event.value("event", std::string()) == "start")
    {
        RunLog restarted;
        ReadEvent(event, restarted, true);
        log = std::move(restarted);
        return;
    }
    RunLog next = log;
    ReadEvent(event, next, log.header.task.empty());
    log = std::move(next);
}

RunLog LoadRunLog(std::filesystem::path const& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
    {
        throw schema::SchemaError(path.filename().string(), "", "cannot open the journal");
    }
    std::stringstream text;
    text << file.rdbuf();
    return ParseRunLog(text.str(), path.filename().string());
}

} // namespace ppokit::runtime
