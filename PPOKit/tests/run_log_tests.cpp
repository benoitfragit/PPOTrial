/**
 * @file PPOKit/tests/run_log_tests.cpp
 * @brief Reward ledger (per-channel step, episode and cumulative means, weight changes,
 *        several streams) and run journal (write, read back, a run still writing, errors).
 */
#include "ppokit/runtime/reward_ledger.hpp"
#include "ppokit/runtime/run_log.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

using namespace ppokit::runtime;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

bool Near(double a, double b)
{
    return std::abs(a - b) <= 1e-12;
}

/** @brief Two channels, two streams (agents), weights changed between two windows. */
void Ledger()
{
    RewardLedger ledger({1.0, 0.5}, 2);
    double const step[2] = {2.0, -4.0};
    ledger.Record(0, step); // weighted: 2, -2
    ledger.Record(1, step);
    ledger.EndEpisode(0);   // episode of stream 0: 2, -2
    ledger.Record(1, step); // stream 1 continues: 4, -4 so far
    RewardSummary const first = ledger.Summarize();
    Check(first.steps == 3 && first.episodes == 1, "ledger: steps and completed episodes of the window");
    Check(Near(first.channels[0].stepMean, 2.0) && Near(first.channels[1].stepMean, -4.0), "ledger: raw step means");
    Check(Near(first.channels[1].weightedStepMean, -2.0), "ledger: weighted step means");
    Check(Near(first.channels[0].episodeMean, 2.0) && Near(first.channels[1].episodeMean, -2.0)
            && Near(first.episodeReturnMean, 0.0),
        "ledger: episode means count completed episodes only");
    Check(Near(first.channels[0].cumulative, 6.0) && Near(first.channels[1].cumulative, -6.0),
        "ledger: cumulative totals since the start");

    ledger.SetWeights({1.0, 0.0}); // the second channel is switched off from now on
    ledger.Record(1, step);
    ledger.EndEpisode(1); // 6 and -4 (the last step weighs 0 on channel 1)
    RewardSummary const second = ledger.Summarize();
    Check(second.steps == 1 && second.episodes == 1, "ledger: a new window after Summarize");
    Check(Near(second.channels[0].episodeMean, 6.0) && Near(second.channels[1].episodeMean, -4.0),
        "ledger: a running episode carries over to the next window");
    Check(Near(second.channels[1].weightedStepMean, 0.0) && Near(second.channels[1].cumulative, -6.0),
        "ledger: a step is weighted with the weights in force when recorded");

    bool refused = false;
    try
    {
        ledger.SetWeights({1.0});
    }
    catch (std::invalid_argument const&)
    {
        refused = true;
    }
    Check(refused, "ledger: one weight per channel");
    refused = false;
    try
    {
        ledger.Record(2, step);
    }
    catch (std::out_of_range const&)
    {
        refused = true;
    }
    Check(refused, "ledger: unknown stream refused");
}

std::string ReadText(std::filesystem::path const& path)
{
    std::ifstream file(path, std::ios::binary);
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

void JournalRoundTrip()
{
    std::filesystem::path const path = std::filesystem::temp_directory_path() / "ppokit_run_log_tests.jsonl";
    RunHeader header;
    header.task = "corridor";
    header.channels = {{"progress", "Progress to the goal", 1.0}, {"time", "Time penalty", 0.1}};
    ppokit::core::PpoSettings settings;
    header.settings = PpoSettingsValues(settings);
    {
        RunLogWriter writer(path, header);
        RewardLedger ledger({1.0, 0.1}, 1);
        double const step[2] = {1.0, -1.0};
        for (std::uint64_t update = 1; update <= 3; ++update)
        {
            ledger.Record(0, step);
            ledger.EndEpisode(0);
            UpdateRecord record;
            record.update = update;
            record.elapsedSeconds = 0.5 * static_cast<double>(update);
            ppokit::core::UpdateStatistics statistics;
            statistics.policyLoss = -0.01 * static_cast<double>(update);
            statistics.explainedVariance = std::nan("");
            record.ppo = PpoMetrics(statistics);
            record.rewards = ledger.Summarize();
            writer.WriteUpdate(record);
            if (update == 2)
            {
                writer.WriteWeights({3, {1.0, 0.1}, {1.0, 0.2}});
                EvaluationRecord campaign;
                campaign.update = 2;
                campaign.suite = "validation";
                campaign.episodes = 20;
                campaign.passed = false;
                campaign.promoted = false;
                campaign.metrics = {{"mean_return", 1.5}, {"stopped.rate", 0.85}, {"stop_m.mean", std::nan("")}};
                campaign.gates = {{"stopped.rate", ">=", 0.92, 0.85, false}};
                writer.WriteEvaluation(campaign);
            }
        }
    }
    RunLog const log = LoadRunLog(path);
    Check(log.header.task == "corridor" && log.header.channels.size() == 2
            && log.header.channels[1].label == "Time penalty",
        "journal: the header reads back");
    Check(log.header.settings.front().first == "gamma" && Near(log.header.settings.front().second, 0.99),
        "journal: hyper-parameters in stable order");
    Check(log.updates.size() == 3 && log.updates[2].update == 3 && Near(log.updates[1].elapsedSeconds, 1.0),
        "journal: one record per update");
    Check(Near(log.updates[2].rewards.channels[0].cumulative, 3.0)
            && Near(log.updates[2].rewards.channels[1].episodeMean, -0.1),
        "journal: per-channel rewards read back");
    Check(log.updates[0].ppo[0].first == "policy_loss" && Near(log.updates[0].ppo[0].second, -0.01),
        "journal: PPO statistics read back");
    Check(std::isnan(log.updates[0].ppo[5].second), "journal: a NaN statistic survives as NaN (null in JSON)");
    Check(log.weightChanges.size() == 1 && log.weightChanges[0].fromUpdate == 3
            && Near(log.weightChanges[0].weights[1], 0.2),
        "journal: weight changes read back");
    Check(log.evaluations.size() == 1 && log.evaluations[0].suite == "validation" && log.evaluations[0].episodes == 20
            && !log.evaluations[0].passed && log.evaluations[0].metrics[1].first == "stopped.rate"
            && Near(log.evaluations[0].metrics[1].second, 0.85) && std::isnan(log.evaluations[0].metrics[2].second)
            && log.evaluations[0].gates.size() == 1 && log.evaluations[0].gates[0].comparison == ">="
            && Near(log.evaluations[0].gates[0].threshold, 0.92),
        "journal: an evaluation campaign reads back, NaN included");

    // A run still writing: the last event has no newline yet and is ignored.
    std::string const text = ReadText(path);
    std::string const partial = text + R"({"event":"update","update":4,"elap)";
    Check(ParseRunLog(partial, "metrics.jsonl").updates.size() == 3, "journal: an incomplete last line is ignored");

    // A broken line in the middle is reported with its number.
    std::string broken = text;
    broken.insert(text.find('\n') + 1, "{not json}\n");
    std::string message;
    try
    {
        ParseRunLog(broken, "metrics.jsonl");
    }
    catch (ppokit::schema::SchemaError const& error)
    {
        message = error.what();
    }
    Check(message.find("metrics.jsonl: line 2:") == 0, "journal: a malformed line is named, got: " + message);

    message.clear();
    try
    {
        ParseRunLog(text.substr(text.find('\n') + 1), "metrics.jsonl");
    }
    catch (ppokit::schema::SchemaError const& error)
    {
        message = error.what();
    }
    Check(message.find("must begin with a start event") != std::string::npos, "journal: the start event comes first");
    std::filesystem::remove(path);
}

/** @brief A task that reports weighted values: raw values are recovered, unknown when the weight is 0. */
void WeightedRecording()
{
    RewardLedger ledger({2.0, 0.0}, 1);
    double const weighted[2] = {4.0, 0.0};
    ledger.RecordWeighted(0, weighted);
    ledger.EndEpisode(0);
    RewardSummary const summary = ledger.Summarize();
    Check(Near(summary.channels[0].stepMean, 2.0) && Near(summary.channels[0].weightedStepMean, 4.0),
        "ledger: raw value recovered from a weighted one");
    Check(std::isnan(summary.channels[1].stepMean) && Near(summary.channels[1].weightedStepMean, 0.0),
        "ledger: a zero weight leaves the raw value unknown (a gap), not zero");
}

/** @brief A task can bound its scalar reward after weighting. Carry each learner return across updates. */
void BoundedReturns()
{
    RewardLedger ledger({-2.0, 0.0}, 2);
    double const penalty[] = {-4.0, 0.0};
    double const bonus[] = {6.0, 0.0};
    ledger.RecordWeighted(0, penalty, -0.5);
    ledger.RecordWeighted(1, bonus, 0.25);
    ledger.EndEpisode(1);
    auto const first = ledger.Summarize();
    Check(first.episodes == 1 && Near(first.episodeReturnMean, 0.25) && Near(first.channels[0].episodeMean, 6.0),
        "the learner return and weighted contribution of the completed stream are distinct");

    ledger.RecordWeighted(0, bonus, 0.25);
    ledger.RecordWeighted(1, penalty, -0.5);
    ledger.EndEpisode(0);
    ledger.EndEpisode(1);
    auto const second = ledger.Summarize();
    Check(second.episodes == 2 && Near(second.episodeReturnMean, -0.375)
            && Near(second.channels[0].episodeMean, -1.0) && std::isnan(second.channels[1].stepMean),
        "a partial episode spans updates and a finished stream restarts at zero");
}

/** @brief A resumed run keeps its history up to the checkpoint and records new weights. */
void ResumedJournal()
{
    std::filesystem::path const path = std::filesystem::temp_directory_path() / "ppokit_run_log_resume.jsonl";
    RunHeader header;
    header.task = "flight";
    header.channels = {{"progress", "Progress", 1.0}, {"crash", "Crash", -10.0}};
    RewardLedger ledger({1.0, -10.0}, 1);
    double const step[2] = {1.0, 1.0};
    auto const writeUpdates = [&](RunLogWriter& writer, std::uint64_t first, std::uint64_t last) {
        for (std::uint64_t update = first; update <= last; ++update)
        {
            ledger.Record(0, step);
            UpdateRecord record;
            record.update = update;
            // Each session counts its own time: 10 s per update since the session started.
            record.elapsedSeconds = 10.0 * static_cast<double>(update - first + 1);
            record.rewards = ledger.Summarize();
            writer.WriteUpdate(record);
        }
    };
    {
        RunLogWriter writer(path, header);
        writeUpdates(writer, 1, 5);
        writer.WriteWeights({4, {1.0, -10.0}, {1.0, -20.0}});
    }
    // The resumed process has a fresh ledger. Events after its checkpoint belong to the discarded future.
    ledger = RewardLedger({1.0, -5.0}, 1);
    header.channels[1].weight = -5.0;
    {
        RunLogWriter writer = RunLogWriter::Resume(path, header, 3, &ledger);
        writeUpdates(writer, 4, 4);
    }
    RunLog const resumed = LoadRunLog(path);
    Check(resumed.updates.size() == 4 && resumed.updates[3].update == 4 && resumed.updates[2].update == 3,
        "resume: history up to the checkpoint, then the new updates");
    Check(Near(resumed.updates[2].elapsedSeconds, 30.0) && Near(resumed.updates[3].elapsedSeconds, 40.0),
        "resume: the training time continues from the kept history");
    auto const& totals = resumed.updates.back().rewards.channels;
    Check(Near(totals[0].cumulative, 4.0) && Near(totals[1].cumulative, -35.0),
        "resume: the exact checkpoint supplies cumulative rewards before applying the new weights");
    Check(resumed.weightChanges.size() == 2 && resumed.weightChanges[0].fromUpdate == 4
            && Near(resumed.weightChanges[1].weights[1], -5.0) && Near(resumed.weightChanges[1].previous[1], -20.0),
        "resume: the weights of the resumed run are recorded when they differ");

    {
        // A session still open on the journal while another resumes it at its last update
        // (the Windows case: nothing to drop, so the file is never replaced).
        RunLogWriter first = RunLogWriter::Resume(path, header, 4);
        writeUpdates(first, 5, 5);
        RunLogWriter second = RunLogWriter::Resume(path, header, 5);
        writeUpdates(second, 6, 6);
    }
    Check(LoadRunLog(path).updates.size() == 6 && LoadRunLog(path).updates[5].update == 6,
        "resume: a journal with nothing to drop is appended to, not replaced");

    // A checkpoint ahead of its recorded history has unknown cumulative rewards.
    ledger = RewardLedger({1.0, -5.0}, 1);
    {
        auto writer = RunLogWriter::Resume(path, header, 8, &ledger);
        writeUpdates(writer, 9, 9);
    }
    auto const incomplete = LoadRunLog(path).updates.back().rewards;
    Check(std::isnan(incomplete.channels[0].cumulative) && Near(incomplete.channels[0].weightedStepMean, 1.0),
        "resume: a missing checkpoint row leaves the total unknown while window statistics remain available");

    header.task = "another task";
    {
        RunLogWriter writer = RunLogWriter::Resume(path, header, 3);
    }
    Check(LoadRunLog(path).updates.empty() && LoadRunLog(path).header.task == "another task",
        "resume: another task starts a new journal");
    std::filesystem::remove(path);
}

} // namespace

int main()
{
    try
    {
        Ledger();
        WeightedRecording();
        BoundedReturns();
        JournalRoundTrip();
        ResumedJournal();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: unexpected exception: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " run log check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit run log tests passed\n";
    return 0;
}
