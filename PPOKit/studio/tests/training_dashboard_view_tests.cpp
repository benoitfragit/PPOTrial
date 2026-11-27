/**
 * @file PPOKit/studio/tests/training_dashboard_view_tests.cpp
 * @brief The training dashboard, run headless on a journal written like a live run: every
 *        tab and reward view, a growing journal followed live, weights changed while the
 *        run trains, a malformed append.
 */
#include "training_dashboard_view.hpp"

#include "ppokit/live/live_channel.hpp"
#include "ppokit/runtime/weight_request.hpp"

#include <imgui.h>
#include <implot.h>

#include <chrono>
#include <filesystem>
#include <thread>
#include <fstream>
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

void Frame(studio::TrainingDashboardView& view)
{
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0.F, 0.F));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("dashboard", nullptr, ImGuiWindowFlags_NoDecoration);
    view.Draw();
    ImGui::End();
    ImGui::Render();
}

void WriteUpdate(runtime::RunLogWriter& writer, runtime::RewardLedger& ledger, std::uint64_t update)
{
    double const step[2] = {1.0, -0.5};
    ledger.Record(0, step);
    if (update % 2 == 1)
    {
        ledger.EndEpisode(0);
    }
    runtime::UpdateRecord record;
    record.update = update;
    record.elapsedSeconds = static_cast<double>(update);
    runtime::TeamUpdateStatistics statistics;
    statistics.approximateKl = 0.001F * static_cast<float>(update);
    record.ppo = runtime::PpoMetrics(statistics);
    record.rewards = ledger.Summarize();
    writer.WriteUpdate(record);
}

/** @brief The checks; an exception is reported by main rather than aborting the process. */
int Run()
{
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1400.F, 900.F);
    io.DeltaTime = 1.F / 60.F;
    io.IniFilename = nullptr;
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    std::filesystem::path const path = std::filesystem::temp_directory_path() / "ppokit_dashboard_tests.jsonl";
    std::filesystem::remove(path);
    {
        studio::TrainingDashboardView waiting(path);
        Frame(waiting);
        Check(waiting.Log() == nullptr && !waiting.Error().empty(), "no journal yet: the view waits");
    }

    runtime::RunHeader header;
    header.task = "team_reach";
    header.channels = {{"distance", "Distance to the targets", 1.0}, {"effort", "Control effort", 0.1}};
    header.settings = runtime::PpoSettingsValues(runtime::TeamPpoSettings{});
    header.acceptsWeightRequests = true;
    runtime::RunLogWriter writer(path, header);
    runtime::RewardLedger ledger({1.0, 0.1}, 1);
    for (std::uint64_t update = 1; update <= 4; ++update)
    {
        WriteUpdate(writer, ledger, update);
    }
    writer.WriteWeights({5, {1.0, 0.1}, {1.0, 0.3}});

    studio::TrainingDashboardView view(path);
    Check(view.Log() != nullptr && view.Log()->updates.size() == 4, "the journal is loaded");
    for (auto const tab : {studio::TrainingDashboardView::Tab::Rewards, studio::TrainingDashboardView::Tab::Weights,
             studio::TrainingDashboardView::Tab::Ppo, studio::TrainingDashboardView::Tab::Probes,
             studio::TrainingDashboardView::Tab::Settings,
             studio::TrainingDashboardView::Tab::Rewards})
    {
        view.SelectTab(tab);
        Frame(view);
        Frame(view);
    }
    for (auto const mode : {editor::RewardView::EpisodeMean, editor::RewardView::Cumulative,
             editor::RewardView::StepMean, editor::RewardView::WeightedStepMean})
    {
        view.SetRewardView(mode);
        Frame(view);
    }

    // The run goes on: following picks up the new updates within half a second of frames.
    WriteUpdate(writer, ledger, 5);
    WriteUpdate(writer, ledger, 6);
    for (int frame = 0; frame < 31; ++frame)
    {
        Frame(view);
    }
    Check(view.Log()->updates.size() == 6 && view.Log()->weightChanges.size() == 1,
        "following a live run reloads the journal");

    // Probes appear in their tab, with their remarks, for every statistic.
    runtime::BlockProbe probe;
    probe.block = "trunk";
    probe.type = "mlp";
    probe.samples = 64;
    probe.deviation = 0.5;
    probe.deadShare = 0.8;
    probe.hasGradient = true;
    probe.gradientNorm = 0.01;
    writer.WriteProbes({4, "actor", {probe}});
    view.Refresh();
    view.SelectTab(studio::TrainingDashboardView::Tab::Probes);
    Frame(view);
    Check(view.Log()->probes.size() == 1, "the probes are loaded");
    Frame(view);

    // Weights changed while the run trains: the run answers at its next update.
    view.SelectTab(studio::TrainingDashboardView::Tab::Weights);
    Check(view.Weights().Current(1) == 0.3, "the weights in force include the journaled change");
    view.Weights().SetDraft(0, 2.0);
    Frame(view);
    Check(view.Weights().Send(), "the edited weight is sent");
    runtime::WeightRequestInbox inbox(path, {"distance", "effort"});
    auto const request = inbox.Take();
    Check(request && request->weights.size() == 1 && request->weights[0].first == 0, "the run receives it");
    writer.WriteWeights({7, {1.0, 0.3}, {2.0, 0.3}});
    inbox.Answer({request->id, true, "Applied from update 7."});
    for (int frame = 0; frame < 31; ++frame)
    {
        Frame(view);
    }
    Check(view.Weights().Status() == editor::LiveWeights::State::Applied && view.Weights().Current(0) == 2.0
            && !view.Weights().AnyEdited(),
        "the answer and the new weight in force are shown");
    std::error_code removed;
    std::filesystem::remove(runtime::WeightAnswerPath(path), removed);

    // A corrupted line: the last good journal stays on screen with the error.
    {
        std::ofstream append(path, std::ios::app);
        append << "{broken\n";
    }
    for (int frame = 0; frame < 31; ++frame)
    {
        Frame(view);
    }
    Check(view.Log()->updates.size() == 6 && view.Error().find("line 11") != std::string::npos,
        "a malformed journal keeps the last good one, got: " + view.Error());

    // A run with a live channel: the view attaches by itself and follows it without the file.
    std::filesystem::path const folder = std::filesystem::temp_directory_path() / "ppokit_dashboard_live";
    std::error_code cleared;
    std::filesystem::remove_all(folder, cleared);
    std::filesystem::create_directories(folder);
    {
        runtime::RunLogWriter liveWriter(folder / "metrics.jsonl", header);
        runtime::RewardLedger liveLedger({1.0, 0.1}, 1);
        WriteUpdate(liveWriter, liveLedger, 1);
        live::LiveServer server(folder, folder / "metrics.jsonl", 2);
        liveWriter.SetSink(&server);
        studio::TrainingDashboardView followed(folder / "metrics.jsonl");
        followed.SetFollow(false); // only the live channel can bring the next updates
        live::RunStatus status;
        status.windowRewards = {0.5, -0.1};
        status.windowSteps = 5;
        for (int frame = 0; frame < 120 && !followed.Live(); ++frame)
        {
            server.Poll(status);
            Frame(followed);
        }
        Check(followed.Live(), "the view attaches to the run's live channel");
        WriteUpdate(liveWriter, liveLedger, 2);
        for (int frame = 0; frame < 200 && followed.Log()->updates.size() < 2; ++frame)
        {
            server.Poll(status);
            Frame(followed);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        Check(followed.Log()->updates.size() == 2, "an update arrives live, without re-reading the file");
    }
    std::filesystem::remove_all(folder, cleared);

    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    if (failures != 0)
    {
        std::cerr << failures << " dashboard check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit studio dashboard tests passed\n";
    return 0;
}

} // namespace

int main()
{
    try
    {
        return Run();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
}
