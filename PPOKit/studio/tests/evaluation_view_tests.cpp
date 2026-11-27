/**
 * @file PPOKit/studio/tests/evaluation_view_tests.cpp
 * @brief The evaluation suites editor, run headless: suites read with the task's metrics,
 *        added, edited, refused when ppokit-run would refuse them, saved (and renamed), and
 *        the campaigns of a journal and the BEST policy drawn.
 */
#include "evaluation_view.hpp"

#include "ppokit/artifacts/sha256.hpp"
#include "ppokit/eval/best_store.hpp"
#include "ppokit/eval/evaluation.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <imgui.h>
#include <imgui_internal.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

namespace fs = std::filesystem;
using ppokit::studio::EvaluationView;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

std::string Frame(EvaluationView& view, ppokit::runtime::RunLog const* journal)
{
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0.F, 0.F));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("evaluation", nullptr, ImGuiWindowFlags_NoDecoration);
    ImGui::LogToBuffer(2);
    view.Draw(journal);
    std::string const text = ImGui::GetCurrentContext()->LogBuffer.c_str();
    ImGui::LogFinish();
    ImGui::End();
    ImGui::Render();
    return text;
}

std::vector<ppokit::schema::EpisodeMetric> LandingMetrics()
{
    std::string const task = R"({ "schema": "ppokit.task/1", "name": "landing",
      "observation": { "state": { "fields": ["x"] } },
      "actions": { "move": { "head": "gaussian", "names": ["dx"] } },
      "rewards": { "R": {} },
      "metrics": { "stopped": { "kind": "flag" }, "stop_m": { "kind": "number", "unit": "m" } } })";
    return ppokit::schema::ParseTaskSchema(task, "task.json").metrics;
}

/** @brief A journal with one campaign of `suite`. */
ppokit::runtime::RunLog Journal(std::string const& suite)
{
    ppokit::runtime::RunLog log;
    ppokit::runtime::EvaluationRecord record;
    record.update = 10;
    record.suite = suite;
    record.episodes = 20;
    record.passed = true;
    record.promoted = true;
    record.metrics = {{"stopped.rate", 0.95}};
    record.gates = {{"stopped.rate", ">=", 0.92, 0.95, true}};
    log.evaluations.push_back(record);
    return log;
}

int Run()
{
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1400.F, 1000.F);
    io.DeltaTime = 1.F / 60.F;
    io.IniFilename = nullptr;
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    fs::path const root = fs::temp_directory_path() / "ppokit_evaluation_view";
    std::error_code ignored;
    fs::remove_all(root, ignored);
    fs::path const folder = root / "eval";
    fs::path const checkpoints = root / "checkpoints";
    auto const metrics = LandingMetrics();
    {
        fs::create_directories(root);
        ppokit::schema::PolicyReference reference;
        reference.role = "HISTORICAL_BEST";
        reference.suite = "validation";
        reference.sourceUpdate = 610;
        reference.file = root / "historical.ppokit";
        std::ofstream(reference.file) << "packaged actor";
        reference.sha256 = ppokit::artifacts::Sha256OfFile(reference.file);
        EvaluationView view(folder, checkpoints, metrics, {reference});
        auto const empty = Frame(view, nullptr);
        Check(view.Document().Suites().empty() && !view.Dirty(), "a project without eval/ has no suite");
        Check(empty.find("select Add") != std::string::npos,
            "the empty evaluation view explains how to create a suite");

        std::size_t const added = view.Document().Add();
        Frame(view, nullptr);
        Check(view.Dirty() && view.Document().Problem(added).empty(), "a new suite is valid as it is");
        auto const& names = view.Document().MetricNames();
        Check(names.back() == "stop_m.mean" && names[names.size() - 2] == "stopped.rate",
            "the gates may use the task's episode metrics");

        ppokit::editor::EditableSuite& suite = view.Document().Suite(added);
        suite.name = "validation";
        suite.firstSeed = 4342;
        suite.seedCount = 20;
        suite.gates = {{"stopped.rate", true, 0.92}};
        suite.duringTraining = true;
        suite.everyUpdates = 10;
        suite.goal = {{"stop_m.mean", false, 450.0}};
        suite.ranking = {{"stopped.rate", true}, {"stop_m.mean", false}};
        Frame(view, nullptr);
        Check(view.Document().Problem(added).empty(), "the T-6 validation is a valid suite");

        suite.gates.clear();
        Check(!view.Save() && !fs::exists(folder / "validation.json"), "a suite without gate is not saved");
        suite.gates = {{"stopped.rate", true, 0.92}};
        suite.name = "not a name";
        Check(!view.Document().Problem(added).empty() && !view.Save(), "a name that is not an identifier is refused");
        suite.name = "validation";
        Check(view.Save() && !view.Dirty() && fs::exists(folder / "validation.json"), "the suite is saved");

        auto const parsed = ppokit::eval::LoadEvaluationSuite(folder / "validation.json", metrics);
        Check(parsed.seeds.size() == 20 && parsed.seeds.front() == 4342 && parsed.everyUpdates == 10
                && parsed.best.size() == 2 && !parsed.best[1].higher && parsed.gates[0].threshold == 0.92
                && parsed.goal.size() == 1 && parsed.goal[0].comparison == ppokit::eval::Comparison::LessOrEqual,
            "ppokit-run reads the saved suite as edited");

        auto const packaged = Frame(view, nullptr);
        Check(packaged.find("Packaged BEST") != std::string::npos && packaged.find("update 610") != std::string::npos,
            "a suite with a packaged historical BEST shows its update before any promotion");
        auto const written = fs::last_write_time(reference.file);
        std::ofstream(reference.file) << "changed";
        fs::last_write_time(reference.file, written + std::chrono::seconds(3));
        Check(Frame(view, nullptr).find("SHA-256 differs") != std::string::npos,
            "a packaged BEST whose bytes changed is reported");
        std::ofstream(reference.file) << "packaged actor";
        fs::last_write_time(reference.file, written + std::chrono::seconds(6));

        // Results of a standalone evaluation and a separate training campaign with its own BEST.
        fs::create_directories(checkpoints);
        std::ofstream(checkpoints / "policy.ppokit") << "policy";
        ppokit::eval::EvaluationReport report;
        report.suite = "validation";
        report.policySha256 = ppokit::artifacts::Sha256OfFile(checkpoints / "policy.ppokit");
        report.successRate = 1.0;
        report.aggregates = {{"stopped.rate", 0.95}, {"stop_m.mean", 410.0}};
        report.gates = {{parsed.gates[0], 0.95, true}};
        report.episodes = {{4342, 4.0, 12, true, {1, 410}}, {4343, 3.0, 9, true, {1, 396}}};
        auto standalone = nlohmann::json::parse(report.ToJson());
        standalone["update"] = 610;
        standalone["policy_role"] = "HISTORICAL_BEST";
        auto const resultFile = checkpoints / "evaluation_validation.json";
        std::ofstream(resultFile) << standalone.dump();
        auto const rendered = Frame(view, nullptr);
        Check(rendered.find("Last completed evaluation  passed") != std::string::npos
                && rendered.find("HISTORICAL_BEST") != std::string::npos && rendered.find("update 610") != std::string::npos
                && rendered.find("2 episodes") != std::string::npos && rendered.find("stopped.rate") != std::string::npos,
            "a standalone evaluation displays its actor, update, episodes and metrics without a training journal");
        fs::remove(resultFile);
        Check(Frame(view, nullptr).find("Last completed evaluation") == std::string::npos,
            "removing the report clears its old result");
        standalone["suite"] = "another_suite";
        std::ofstream(resultFile) << standalone.dump();
        Check(Frame(view, nullptr).find("another_suite") != std::string::npos, "a report filed under the wrong suite is identified");
        standalone["suite"] = "validation";
        standalone["seeds"] = {4342, 4343};
        standalone["episodes_per_seed"] = 2;
        std::ofstream(resultFile) << standalone.dump();
        view.RefreshResults();
        Check(Frame(view, nullptr).find("incomplete campaign") != std::string::npos,
            "a partial seed schedule cannot show a successful campaign");
        standalone.erase("seeds");
        standalone.erase("episodes_per_seed");
        standalone.erase("policy_role");
        standalone["update"] = nullptr;
        std::ofstream(resultFile) << standalone.dump();
        view.RefreshResults();
        Check(Frame(view, nullptr).find("Last completed evaluation  passed") != std::string::npos,
            "older standalone reports without actor attribution still display their results");
        fs::remove(resultFile);
        Check(ppokit::eval::OfferBest(checkpoints, parsed, 10, checkpoints / "policy.ppokit", report),
            "a passing campaign becomes the BEST");
        ppokit::runtime::RunLog const journal = Journal("validation");
        auto const promoted = Frame(view, &journal);
        Check(promoted.find("update 10") != std::string::npos && promoted.find("Packaged BEST") == std::string::npos,
            "a newly promoted local BEST takes precedence for its suite");
        if (auto const best = ppokit::eval::ReadBest(checkpoints, parsed.name))
        {
            auto const promotionTime = fs::last_write_time(best->policy);
            std::ofstream(best->policy) << "wrong weights";
            fs::last_write_time(best->policy, promotionTime + std::chrono::seconds(3));
            auto const damaged = Frame(view, &journal);
            Check(damaged.find("SHA-256 differs") != std::string::npos && damaged.find("Packaged BEST") == std::string::npos,
                "a damaged promoted actor is identified without silently selecting the archive");
        }

        std::size_t const copy = view.Document().Duplicate(0);
        Check(view.Document().Suites()[copy].name == "validation_copy", "a duplicate gets a free name");
        view.Document().Suite(copy).name = "landing";
        view.Document().Remove(0);
        Check(view.Save() && fs::exists(folder / "landing.json") && !fs::exists(folder / "validation.json"),
            "saving removes the files of the suites renamed or removed");
    }
    {
        EvaluationView reopened(folder, checkpoints, metrics);
        Check(reopened.Document().Suites().size() == 1 && reopened.Document().Suites()[0].name == "landing"
                && !reopened.Dirty(),
            "the folder reads back");
        std::ofstream(folder / "broken.json") << R"({ "schema": "ppokit.eval/1", "suite": "broken",
          "seeds": [1], "episodes": 1, "gates": { "landed.rate": ">= 1" } })";
        bool refused = false;
        try
        {
            EvaluationView broken(folder, checkpoints, metrics);
        }
        catch (std::exception const&)
        {
            refused = true;
        }
        Check(refused, "a suite on an unknown metric is reported when the project opens");
    }
    fs::remove_all(root, ignored);
    ImGui::DestroyContext();
    if (failures != 0)
    {
        std::cerr << failures << " evaluation view check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit studio evaluation view tests passed\n";
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
