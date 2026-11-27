/**
 * @file PPOKit/tests/curriculum_tests.cpp
 * @brief Scenario parameters, train.json curriculum, sampling, promotion windows,
 *        rehearsal, stage persistence and the generated typed Scenario.
 */
#include "ppokit/runtime/curriculum.hpp"
#include "ppokit/schema/task_schema.hpp"
#include "ppokit/schema/train_schema.hpp"
#include "sample_task_layout.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

namespace {

using ppokit::runtime::Curriculum;
using ppokit::runtime::ScenarioDraw;
using ppokit::schema::ParseTrainingPlan;
using ppokit::schema::SchemaError;
using ppokit::schema::TaskSchema;

int failures = 0;

void Check(bool condition, char const* description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

TaskSchema SampleTask()
{
    return ppokit::schema::LoadTaskSchema(PPOKIT_SAMPLE_TASK);
}

Curriculum SampleCurriculum()
{
    return Curriculum(*ppokit::schema::LoadTrainingPlan(PPOKIT_SAMPLE_TRAIN, SampleTask()).curriculum);
}

/** @brief Curriculum file with one stage whose "sample" object is `sample`. */
std::string OneStage(std::string const& sample)
{
    return R"({"schema": "ppokit.train/1", "curriculum": {"stages": [{"name": "s", "sample": )" + sample + "}]}}";
}

void ExpectError(std::string const& text, std::string const& expected, char const* description)
{
    try
    {
        ParseTrainingPlan(text, "train.json", SampleTask());
    }
    catch (SchemaError const& error)
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

void ValidatesFiles()
{
    auto const task = SampleTask();
    Check(task.scenario.size() == 4 && task.scenario[1].values.size() == 2, "scenario parameters read in order");
    Check(task.scenario[1].defaultValue == 1.0, "enum default stored as its index");
    ExpectError(OneStage(R"({"opponents": 1, "altitude": 3})"), "sample.altitude: not a scenario parameter",
        "unknown parameter");
    ExpectError(OneStage(R"({"opponents": 1, "start": "flying"})"), "'flying' is not a value of this enum",
        "unknown enum value");
    ExpectError(OneStage(R"({"opponents": 1, "start": {"uniform": [0, 1]}})"), "use \"choice\"",
        "uniform on an enum");
    ExpectError(OneStage(R"({"wind_kt": 3})"), "sample.opponents: missing: the stage must sample it",
        "parameter without default must be sampled");
    ExpectError(OneStage(R"({"opponents": 11})"), "outside the range of 'opponents'", "value outside the range");
    ExpectError(OneStage(R"({"opponents": 1.5})"), "expected an integer", "fractional int");
    ExpectError(R"({"schema": "ppokit.train/1", "curriculum": {"promote_at": 1.5, "stages": []}})",
        "curriculum.promote_at: expected a value between 0 and 1", "rate above one");
}

void SamplesWithinRules()
{
    Curriculum curriculum = SampleCurriculum();
    std::mt19937 rng(3);
    std::vector<double> values;
    curriculum.Restore(1);
    std::size_t airborne = 0;
    std::size_t rehearsals = 0;
    constexpr int Draws = 4000;
    for (int draw = 0; draw < Draws; ++draw)
    {
        auto const result = curriculum.Sample(rng, values);
        if (result.rehearsal)
        {
            ++rehearsals;
            Check(result.stage == 0 && values[3] == 0.0 && values[1] == 0.0, "rehearsal replays the calm stage");
            continue;
        }
        Check(values[3] >= 1 && values[3] <= 3 && std::floor(values[3]) == values[3], "integral uniform draw");
        Check(values[0] >= 5 && values[0] <= 15, "real uniform draw");
        Check(values[2] == 0.0, "unsampled parameter takes its default");
        airborne += values[1] == 1.0 ? 1 : 0;
    }
    double const rehearsalShare = static_cast<double>(rehearsals) / Draws;
    double const airborneShare = static_cast<double>(airborne) / static_cast<double>(Draws - rehearsals);
    Check(std::abs(rehearsalShare - 0.2) < 0.03, "rehearsal share close to 0.2");
    Check(std::abs(airborneShare - 0.75) < 0.03, "choice weights 1:3");
    curriculum.Restore(2);
    for (int draw = 0; draw < 500; ++draw)
    {
        if (!curriculum.Sample(rng, values).rehearsal)
        {
            Check(values[3] >= 0 && values[3] <= 10 && std::floor(values[3]) == values[3],
                "normal draws are clamped to the range and rounded for an int");
        }
    }
}

void PromotesByWindow()
{
    Curriculum curriculum = SampleCurriculum();
    ScenarioDraw const current{0, false};
    Check(!curriculum.Record(current, true) && !curriculum.Record(current, false) && !curriculum.Record(current, true),
        "no decision before the window is complete");
    Check(curriculum.Record(current, true) && curriculum.StageName() == "windy", "3/4 >= 0.75 promotes");
    Check(curriculum.WindowEpisodes() == 0, "a new window starts after a promotion");
    Check(!curriculum.Record(current, true), "episodes of the previous stage no longer count");
    ScenarioDraw const windy{1, false};
    for (int episode = 0; episode < 4; ++episode)
    {
        curriculum.Record(windy, episode == 0);
    }
    Check(curriculum.StageName() == "windy" && curriculum.WindowEpisodes() == 0,
        "a failed window starts again empty");
    Check(!curriculum.Record(ScenarioDraw{1, true}, true) && curriculum.WindowEpisodes() == 0,
        "rehearsal episodes never count");
    curriculum.Restore(2);
    for (int episode = 0; episode < 20; ++episode)
    {
        curriculum.Record(ScenarioDraw{2, false}, true);
    }
    Check(curriculum.AtFinalStage() && curriculum.Stage() == 2, "the final stage is never left");
}

void PersistsStageByName()
{
    auto const path = std::filesystem::temp_directory_path() / "ppokit_curriculum_stage.txt";
    Curriculum saved = SampleCurriculum();
    saved.Restore(1);
    ppokit::runtime::SaveCurriculumStage(path, saved);
    Curriculum restored = SampleCurriculum();
    Check(ppokit::runtime::LoadCurriculumStage(path, restored) && restored.StageName() == "windy",
        "stage restored by name");
    Curriculum partial = SampleCurriculum();
    ScenarioDraw const calm{0, false};
    partial.Record(calm, true);
    partial.Record(calm, false);
    partial.Record(calm, true);
    ppokit::runtime::SaveCurriculumStage(path, partial);
    Curriculum continued = SampleCurriculum();
    Check(ppokit::runtime::LoadCurriculumStage(path, continued)
            && continued.WindowEpisodes() == 3 && continued.WindowSuccesses() == 2
            && continued.Record(calm, true) && continued.StageName() == "windy",
        "an interrupted promotion window continues exactly after resume");
    {
        std::ofstream stream(path, std::ios::trunc);
        stream << "windy\n";
    }
    Curriculum legacy = SampleCurriculum();
    Check(ppokit::runtime::LoadCurriculumStage(path, legacy) && legacy.StageName() == "windy"
            && legacy.WindowEpisodes() == 0,
        "legacy stage-only curriculum state remains readable");
    {
        std::ofstream stream(path, std::ios::trunc);
        stream << "unknown\n";
    }
    bool rejected = false;
    try
    {
        ppokit::runtime::LoadCurriculumStage(path, restored);
    }
    catch (std::runtime_error const&)
    {
        rejected = true;
    }
    Check(rejected, "a stage missing from train.json is reported");
    std::filesystem::remove(path);
    Check(!ppokit::runtime::LoadCurriculumStage(path, restored), "no saved stage: start from the first one");
}

using Scenario = ppokit::tests::sample::TaskLayout::Scenario;
static_assert(Scenario::Count == 4);
static_assert(static_cast<std::size_t>(Scenario::Start::Airborne) == 1);

void GeneratedScenarioIsTyped()
{
    std::vector<double> const values{12.5, 1.0, 1.0, 3.0};
    auto const scenario = Scenario::From(values);
    Check(scenario.windKt == 12.5 && scenario.start == Scenario::Start::Airborne && scenario.gusts
            && scenario.opponents == 3,
        "typed fields from curriculum values");
    Check(std::string(Scenario::Names[0]) == "wind_kt", "parameter names");
}

} // namespace

int main()
{
    try
    {
        ValidatesFiles();
        SamplesWithinRules();
        PromotesByWindow();
        PersistsStageByName();
        GeneratedScenarioIsTyped();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0)
    {
        std::cerr << failures << " curriculum check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit curriculum tests passed\n";
    return EXIT_SUCCESS;
}
