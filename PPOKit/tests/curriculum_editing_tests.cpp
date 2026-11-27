/**
 * @file PPOKit/tests/curriculum_editing_tests.cpp
 * @brief Curriculum editing for the studio: JSON <-> stages without losing meaning, new
 *        stages that are valid as created, the training parser's messages, previews drawn
 *        by the curriculum engine, and the document's undo and clean state.
 */
#include "ppokit/editor/curriculum_editing.hpp"
#include "ppokit/editor/train_settings_document.hpp"
#include "ppokit/schema/train_schema.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {

using namespace ppokit;
using editor::SamplerKind;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

std::string ReadFile(char const* path)
{
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::size_t Parameter(schema::TaskSchema const& task, std::string const& name)
{
    for (std::size_t index = 0; index < task.scenario.size(); ++index)
    {
        if (task.scenario[index].name == name)
        {
            return index;
        }
    }
    throw std::runtime_error("no parameter " + name);
}

/** @brief Same stages, same draw rules, as training reads them. */
bool SamePlan(schema::CurriculumPlan const& a, schema::CurriculumPlan const& b)
{
    if (a.stages.size() != b.stages.size() || a.rehearsal != b.rehearsal)
    {
        return false;
    }
    for (std::size_t stage = 0; stage < a.stages.size(); ++stage)
    {
        auto const& x = a.stages[stage];
        auto const& y = b.stages[stage];
        if (x.name != y.name || x.window != y.window || x.promotionRate != y.promotionRate)
        {
            return false;
        }
        for (std::size_t index = 0; index < x.samplers.size(); ++index)
        {
            auto const& p = x.samplers[index];
            auto const& q = y.samplers[index];
            bool const same = p.kind == q.kind && p.value == q.value && p.low == q.low && p.high == q.high
                && p.mean == q.mean && p.deviation == q.deviation && p.choices.size() == q.choices.size();
            if (!same)
            {
                return false;
            }
        }
    }
    return true;
}

void RoundTrip(schema::TaskSchema const& task)
{
    std::string const original = ReadFile(PPOKIT_SAMPLE_TRAIN);
    auto const before = schema::ParseTrainingPlan(original, "train.json", task);
    auto const editable = editor::ReadCurriculum(nlohmann::ordered_json::parse(original).at("curriculum"), task);
    Check(editable.stages.size() == 3 && editable.stages[2].window == std::size_t{10}
            && editable.stages[1].samplers[Parameter(task, "start")].kind == SamplerKind::Choice,
        "the curriculum reads as editable stages");
    nlohmann::ordered_json rewritten = {{"schema", "ppokit.train/1"}};
    rewritten["curriculum"] = editor::WriteCurriculum(editable, task);
    auto const after = schema::ParseTrainingPlan(rewritten.dump(), "train.json", task);
    Check(SamePlan(*before.curriculum, *after.curriculum), "written back, it trains exactly the same");
    std::string const text = rewritten.dump();
    Check(text.find("\"start\":\"rolling\"") != std::string::npos && text.find("\"gusts\":{\"choice\":{\"true\"")
            != std::string::npos,
        "values are written as a user writes them: enum names, true/false");
}

void NewStagesAreValid(schema::TaskSchema const& task)
{
    editor::EditableCurriculum curriculum;
    curriculum.stages.push_back(editor::NewStage(task, "start"));
    auto const& samplers = curriculum.stages[0].samplers;
    Check(samplers[Parameter(task, "wind_kt")].kind == SamplerKind::TaskDefault
            && samplers[Parameter(task, "opponents")].kind == SamplerKind::Uniform
            && samplers[Parameter(task, "opponents")].high == 10.0,
        "a new stage uses the task defaults, and draws the whole range of a parameter without one");
    nlohmann::ordered_json file = {{"schema", "ppokit.train/1"}};
    file["curriculum"] = editor::WriteCurriculum(curriculum, task);
    Check(editor::CheckCurriculum(file.dump(), task).empty(), "a new stage is valid as created");

    curriculum.stages[0].name = "first stage";
    file["curriculum"] = editor::WriteCurriculum(curriculum, task);
    Check(editor::CheckCurriculum(file.dump(), task).find("curriculum.stages[0].name") != std::string::npos,
        "a problem is reported with the training parser's message and path");

    auto const choice = editor::SamplerFor(task.scenario[Parameter(task, "start")], SamplerKind::Choice);
    Check(choice.choices.size() == 2 && choice.choices[1].value == 1.0, "a choice starts with every enum value");
}

/** @brief Required discrete parameters use a valid choice rule when a stage is created. */
void RequiredDiscreteParameters(schema::TaskSchema task)
{
    task.scenario[Parameter(task, "start")].defaultValue.reset();
    task.scenario[Parameter(task, "gusts")].defaultValue.reset();
    editor::EditableCurriculum curriculum;
    curriculum.stages.push_back(editor::NewStage(task, "start"));
    auto const& stage = curriculum.stages.front();
    Check(stage.samplers[Parameter(task, "start")].kind == SamplerKind::Choice
            && stage.samplers[Parameter(task, "gusts")].kind == SamplerKind::Choice,
        "required enum and bool parameters are sampled by choice");
    nlohmann::ordered_json file = {{"schema", "ppokit.train/1"}};
    file["curriculum"] = editor::WriteCurriculum(curriculum, task);
    Check(editor::CheckCurriculum(file.dump(), task).empty(), "a stage without discrete defaults is valid");
}

/** @brief Opening a bad plan must not silently remove invalid curriculum settings. */
void InvalidCurriculumIsNotNormalized(schema::TaskSchema const& task)
{
    auto source = nlohmann::ordered_json::parse(ReadFile(PPOKIT_SAMPLE_TRAIN));
    source["curriculum"]["stages"][0]["sample"]["start"] = "not-a-valid-enum";
    auto document = editor::TrainSettingsDocument::Open(source.dump(), "train.json");
    auto const before = document.Text();
    bool refused = false;
    try
    {
        document.AttachTask(task);
    }
    catch (schema::SchemaError const& error)
    {
        refused = std::string(error.what()).find("sample.start") != std::string::npos;
    }
    Check(refused && document.Text() == before && !document.Dirty() && !document.Task(),
        "invalid source is refused before changing the document");
    source["curriculum"]["stages"][0]["sample"]["start"] = "rolling";
    source["curriculum"]["stages"][0]["sample"]["unknown_parameter"] = 1;
    refused = false;
    try
    {
        editor::ReadCurriculum(source.at("curriculum"), task);
    }
    catch (schema::SchemaError const& error)
    {
        refused = std::string(error.what()).find("unknown_parameter") != std::string::npos;
    }
    Check(refused, "unknown scenario parameter is not dropped by the editor");
}

/** @brief Invalid edits must not overwrite the last valid training settings. */
void InvalidSavePreservesFile(schema::TaskSchema const& task)
{
    auto const path = std::filesystem::temp_directory_path() / "ppokit_curriculum_invalid_save.json";
    auto const original = ReadFile(PPOKIT_SAMPLE_TRAIN);
    std::ofstream(path, std::ios::binary) << original;
    auto document = editor::TrainSettingsDocument::Load(path);
    document.AttachTask(task);
    auto curriculum = *document.Curriculum();
    curriculum.stages[0].name = "not valid";
    document.SetCurriculum(curriculum);
    bool refused = false;
    try
    {
        document.Save(path);
    }
    catch (schema::SchemaError const&)
    {
        refused = true;
    }
    Check(refused && document.Dirty() && ReadFile(path.string().c_str()) == original,
        "invalid save leaves the previous file byte-for-byte unchanged");
    document.Undo();
    document.Save(path);
    Check(!document.Dirty(), "corrected curriculum can be saved");
    std::filesystem::remove(path);
}

void Previews(schema::TaskSchema const& task)
{
    std::string const original = ReadFile(PPOKIT_SAMPLE_TRAIN);
    Check(editor::PreviewStage(original, task, 0, 0, 3).empty(), "zero preview draws produce no histogram");
    auto const calm = editor::PreviewStage(original, task, 0, 1000, 3);
    auto const& start = calm[Parameter(task, "start")];
    Check(start.labels == std::vector<std::string>{"rolling", "airborne"} && start.shares[0] == 1.0,
        "a fixed enum: every draw on its value");
    Check(calm[Parameter(task, "opponents")].shares[0] == 1.0, "a fixed number: one bar");
    auto const windy = editor::PreviewStage(original, task, 1, 4000, 3);
    double const airborne = windy[Parameter(task, "start")].shares[1];
    Check(std::abs(airborne - 0.75) < 0.03, "a 1:3 choice: about three draws in four airborne");
    double total = 0.0;
    for (double const share : windy[Parameter(task, "wind_kt")].shares)
    {
        total += share;
    }
    Check(std::abs(total - 1.0) < 1e-9, "every draw lands in a bar");
}

void Document(schema::TaskSchema const& task)
{
    auto document = editor::TrainSettingsDocument::Open(ReadFile(PPOKIT_SAMPLE_TRAIN), "train.json");
    document.AttachTask(task);
    Check(!document.Dirty() && document.Curriculum() && document.Curriculum()->stages.size() == 3,
        "attaching the task reads the curriculum and leaves the document clean");
    Check(document.CurriculumProblem().empty(), "the sample curriculum is valid");
    auto edited = *document.Curriculum();
    edited.stages.push_back(editor::NewStage(task, "hard"));
    document.SetCurriculum(edited);
    Check(document.Dirty() && document.Text().find("\"hard\"") != std::string::npos, "a new stage is written");
    document.Undo();
    Check(!document.Dirty() && document.Curriculum()->stages.size() == 3, "and undone");
    document.SetCurriculum(std::nullopt);
    Check(document.Text().find("curriculum") == std::string::npos, "no curriculum: the section leaves the file");
}

} // namespace

int main()
{
    try
    {
        auto const task = schema::LoadTaskSchema(PPOKIT_SAMPLE_TASK);
        RoundTrip(task);
        NewStagesAreValid(task);
        RequiredDiscreteParameters(task);
        InvalidCurriculumIsNotNormalized(task);
        InvalidSavePreservesFile(task);
        Previews(task);
        Document(task);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0)
    {
        std::cerr << failures << " curriculum editing check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit curriculum editing tests passed\n";
    return EXIT_SUCCESS;
}
