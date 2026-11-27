/**
 * @file PPOKit/studio/tests/curriculum_view_tests.cpp
 * @brief The curriculum editor, run headless: no curriculum, the sample curriculum with
 *        every rule kind, each stage selected with its preview, an invalid stage, and the
 *        settings view with its Curriculum tab.
 */
#include "curriculum_view.hpp"
#include "train_settings_view.hpp"

#include <imgui.h>
#include <implot.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
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

template<typename View>
void Frame(View& view)
{
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0.F, 0.F));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("curriculum", nullptr, ImGuiWindowFlags_NoDecoration);
    view.Draw();
    ImGui::End();
    ImGui::Render();
}

std::string ReadFile(char const* path)
{
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

int Run()
{
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1400.F, 1000.F);
    io.DeltaTime = 1.F / 60.F;
    io.IniFilename = nullptr;
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
    auto const task = schema::LoadTaskSchema(PPOKIT_SAMPLE_TASK);

    {
        auto document = editor::TrainSettingsDocument::Open(R"({ "schema": "ppokit.train/1" })", "train.json");
        document.AttachTask(task);
        studio::CurriculumView view(document);
        Frame(view);
        Check(!document.Curriculum(), "no curriculum: the view offers to create one");
    }

    auto document = editor::TrainSettingsDocument::Open(ReadFile(PPOKIT_SAMPLE_TRAIN), "train.json");
    document.AttachTask(task);
    studio::CurriculumView view(document);
    for (std::size_t stage = 0; stage < document.Curriculum()->stages.size(); ++stage)
    {
        view.SelectStage(stage);
        Frame(view);
        Frame(view);
    }
    Check(!document.Dirty(), "looking at every stage changes nothing");

    auto broken = *document.Curriculum();
    broken.stages[1].name = "not valid";
    document.SetCurriculum(broken);
    Frame(view);
    Check(document.CurriculumProblem().find("stages[1].name") != std::string::npos,
        "an invalid stage: the training message is shown, the preview hidden");
    document.Undo();
    Frame(view);

    // The settings view with a task shows the PPO and Curriculum tabs.
    std::filesystem::path const path = std::filesystem::temp_directory_path() / "ppokit_curriculum_view_train.json";
    std::ofstream(path, std::ios::binary) << ReadFile(PPOKIT_SAMPLE_TRAIN);
    {
        studio::TrainSettingsView settings(path, task);
        Frame(settings);
        Frame(settings);
        Check(settings.Document().Curriculum().has_value() && !settings.Dirty(), "the settings view reads the curriculum");
        settings.Document().Set(core::PpoParameter::Gamma, 0.995);
        settings.Revert();
        Check(settings.Document().Curriculum().has_value(), "revert keeps the task, hence the curriculum");
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);

    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    if (failures != 0)
    {
        std::cerr << failures << " curriculum view check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit studio curriculum view tests passed\n";
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
