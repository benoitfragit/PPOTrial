/**
 * @file PPOKit/studio/tests/train_settings_view_tests.cpp
 * @brief The PPO settings editor, run headless: every group drawn, with and without
 *        warnings, a save that keeps the curriculum, a revert, a new file and an invalid one.
 */
#include "train_settings_view.hpp"

#include "ppokit/schema/task_schema.hpp"

#include <imgui.h>
#include <imgui_internal.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {

using ppokit::core::PpoParameter;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

void Frame(ppokit::studio::TrainSettingsView& view, std::string* rendered = nullptr)
{
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(0.F, 0.F));
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("settings", nullptr, ImGuiWindowFlags_NoDecoration);
    if (rendered) ImGui::LogToBuffer(2);
    view.Draw();
    if (rendered)
    {
        *rendered = ImGui::GetCurrentContext()->LogBuffer.c_str();
        ImGui::LogFinish();
    }
    ImGui::End();
    ImGui::Render();
}

std::string ReadFile(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void WriteFile(std::filesystem::path const& path, std::string const& text)
{
    std::ofstream(path, std::ios::binary) << text;
}

int Run()
{
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1200.F, 900.F);
    io.DeltaTime = 1.F / 60.F;
    io.IniFilename = nullptr;
    unsigned char* pixels = nullptr;
    int width = 0;
    int height = 0;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    std::filesystem::path const folder = std::filesystem::temp_directory_path() / "ppokit_train_settings_view";
    std::error_code ignored;
    std::filesystem::remove_all(folder, ignored);
    std::filesystem::create_directories(folder);
    std::filesystem::path const path = folder / "train.json";
    WriteFile(path, R"({ "schema": "ppokit.train/1", "ppo": { "epochs": 8 },
  "curriculum": { "stages": [ { "name": "calm", "sample": {} } ] } })");

    {
        ppokit::studio::TrainSettingsView view(path);
        Frame(view);
        Check(!view.Dirty() && view.Document().Values()[PpoParameter::Epochs] == 8.0, "the file is loaded");
        // Values that raise warnings: the rows are highlighted and the remarks listed.
        view.Document().Set(PpoParameter::Minibatch, 8192);
        view.Document().Set(PpoParameter::ActorLearningRate, 0.05);
        Frame(view);
        Frame(view);
        Check(view.Dirty(), "an edit makes the document dirty");
        Check(view.Save() && !view.Dirty(), "save");
        std::string const saved = ReadFile(path);
        Check(saved.find("\"minibatch\": 8192") != std::string::npos && saved.find("\"calm\"") != std::string::npos,
            "the file holds the new value and still the curriculum");
        view.Document().Set(PpoParameter::Gamma, 0.9);
        view.Revert();
        Frame(view);
        Check(!view.Dirty() && view.Document().Values()[PpoParameter::Gamma] == 0.99, "revert drops the edit");
        // A file broken by hand while the editor is open: the edits stay, the reason is shown.
        view.Document().Set(PpoParameter::Gamma, 0.9);
        WriteFile(path, "{ broken");
        view.Revert();
        Frame(view);
        Check(view.Dirty() && view.Status().find("Cannot reload") == 0, "a broken file does not lose the edits");
    }

    std::filesystem::path const created = folder / "new" / "train.json";
    {
        ppokit::studio::TrainSettingsView view(created);
        Frame(view);
        view.Document().Set(PpoParameter::Seed, 7);
        Check(view.Save() && std::filesystem::exists(created), "a new file is created by the first save");
    }
    {
        ppokit::schema::TaskSchema team;
        team.maximumAgents = 3;
        ppokit::studio::TrainSettingsView view(created, team, true);
        Frame(view);
        std::string rendered;
        Frame(view, &rendered);
        Check(rendered.find("Not applicable") != std::string::npos,
            "a separate team actor and critic do not offer the unused value coefficient");
        Check(!view.Dirty(), "displaying the unavailable coefficient does not change the file");
    }

    WriteFile(path, R"({ "schema": "ppokit.train/1", "ppo": { "gama": 0.9 } })");
    try
    {
        ppokit::studio::TrainSettingsView invalid(path);
        Check(false, "an invalid file is refused");
    }
    catch (ppokit::schema::SchemaError const& error)
    {
        Check(std::string(error.what()).find("did you mean 'gamma'") != std::string::npos,
            "the refusal says what to fix");
    }

    std::filesystem::remove_all(folder, ignored);
    ImGui::DestroyContext();
    if (failures != 0)
    {
        std::cerr << failures << " settings view check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit studio settings view tests passed\n";
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
