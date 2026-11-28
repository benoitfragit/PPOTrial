/** @file launch_settings.cpp
 * @brief Show the effective project defaults and replace selected arguments when launching.
 */
#include "launch_settings.hpp"

#include "studio_theme.hpp"

#include <imgui.h>
#include <algorithm>
#include <filesystem>
#include <initializer_list>
#include <stdexcept>
#include <string_view>

namespace ppokit::studio {
namespace {
void EraseOptions(std::vector<std::string>& args, std::initializer_list<std::string_view> flags, bool withValue)
{
    for (auto it = args.begin(); it != args.end();)
    {
        bool matched = false;
        for (auto const flag : flags) matched = matched || *it == flag;
        if (matched)
        {
            it = args.erase(it);
            if (withValue && it != args.end()) it = args.erase(it);
        }
        else ++it;
    }
}

void Option(std::string const& label, std::string const& value, std::string& selected)
{
    if (ImGui::Selectable(label.c_str(), selected == value)) selected = value;
}
/** @brief Show the mode's declared default next to its independent override. */
void DrawPolicy(schema::Project const& project, schema::ProjectCommand command, LaunchSettings::PolicyChoice& choice,
    std::vector<schema::PolicyReference> const& references)
{
    auto const label = command == schema::ProjectCommand::Evaluate ? "Evaluation policy" : "Inference policy";
    std::string defaultLabel = "Project default";
    try
    {
        auto const declared = schema::DeclaredPolicy(project, command);
        auto source = declared.role;
        if (source.empty()) source = (declared.file.empty() ? project.Policy() : declared.file).string();
        defaultLabel += " (" + source + ")";
    }
    catch (std::exception const& error) { defaultLabel = error.what(); }
    ImGui::PushID(label);
    if (ImGui::BeginCombo(label, choice.role.empty() ? defaultLabel.c_str() : choice.role.c_str()))
    {
        Option(defaultLabel, "", choice.role);
        Option("LATEST", "LATEST", choice.role);
        Option("BEST", "BEST", choice.role);
        for (auto const& ref : references) Option(ref.role, ref.role, choice.role);
        Option("Custom file", "CUSTOM", choice.role);
        ImGui::EndCombo();
    }
    if (choice.role == "CUSTOM") ImGui::InputText("Policy file", choice.customFile.data(), choice.customFile.size());
    ImGui::PopID();
}
} // namespace

std::vector<std::string> LaunchSettings::Arguments(schema::ProjectCommand command, std::vector<std::string> args) const
{
    if (command == schema::ProjectCommand::Train)
    {
        if (start != Start::Project)
        {
            EraseOptions(args, {"--fresh", "--resume"}, false);
            args.push_back(start == Start::Fresh ? "--fresh" : "--resume");
        }
        if (budget != Budget::Project)
        {
            if (!updates) throw std::runtime_error("The update budget must be positive.");
            EraseOptions(args, {"--updates", "--until-update"}, true);
            args.insert(args.end(), {budget == Budget::Additional ? "--updates" : "--until-update", std::to_string(updates)});
        }
    }
    else
    {
        auto const& policy = command == schema::ProjectCommand::Evaluate ? evaluation : inference;
        if (policy.role.empty()) return args;
        EraseOptions(args, {"--best"}, false);
        EraseOptions(args, {"--policy", "--model"}, true);
        if (policy.role == "CUSTOM")
        {
            if (!policy.customFile.front()) throw std::runtime_error("Choose a custom policy file.");
            args.insert(args.end(), {"--policy", policy.customFile.data()});
        }
        else args.insert(args.end(), {"--model", policy.role});
    }
    return args;
}

void LaunchSettings::Draw(schema::Project const& project, std::vector<schema::PolicyReference> const& references)
{
    if (ImGui::Button("Launch settings...")) ImGui::OpenPopup("Launch settings");
    ItemTooltip("Choose how the next Train, Evaluate or Inference command starts.");

    // Limit the panel to the viewport; ImGui scrolls it when the screen is small.
    ImVec2 const available = ImGui::GetMainViewport()->WorkSize;
    ImGui::SetNextWindowSizeConstraints(ImVec2(0.F, 0.F),
        ImVec2(std::max(1.F, available.x * 0.9F), std::max(1.F, available.y * 0.8F)));
    if (!ImGui::BeginPopup("Launch settings")) return;
    Heading("Launch settings", "These choices apply to launches in this window. "
        "Project default uses the command configured by the project.");
    ImGui::SeparatorText("Training");
    char const* starts[] = {"Project default", "Resume checkpoint", "From scratch (archive prior run)"};
    char const* budgets[] = {"Project default", "Additional updates", "Absolute target update"};
    int startIndex = static_cast<int>(start), budgetIndex = static_cast<int>(budget);
    if (ImGui::Combo("Training start", &startIndex, starts, 3)) start = static_cast<Start>(startIndex);
    if (start == Start::Resume) MutedWrapped("Resume requires an existing checkpoint.");
    if (start == Start::Fresh) MutedWrapped("From scratch archives the previous run before starting again.");
    if (ImGui::Combo("Training budget", &budgetIndex, budgets, 3)) budget = static_cast<Budget>(budgetIndex);
    if (budget != Budget::Project)
    {
        ImGui::InputScalar("Updates", ImGuiDataType_U64, &updates);
        MutedWrapped(budget == Budget::Additional ? "Run this many updates in addition to the starting point."
                                                : "Stop when this total update number is reached.");
        if (updates == 0) ImGui::TextColored(HealthColor(editor::Health::Problem), "Enter a positive update count.");
    }
    auto const resume = project.LatestCheckpoint();
    ImGui::TextWrapped("Resume checkpoint [%s]  %s", std::filesystem::is_regular_file(resume) ? "available" : "missing",
        resume.string().c_str());
    ImGui::SeparatorText("Evaluation and inference");
    MutedWrapped("Choose a policy for each command. LATEST is the last trained policy. BEST uses the promoted "
                 "policy or the declared historical reference if no promotion exists.");
    DrawPolicy(project, schema::ProjectCommand::Evaluate, evaluation, references);
    DrawPolicy(project, schema::ProjectCommand::Inference, inference, references);
    if ((evaluation.role == "CUSTOM" && !evaluation.customFile.front())
        || (inference.role == "CUSTOM" && !inference.customFile.front()))
        ImGui::TextColored(HealthColor(editor::Health::Watch), "Enter a policy file for each custom selection.");
    if (!references.empty() && ImGui::TreeNode("Policy reference details"))
    {
        for (auto const& ref : references)
        {
            ImGui::TextWrapped("%s [%s]  %s", ref.role.c_str(), std::filesystem::is_regular_file(ref.file) ? "available" : "missing",
                ref.file.string().c_str());
            ImGui::TextWrapped("Expected SHA-256  %s", ref.sha256.c_str());
        }
        ImGui::TreePop();
    }
    ImGui::Spacing();
    if (ImGui::Button("Done")) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}
} // namespace ppokit::studio
