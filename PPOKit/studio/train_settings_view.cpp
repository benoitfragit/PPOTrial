/**
 * @file PPOKit/studio/train_settings_view.cpp
 * @brief Drawing of the PPO parameter editor of train.json.
 */
#include "train_settings_view.hpp"
#include "scenario_view.hpp"

#include "help_tooltip.hpp"

#include "ppokit/editor/ppo_help.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <sstream>
#include <utility>

namespace ppokit::studio {
namespace {

using core::PpoParameter;

constexpr ImVec4 WarningText{1.F, 0.78F, 0.3F, 1.F};
constexpr ImVec4 ErrorText{1.F, 0.45F, 0.4F, 1.F};
constexpr ImVec4 SavedText{0.45F, 0.85F, 0.45F, 1.F};
/** @brief A drag over the whole range takes about this many pixels. */
constexpr double DragPixelsPerRange = 2000.0;
constexpr float ValueWidthInEms = 14.F;
/** @brief Probe interval proposed when the probes are switched on. */
constexpr std::size_t DefaultProbeEvery = 10;

/** @brief A number as a user would read it (0.0003, 4096, 0.99). */
std::string Shown(double value)
{
    std::ostringstream text;
    text << std::setprecision(10) << value;
    return text.str();
}

/** @brief Sizes that are usually powers of two get halve/double buttons. */
bool IsSize(PpoParameter parameter) noexcept
{
    return parameter == PpoParameter::RolloutSteps || parameter == PpoParameter::Minibatch;
}

/** @brief "key gamma · default 0.99 · range [0, 1]", the footer of a parameter's tooltip. */
std::string Footer(core::PpoParameterSpec const& spec, char const* gesture)
{
    return std::string("train.json key: ") + spec.name + "\nDefault " + Shown(spec.defaultValue) + ", range ["
        + Shown(spec.minimum) + ", " + Shown(spec.maximum) + "]\n" + gesture;
}

char const* GestureHint(PpoParameter parameter) noexcept
{
    core::PpoParameterSpec const& spec = core::SpecOf(parameter);
    if (IsSize(parameter))
    {
        return "Type a whole number, or halve / double it with the buttons.";
    }
    if (spec.logarithmic)
    {
        return "Drag the slider (logarithmic scale), or Ctrl+click to type a value.";
    }
    if (spec.integral)
    {
        return "Type a whole number, or use the - / + buttons.";
    }
    return "Drag to adjust, or double-click to type a value.";
}

/** @brief The groups in catalog order, each once (computed on first use). */
std::vector<char const*> const& Groups()
{
    static std::vector<char const*> const groups = [] {
        std::vector<char const*> names;
        for (core::PpoParameterSpec const& spec : core::PpoParameterSpecs)
        {
            if (names.empty() || std::string(names.back()) != spec.group)
            {
                names.push_back(spec.group);
            }
        }
        return names;
    }();
    return groups;
}

} // namespace

TrainSettingsView::TrainSettingsView(std::filesystem::path path, std::optional<schema::TaskSchema> task,
    bool separateTeamNetworks)
    : path_(std::move(path)), document_(editor::TrainSettingsDocument::Load(path_)),
      separateTeamNetworks_(separateTeamNetworks), curriculum_(document_)
{
    if (task)
    {
        document_.AttachTask(std::move(*task));
    }
}

void TrainSettingsView::Draw(ScenarioView* nativeCurriculum)
{
    if (document_.Text() != analysedText_)
    {
        Analyse();
    }
    HandleShortcuts();
    if (!editingAllowed_) ImGui::TextDisabled("Read-only while a process is running.");
    ImGui::BeginDisabled(!editingAllowed_);
    DrawToolbar();
    ImGui::Separator();
    if (!document_.Task())
    {
        DrawPpo();
        ImGui::EndDisabled();
        return;
    }
    if (ImGui::BeginTabBar("##train-settings"))
    {
        ImGuiTabItemFlags const ppoFlags = shown_ == Section::Ppo ? ImGuiTabItemFlags_SetSelected : 0;
        ImGuiTabItemFlags const curriculumFlags = shown_ == Section::Curriculum ? ImGuiTabItemFlags_SetSelected : 0;
        shown_.reset();
        if (ImGui::BeginTabItem("PPO and GAE", nullptr, ppoFlags))
        {
            DrawPpo();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Curriculum", nullptr, curriculumFlags))
        {
            ImGui::BeginChild("##curriculum", ImVec2(0.F, 0.F));
            if (nativeCurriculum) nativeCurriculum->DrawCurriculum();
            else curriculum_.Draw();
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::EndDisabled();
}

void TrainSettingsView::DrawCurriculum()
{
    HandleShortcuts();
    if (!editingAllowed_) ImGui::TextDisabled("Read-only while a process is running.");
    ImGui::BeginDisabled(!editingAllowed_);
    DrawToolbar();
    ImGui::Separator();
    if (document_.Task()) curriculum_.Draw();
    else ImGui::TextDisabled("A valid task is required to edit curriculum stages.");
    ImGui::EndDisabled();
}

void TrainSettingsView::DrawPpo()
{
    ImGui::BeginChild("##ppo-parameters", ImVec2(0.F, 0.F));
    for (char const* group : Groups())
    {
        DrawGroup(group);
    }
    DrawProbes();
    DrawAdvice();
    ImGui::EndChild();
}

bool TrainSettingsView::Save()
{
    if (!editingAllowed_)
    {
        status_ = "Not saved: a process is running.";
        statusIsError_ = true;
        return false;
    }
    try
    {
        document_.Save(path_);
        status_ = "Saved " + path_.filename().string() + ".";
        statusIsError_ = false;
        return true;
    }
    catch (std::exception const& error)
    {
        status_ = std::string("Not saved: ") + error.what();
        statusIsError_ = true;
        return false;
    }
}

void TrainSettingsView::Revert()
{
    if (!editingAllowed_) return;
    try
    {
        auto reloaded = editor::TrainSettingsDocument::Load(path_);
        if (document_.Task())
        {
            reloaded.AttachTask(*document_.Task());
        }
        document_ = std::move(reloaded);
        status_ = "Reloaded " + path_.filename().string() + ".";
        statusIsError_ = false;
    }
    catch (std::exception const& error)
    {
        // The file was broken by hand meanwhile: keep the edits on screen, say why.
        status_ = std::string("Cannot reload: ") + error.what();
        statusIsError_ = true;
    }
}

void TrainSettingsView::HandleShortcuts()
{
    if (!editingAllowed_ || ImGui::GetIO().WantTextInput)
    {
        return; // Ctrl+Z in a text field edits the text, not the document.
    }
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Z))
    {
        document_.Undo();
    }
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Y))
    {
        document_.Redo();
    }
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S))
    {
        Save();
    }
}

void TrainSettingsView::DrawToolbar()
{
    ImGui::TextUnformatted("Training settings");
    ImGui::SameLine();
    ImGui::TextDisabled("%s", path_.string().c_str());

    ImGui::BeginDisabled(!document_.Dirty());
    if (ImGui::Button("Save"))
    {
        Save();
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Write train.json (Ctrl+S). Only the parameters you set are written.");
    ImGui::SameLine();
    ImGui::BeginDisabled(!document_.Dirty() || !std::filesystem::exists(path_));
    if (ImGui::Button("Revert"))
    {
        Revert();
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Drop the unsaved changes and reload the file");
    ImGui::SameLine();
    ImGui::BeginDisabled(!document_.CanUndo());
    if (ImGui::Button("Undo"))
    {
        document_.Undo();
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Undo the last change (Ctrl+Z)");
    ImGui::SameLine();
    ImGui::BeginDisabled(!document_.CanRedo());
    if (ImGui::Button("Redo"))
    {
        document_.Redo();
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Redo (Ctrl+Y)");
    ImGui::SameLine();
    if (ImGui::Button("All defaults"))
    {
        document_.ResetAll();
    }
    ImGui::SetItemTooltip("Every parameter back to its default (can be undone)");

    ImGui::SameLine();
    DrawStatus();
}

void TrainSettingsView::DrawStatus()
{
    if (statusIsError_)
    {
        ImGui::TextColored(ErrorText, "%s", status_.c_str());
        return;
    }
    if (!std::filesystem::exists(path_))
    {
        ImGui::TextColored(WarningText, "New file: Save creates it.");
        return;
    }
    if (document_.Dirty())
    {
        ImGui::TextColored(WarningText, "Unsaved changes");
        return;
    }
    ImGui::TextColored(SavedText, "%s", status_.empty() ? "Up to date" : status_.c_str());
}

void TrainSettingsView::DrawGroup(char const* group)
{
    if (!ImGui::CollapsingHeader(group, ImGuiTreeNodeFlags_DefaultOpen))
    {
        return;
    }
    std::string const id = std::string("##group-") + group;
    if (!ImGui::BeginTable(id.c_str(), 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_RowBg))
    {
        return;
    }
    ImGui::TableSetupColumn("Parameter", ImGuiTableColumnFlags_WidthStretch, 1.F);
    ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * (ValueWidthInEms + 5.F));
    ImGui::TableSetupColumn("Default", ImGuiTableColumnFlags_WidthFixed, ImGui::GetFontSize() * 9.F);
    for (std::size_t index = 0; index < core::PpoParameterCount; ++index)
    {
        if (std::string(core::PpoParameterSpecs[index].group) == group)
        {
            DrawRow(static_cast<PpoParameter>(index));
        }
    }
    ImGui::EndTable();
}

void TrainSettingsView::DrawRow(PpoParameter parameter)
{
    core::PpoParameterSpec const& spec = core::SpecOf(parameter);
    editor::HelpEntry const* const help = editor::FindHelp(spec.name);
    std::size_t const index = static_cast<std::size_t>(parameter);
    ImGui::PushID(spec.name);
    ImGui::TableNextRow();

    // Name: highlighted when a warning below is about it; the tooltip explains it.
    ImGui::TableNextColumn();
    char const* const title = help != nullptr ? help->title : spec.name;
    if (warned_[index])
    {
        ImGui::TextColored(WarningText, "%s  (see below)", title);
    }
    else
    {
        ImGui::TextUnformatted(title);
    }
    if (help != nullptr)
    {
        HelpTooltip(*help, Footer(spec, GestureHint(parameter)));
    }

    // Value.
    ImGui::TableNextColumn();
    bool const teamValueCoefficient = parameter == PpoParameter::ValueCoefficient && separateTeamNetworks_;
    if (teamValueCoefficient)
    {
        ImGui::TextDisabled("Not applicable");
        ImGui::SetItemTooltip("The team actor and critic have separate optimizers; this coefficient is unused.");
        ImGui::TableNextColumn();
        ImGui::TextDisabled("-");
        ImGui::PopID();
        return;
    }
    double value = document_.Values()[parameter];
    if (DrawValue(parameter, value))
    {
        document_.Set(parameter, value, ImGui::IsItemActive() && !ImGui::IsItemActivated());
    }

    // Default: either it applies, or a button brings it back.
    ImGui::TableNextColumn();
    if (document_.IsSet(parameter))
    {
        if (ImGui::SmallButton("Use default"))
        {
            document_.Reset(parameter);
        }
        ImGui::SetItemTooltip("Remove %s from train.json: the default %s applies.", spec.name,
            Shown(spec.defaultValue).c_str());
    }
    else
    {
        ImGui::TextDisabled("default");
        ImGui::SetItemTooltip("Not in train.json: the default %s applies. Change the value to set it.",
            Shown(spec.defaultValue).c_str());
    }
    ImGui::PopID();
}

bool TrainSettingsView::DrawValue(PpoParameter parameter, double& value)
{
    core::PpoParameterSpec const& spec = core::SpecOf(parameter);
    float const width = ImGui::GetFontSize() * ValueWidthInEms;
    if (spec.logarithmic)
    {
        ImGui::SetNextItemWidth(width);
        return ImGui::SliderScalar("##value", ImGuiDataType_Double, &value, &spec.minimum, &spec.maximum, "%.2e",
            ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
    }
    if (spec.integral)
    {
        double const one = 1.0;
        double const ten = 10.0;
        bool changed = false;
        if (IsSize(parameter))
        {
            // Sizes are usually powers of two: halve and double are one click away.
            ImGui::SetNextItemWidth(width - ImGui::GetFontSize() * 4.F);
            changed = ImGui::InputScalar("##value", ImGuiDataType_Double, &value, nullptr, nullptr, "%.0f");
            ImGui::SameLine();
            if (ImGui::SmallButton("/2"))
            {
                value = std::floor(value / 2.0);
                changed = true;
            }
            ImGui::SetItemTooltip("Halve");
            ImGui::SameLine();
            if (ImGui::SmallButton("x2"))
            {
                value *= 2.0;
                changed = true;
            }
            ImGui::SetItemTooltip("Double");
            return changed;
        }
        ImGui::SetNextItemWidth(width);
        return ImGui::InputScalar("##value", ImGuiDataType_Double, &value, &one, &ten, "%.0f");
    }
    ImGui::SetNextItemWidth(width);
    float const speed = static_cast<float>((spec.maximum - spec.minimum) / DragPixelsPerRange);
    return ImGui::DragScalar("##value", ImGuiDataType_Double, &value, speed, &spec.minimum, &spec.maximum, "%.4g",
        ImGuiSliderFlags_AlwaysClamp);
}

void TrainSettingsView::DrawProbes()
{
    if (!ImGui::CollapsingHeader("Diagnostics", ImGuiTreeNodeFlags_DefaultOpen))
    {
        return;
    }
    std::size_t const every = document_.ProbeEvery();
    bool enabled = every != 0;
    if (ImGui::Checkbox("Probe the network blocks", &enabled))
    {
        document_.SetProbeEvery(enabled ? DefaultProbeEvery : 0);
    }
    ImGui::SetItemTooltip("Measures every block of the actor and critic during a probed update: what it outputs "
                          "(spread, dead or saturated units) and how much gradient reaches it. Shown in the "
                          "Probes tab of the run. Costs nothing when off; a probed update is a little slower.");
    if (!enabled)
    {
        return;
    }
    ImGui::SameLine();
    int interval = static_cast<int>(every);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 7.F);
    if (ImGui::InputInt("updates between two probed updates", &interval, 1, 10))
    {
        document_.SetProbeEvery(static_cast<std::size_t>(std::max(interval, 1)));
    }
    ImGui::SetItemTooltip("1 probes every update; 10 is enough to follow a long run.");
}

void TrainSettingsView::DrawAdvice()
{
    ImGui::Spacing();
    ImGui::SeparatorText("How these values fit together");
    ImGui::PushTextWrapPos(0.F);
    // Warnings first: they are what to look at.
    for (editor::PpoAdvice const& entry : advice_)
    {
        if (entry.level == editor::AdviceLevel::Warning)
        {
            ImGui::TextColored(WarningText, "Warning: %s", entry.message.c_str());
        }
    }
    for (editor::PpoAdvice const& entry : advice_)
    {
        if (entry.level == editor::AdviceLevel::Info)
        {
            ImGui::BulletText("%s", entry.message.c_str());
        }
    }
    ImGui::PopTextWrapPos();
}

void TrainSettingsView::Analyse()
{
    advice_ = editor::CheckPpoParameters(document_.Values());
    warned_.fill(false);
    for (editor::PpoAdvice const& entry : advice_)
    {
        if (entry.level != editor::AdviceLevel::Warning)
        {
            continue;
        }
        for (PpoParameter const parameter : entry.parameters)
        {
            warned_[static_cast<std::size_t>(parameter)] = true;
        }
    }
    analysedText_ = document_.Text();
}

} // namespace ppokit::studio
