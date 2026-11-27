/**
 * @file PPOKit/studio/curriculum_view.cpp
 * @brief Drawing of the curriculum editor.
 */
#include "curriculum_view.hpp"

#include <imgui.h>
#include <implot.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <utility>

namespace ppokit::studio {
namespace {

using editor::SamplerKind;
using schema::ParameterType;
using schema::ScenarioParameter;

constexpr ImVec4 ValidText{0.45F, 0.85F, 0.45F, 1.F};
constexpr ImVec4 ErrorText{1.F, 0.45F, 0.4F, 1.F};
constexpr float StageListWidthInEms = 15.F;
constexpr float PreviewPlotHeight = 150.F;
/** @brief A stage name: letters, digits and '_', as train.json requires. */
constexpr std::size_t StageNameCapacity = 64;

constexpr std::array<SamplerKind, 5> Kinds{SamplerKind::TaskDefault, SamplerKind::Fixed, SamplerKind::Uniform,
    SamplerKind::Normal, SamplerKind::Choice};

char const* KindName(SamplerKind kind) noexcept
{
    switch (kind)
    {
    case SamplerKind::TaskDefault:
        return "Task default";
    case SamplerKind::Fixed:
        return "Fixed";
    case SamplerKind::Uniform:
        return "Uniform";
    case SamplerKind::Normal:
        return "Normal";
    case SamplerKind::Choice:
        return "Choice";
    }
    return "";
}

char const* KindHelp(SamplerKind kind) noexcept
{
    switch (kind)
    {
    case SamplerKind::TaskDefault:
        return "Not written in the stage: the default of task.json applies.";
    case SamplerKind::Fixed:
        return "The same value in every episode of the stage.";
    case SamplerKind::Uniform:
        return "Any value between low and high, all equally likely (whole numbers for int parameters).";
    case SamplerKind::Normal:
        return "Values around a mean, with a spread; draws are kept inside the parameter's range.";
    case SamplerKind::Choice:
        return "One of a few values, each with a weight: weight 3 is three times as likely as weight 1.";
    }
    return "";
}

/** @brief "float in [0, 25], default 0", for the parameter tooltip. */
std::string Describe(ScenarioParameter const& parameter)
{
    char text[160];
    switch (parameter.type)
    {
    case ParameterType::Bool:
        return "true or false";
    case ParameterType::Enum:
    {
        std::string values = "one of:";
        for (std::string const& value : parameter.values)
        {
            values += " " + value;
        }
        return values;
    }
    case ParameterType::Int:
        std::snprintf(text, sizeof(text), "whole number in [%g, %g]", parameter.minimum, parameter.maximum);
        return text;
    case ParameterType::Float:
        std::snprintf(text, sizeof(text), "number in [%g, %g]", parameter.minimum, parameter.maximum);
        return text;
    }
    return {};
}

/** @brief One value of the parameter, with the widget of its type; true when changed. */
bool ValueWidget(char const* id, ScenarioParameter const& parameter, double& value)
{
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8.F);
    switch (parameter.type)
    {
    case ParameterType::Bool:
    {
        bool checked = value != 0.0;
        bool const changed = ImGui::Checkbox(id, &checked);
        value = checked ? 1.0 : 0.0;
        return changed;
    }
    case ParameterType::Enum:
    {
        int index = static_cast<int>(value);
        bool changed = false;
        std::string const& current = parameter.values.at(static_cast<std::size_t>(index));
        if (ImGui::BeginCombo(id, current.c_str()))
        {
            for (std::size_t option = 0; option < parameter.values.size(); ++option)
            {
                if (ImGui::Selectable(parameter.values[option].c_str(), static_cast<int>(option) == index))
                {
                    value = static_cast<double>(option);
                    changed = true;
                }
            }
            ImGui::EndCombo();
        }
        return changed;
    }
    case ParameterType::Int:
    {
        double const one = 1.0;
        bool const changed = ImGui::InputScalar(id, ImGuiDataType_Double, &value, &one, nullptr, "%.0f");
        value = std::clamp(std::round(value), parameter.minimum, parameter.maximum);
        return changed;
    }
    case ParameterType::Float:
    {
        float const speed = static_cast<float>(std::max(parameter.maximum - parameter.minimum, 1e-6) / 500.0);
        return ImGui::DragScalar(id, ImGuiDataType_Double, &value, speed, &parameter.minimum, &parameter.maximum,
            "%.4g", ImGuiSliderFlags_AlwaysClamp);
    }
    }
    return false;
}

/** @brief Stage name for a new stage: "stage_N" not used yet. */
std::string FreshStageName(editor::EditableCurriculum const& curriculum)
{
    for (std::size_t number = curriculum.stages.size() + 1;; ++number)
    {
        std::string const name = "stage_" + std::to_string(number);
        bool const used = std::any_of(curriculum.stages.begin(), curriculum.stages.end(),
            [&name](editor::EditableStage const& stage) { return stage.name == name; });
        if (!used)
        {
            return name;
        }
    }
}

} // namespace

void CurriculumView::Draw()
{
    if (!document_.Task())
    {
        ImGui::TextDisabled("Open the studio with --task task.json to edit the curriculum.");
        return;
    }
    if (!document_.Curriculum())
    {
        DrawEmpty();
        return;
    }
    DrawStatus();
    // Each section edits a fresh copy: an edit of one is never undone by a stale copy of another.
    DrawPromotion();
    if (document_.Curriculum())
    {
        ImGui::Separator();
        DrawStageList();
        selected_ = std::min(selected_, document_.Curriculum()->stages.size() - 1);
        ImGui::SameLine();
        ImGui::BeginGroup();
        DrawStage();
        ImGui::EndGroup();
        DrawPreview();
    }
    activeLastFrame_ = ImGui::IsAnyItemActive();
}

bool CurriculumView::Continuing() const noexcept
{
    // The same widget held since the last frame: a drag or a typing goes on.
    return activeLastFrame_ && ImGui::IsAnyItemActive();
}

void CurriculumView::DrawEmpty()
{
    ImGui::PushTextWrapPos(0.F);
    ImGui::TextUnformatted("No curriculum: every episode uses the task.json defaults of the scenario parameters.");
    ImGui::TextDisabled("A curriculum trains on easy scenarios first, then harder ones: each stage says how to "
                        "draw the scenario parameters, and training moves to the next stage once the success "
                        "rate over a window of episodes reaches a threshold.");
    ImGui::PopTextWrapPos();
    if (document_.Task()->scenario.empty())
    {
        ImGui::TextDisabled("This task declares no scenario parameter (\"scenario\" in task.json): nothing to vary.");
        return;
    }
    if (ImGui::Button("Create a curriculum"))
    {
        editor::EditableCurriculum curriculum;
        curriculum.stages.push_back(editor::NewStage(*document_.Task(), "start"));
        selected_ = 0;
        document_.SetCurriculum(std::move(curriculum));
    }
}

void CurriculumView::DrawStatus()
{
    if (document_.Text() != checkedText_)
    {
        checkedText_ = document_.Text();
        problem_ = document_.CurriculumProblem();
    }
    ImGui::PushTextWrapPos(0.F);
    if (problem_.empty())
    {
        ImGui::TextColored(ValidText, "Valid: training reads %zu stage(s).", document_.Curriculum()->stages.size());
    }
    else
    {
        ImGui::TextColored(ErrorText, "Training would refuse this curriculum: %s", problem_.c_str());
    }
    ImGui::PopTextWrapPos();
}

void CurriculumView::DrawPromotion()
{
    editor::EditableCurriculum curriculum = *document_.Curriculum();
    bool changed = false;
    int window = static_cast<int>(curriculum.window);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8.F);
    if (ImGui::InputInt("Window (episodes)", &window, 10, 100))
    {
        curriculum.window = static_cast<std::size_t>(std::max(window, 1));
        changed = true;
    }
    ImGui::SetItemTooltip("Episodes of the current stage over which the success rate is measured. Larger windows "
                          "promote later but more surely. A stage can set its own.");
    ImGui::SameLine();
    float promote = static_cast<float>(curriculum.promoteAt * 100.0);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 10.F);
    if (ImGui::SliderFloat("Promote at", &promote, 0.F, 100.F, "%.0f %% success", ImGuiSliderFlags_AlwaysClamp))
    {
        curriculum.promoteAt = static_cast<double>(promote) / 100.0;
        changed = true;
    }
    ImGui::SetItemTooltip("Success rate over a full window that moves training to the next stage. 70 to 90 %% is "
                          "usual; too low and harder stages come before the policy is ready.");
    ImGui::SameLine();
    float rehearse = static_cast<float>(curriculum.rehearse * 100.0);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 10.F);
    if (ImGui::SliderFloat("Rehearse", &rehearse, 0.F, 50.F, "%.0f %% of episodes", ImGuiSliderFlags_AlwaysClamp))
    {
        curriculum.rehearse = static_cast<double>(rehearse) / 100.0;
        changed = true;
    }
    ImGui::SetItemTooltip("Share of episodes that replay an earlier stage, so that the policy does not forget it. "
                          "They never count for promotion. 5 to 20 %% is usual.");
    ImGui::SameLine();
    if (ImGui::Button("Remove the curriculum"))
    {
        document_.SetCurriculum(std::nullopt);
        return;
    }
    ImGui::SetItemTooltip("Every episode will use the task.json defaults again (Undo brings it back).");
    if (changed)
    {
        document_.SetCurriculum(std::move(curriculum), Continuing());
    }
}

void CurriculumView::DrawStageList()
{
    editor::EditableCurriculum curriculum = *document_.Curriculum();
    ImGui::BeginChild("##stages", ImVec2(ImGui::GetFontSize() * StageListWidthInEms, 0.F),
        ImGuiChildFlags_Borders | ImGuiChildFlags_AutoResizeY);
    ImGui::TextDisabled("Stages, easiest first");
    for (std::size_t stage = 0; stage < curriculum.stages.size(); ++stage)
    {
        std::string const label = std::to_string(stage + 1) + ". " + curriculum.stages[stage].name;
        if (ImGui::Selectable(label.c_str(), stage == selected_))
        {
            selected_ = stage;
        }
    }
    ImGui::Spacing();
    bool const any = !curriculum.stages.empty();
    if (ImGui::SmallButton("Add"))
    {
        // A copy of the selected stage: the next stage is usually a harder variant of it.
        editor::EditableStage stage = any ? curriculum.stages[selected_] : editor::NewStage(*document_.Task(), "");
        stage.name = FreshStageName(curriculum);
        std::size_t const position = any ? selected_ + 1 : 0;
        curriculum.stages.insert(curriculum.stages.begin() + static_cast<std::ptrdiff_t>(position), std::move(stage));
        selected_ = position;
        Commit(curriculum);
    }
    ImGui::SetItemTooltip("A copy of the selected stage, right after it");
    ImGui::SameLine();
    ImGui::BeginDisabled(curriculum.stages.size() <= 1);
    if (ImGui::SmallButton("Remove"))
    {
        curriculum.stages.erase(curriculum.stages.begin() + static_cast<std::ptrdiff_t>(selected_));
        selected_ = std::min(selected_, curriculum.stages.size() - 1);
        Commit(curriculum);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(selected_ == 0);
    if (ImGui::SmallButton("Up"))
    {
        std::swap(curriculum.stages[selected_], curriculum.stages[selected_ - 1]);
        --selected_;
        Commit(curriculum);
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Train this stage earlier");
    ImGui::SameLine();
    ImGui::BeginDisabled(selected_ + 1 >= curriculum.stages.size());
    if (ImGui::SmallButton("Down"))
    {
        std::swap(curriculum.stages[selected_], curriculum.stages[selected_ + 1]);
        ++selected_;
        Commit(curriculum);
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Train this stage later");
    ImGui::EndChild();
}

void CurriculumView::DrawStage()
{
    editor::EditableCurriculum curriculum = *document_.Curriculum();
    editor::EditableStage& edited = curriculum.stages[selected_];
    bool changed = false;

    std::array<char, StageNameCapacity> name{};
    std::snprintf(name.data(), name.size(), "%s", edited.name.c_str());
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 12.F);
    if (ImGui::InputText("Name", name.data(), name.size()))
    {
        edited.name = name.data();
        changed = true;
    }
    ImGui::SetItemTooltip("Letters, digits and _ (it names the stage in logs and checkpoints).");

    bool ownWindow = edited.window.has_value();
    if (ImGui::Checkbox("Own window", &ownWindow))
    {
        edited.window = ownWindow ? std::optional<std::size_t>(curriculum.window) : std::nullopt;
        changed = true;
    }
    ImGui::SetItemTooltip("This stage measures its success over its own number of episodes");
    if (edited.window)
    {
        ImGui::SameLine();
        int window = static_cast<int>(*edited.window);
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 7.F);
        if (ImGui::InputInt("episodes##stage-window", &window, 10, 100))
        {
            edited.window = static_cast<std::size_t>(std::max(window, 1));
            changed = true;
        }
    }
    ImGui::SameLine();
    bool ownRate = edited.promoteAt.has_value();
    if (ImGui::Checkbox("Own threshold", &ownRate))
    {
        edited.promoteAt = ownRate ? std::optional<double>(curriculum.promoteAt) : std::nullopt;
        changed = true;
    }
    ImGui::SetItemTooltip("This stage promotes at its own success rate");
    if (edited.promoteAt)
    {
        ImGui::SameLine();
        float rate = static_cast<float>(*edited.promoteAt * 100.0);
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 9.F);
        if (ImGui::SliderFloat("##stage-rate", &rate, 0.F, 100.F, "%.0f %%", ImGuiSliderFlags_AlwaysClamp))
        {
            edited.promoteAt = static_cast<double>(rate) / 100.0;
            changed = true;
        }
    }

    schema::TaskSchema const& task = *document_.Task();
    if (ImGui::BeginTable("##parameters", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Parameter");
        ImGui::TableSetupColumn("Rule");
        ImGui::TableSetupColumn("Values");
        ImGui::TableHeadersRow();
        for (std::size_t index = 0; index < task.scenario.size(); ++index)
        {
            ImGui::PushID(static_cast<int>(index));
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextUnformatted(task.scenario[index].name.c_str());
            ImGui::SetItemTooltip("%s", Describe(task.scenario[index]).c_str());
            changed = DrawSampler(task.scenario[index], edited.samplers[index]) || changed;
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    if (changed)
    {
        document_.SetCurriculum(std::move(curriculum), Continuing());
    }
}

bool CurriculumView::DrawSampler(ScenarioParameter const& parameter, editor::EditableSampler& sampler)
{
    bool changed = false;
    ImGui::TableNextColumn();
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8.F);
    if (ImGui::BeginCombo("##rule", KindName(sampler.kind)))
    {
        for (SamplerKind const kind : Kinds)
        {
            if (kind == SamplerKind::TaskDefault && !parameter.defaultValue)
            {
                continue; // without a task default the stage must draw it
            }
            bool const discrete = parameter.type == schema::ParameterType::Bool
                || parameter.type == schema::ParameterType::Enum;
            if (discrete && (kind == SamplerKind::Uniform || kind == SamplerKind::Normal))
            {
                continue; // The training schema accepts only fixed values or choices for these types.
            }
            if (ImGui::Selectable(KindName(kind), kind == sampler.kind))
            {
                // Switching keeps the values typed for the other rules.
                sampler.kind = kind;
                changed = true;
            }
            ImGui::SetItemTooltip("%s", KindHelp(kind));
        }
        ImGui::EndCombo();
    }
    ImGui::SetItemTooltip("%s", KindHelp(sampler.kind));

    ImGui::TableNextColumn();
    switch (sampler.kind)
    {
    case SamplerKind::TaskDefault:
    {
        double shown = *parameter.defaultValue;
        ImGui::BeginDisabled();
        ValueWidget("##default", parameter, shown);
        ImGui::EndDisabled();
        break;
    }
    case SamplerKind::Fixed:
        changed = ValueWidget("##value", parameter, sampler.value) || changed;
        break;
    case SamplerKind::Uniform:
        changed = ValueWidget("##low", parameter, sampler.low) || changed;
        ImGui::SameLine();
        ImGui::TextUnformatted("to");
        ImGui::SameLine();
        changed = ValueWidget("##high", parameter, sampler.high) || changed;
        break;
    case SamplerKind::Normal:
    {
        changed = ValueWidget("mean", parameter, sampler.mean) || changed;
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 6.F);
        double const zero = 0.0;
        float const speed = static_cast<float>(std::max(parameter.maximum - parameter.minimum, 1e-6) / 500.0);
        changed = ImGui::DragScalar("spread", ImGuiDataType_Double, &sampler.deviation, speed, &zero, nullptr, "%.4g",
                      ImGuiSliderFlags_AlwaysClamp)
            || changed;
        break;
    }
    case SamplerKind::Choice:
    {
        std::size_t removed = sampler.choices.size();
        for (std::size_t choice = 0; choice < sampler.choices.size(); ++choice)
        {
            ImGui::PushID(static_cast<int>(choice));
            changed = ValueWidget("##choice", parameter, sampler.choices[choice].value) || changed;
            ImGui::SameLine();
            ImGui::SetNextItemWidth(ImGui::GetFontSize() * 4.F);
            double const zero = 0.0;
            changed = ImGui::DragScalar("weight", ImGuiDataType_Double, &sampler.choices[choice].weight, 0.05F, &zero,
                          nullptr, "%.3g", ImGuiSliderFlags_AlwaysClamp)
                || changed;
            ImGui::SameLine();
            if (ImGui::SmallButton("x"))
            {
                removed = choice;
            }
            ImGui::SetItemTooltip("Remove this value");
            ImGui::PopID();
        }
        if (removed < sampler.choices.size())
        {
            sampler.choices.erase(sampler.choices.begin() + static_cast<std::ptrdiff_t>(removed));
            changed = true;
        }
        if (ImGui::SmallButton("Add a value"))
        {
            sampler.choices.push_back({sampler.value, 1.0});
            changed = true;
        }
        break;
    }
    }
    return changed;
}

void CurriculumView::DrawPreview()
{
    if (!problem_.empty())
    {
        return;
    }
    if (document_.Text() != previewText_ || selected_ != previewStage_)
    {
        previewText_ = document_.Text();
        previewStage_ = selected_;
        preview_ = editor::PreviewStage(previewText_, *document_.Task(), selected_);
    }
    if (!ImGui::CollapsingHeader("Preview: 2000 scenarios drawn from this stage", ImGuiTreeNodeFlags_DefaultOpen))
    {
        return;
    }
    if (!ImGui::BeginTable("##preview", 3))
    {
        return;
    }
    for (editor::ParameterPreview const& parameter : preview_)
    {
        ImGui::TableNextColumn();
        std::vector<char const*> labels;
        for (std::string const& label : parameter.labels)
        {
            labels.push_back(label.c_str());
        }
        std::string const id = parameter.parameter + "##preview";
        if (ImPlot::BeginPlot(id.c_str(), ImVec2(-1.F, PreviewPlotHeight), ImPlotFlags_NoLegend | ImPlotFlags_NoMenus))
        {
            ImPlot::SetupAxes(nullptr, "share", ImPlotAxisFlags_AutoFit, ImPlotAxisFlags_AutoFit);
            ImPlot::SetupAxisTicks(ImAxis_X1, 0.0, static_cast<double>(labels.size() - 1),
                static_cast<int>(labels.size()), labels.data());
            ImPlot::PlotBars("##share", parameter.shares.data(), static_cast<int>(parameter.shares.size()), 0.67);
            ImPlot::EndPlot();
        }
    }
    ImGui::EndTable();
}

void CurriculumView::Commit(editor::EditableCurriculum curriculum)
{
    document_.SetCurriculum(std::move(curriculum));
}

} // namespace ppokit::studio
