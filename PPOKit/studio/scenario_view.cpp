/** @file scenario_view.cpp
 * @brief Display the configured JSON sources with their actual paths and parse diagnostics.
 */
#include "scenario_view.hpp"
#include "studio_theme.hpp"
#include "ppokit/runtime/publication.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/environment/environment_library.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

#include <fstream>
#include <imgui.h>

namespace ppokit::studio {
namespace {
using Json = nlohmann::ordered_json;
/** @brief Long serialized values expand on demand in the configuration tree. */
constexpr std::size_t InlineStringLimit = 120;

/** @brief Load a new snapshot on open or refresh. An invalid file clears the old snapshot. */
Json Read(std::filesystem::path const& path, std::string& error)
{
    error.clear();
    try
    {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Cannot open file");
        auto document = Json::parse(input);
        if (!document.is_object()) throw std::runtime_error("Expected a JSON object");
        return document;
    }
    catch (std::exception const& failure)
    {
        error = path.lexically_normal().string() + ": " + failure.what();
        return nullptr;
    }
}

/** @brief Expand a configuration subtree without interpreting simulator-specific fields. */
void DrawValue(std::string const& key, Json const& value)
{
    if (value.is_structured())
    {
        std::string label = key;
        if (value.is_object() && value.contains("name") && value.at("name").is_string())
            label += "  " + value.at("name").get<std::string>();
        if (ImGui::TreeNodeEx(key.c_str(), ImGuiTreeNodeFlags_DefaultOpen, "%s (%zu)", label.c_str(), value.size()))
        {
            for (auto const& item : value.items()) DrawValue(item.key(), item.value());
            ImGui::TreePop();
        }
    }
    else if (value.is_string() && value.get_ref<std::string const&>().size() > InlineStringLimit)
    {
        // A provider can include its serialized random generator alongside the curriculum.
        if (ImGui::TreeNode(key.c_str(), "%s (%zu characters)", key.c_str(), value.get_ref<std::string const&>().size()))
        {
            ImGui::TextWrapped("%s", value.get_ref<std::string const&>().c_str());
            ImGui::TreePop();
        }
    }
    else
    {
        ImGui::TextWrapped("%s  %s", key.c_str(), value.dump().c_str());
    }
}

/** @brief Show a read-only configuration snapshot or explain why its source could not be read. */
void DrawSource(std::filesystem::path const& path, Json const& root, std::string const& error)
{
    ImGui::TextWrapped("%s", path.lexically_normal().string().c_str());
    if (!error.empty()) ImGui::TextWrapped("%s", error.c_str());
    else for (auto const& item : root.items()) DrawValue(item.key(), item.value());
}

/** @brief A private validation folder; never write candidate settings to the active project. */
class ValidationFolder final
{
public:
    ValidationFolder()
    {
        std::random_device random;
        for (int attempt = 0; attempt < 16; ++attempt)
        {
            auto candidate = std::filesystem::temp_directory_path() / ("ppokit-scenario-" + std::to_string(random()));
            if (std::filesystem::create_directory(candidate))
            {
                path_ = std::move(candidate);
                return;
            }
        }
        throw std::runtime_error("Cannot allocate a private scenario validation folder");
    }
    ~ValidationFolder()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    ValidationFolder(ValidationFolder const&) = delete;
    ValidationFolder& operator=(ValidationFolder const&) = delete;
    std::filesystem::path const& Path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

/** @brief Draw the existing JSON hierarchy with typed widgets for approved scalar settings. */
void DrawEditableValue(ScenarioView& view, std::string const& key,
    Json::json_pointer const& pointer, Json const& value, bool external = false)
{
    ImGui::PushID(pointer.to_string().c_str());
    if (value.is_structured())
    {
        std::string label = key;
        if (value.is_object() && value.contains("name") && value.at("name").is_string())
            label += "  " + value.at("name").get<std::string>();
        if (ImGui::TreeNodeEx("section", ImGuiTreeNodeFlags_DefaultOpen, "%s (%zu)", label.c_str(), value.size()))
        {
            for (auto const& item : value.items())
                DrawEditableValue(view, item.key(), pointer / item.key(), item.value(), external);
            ImGui::TreePop();
        }
    }
    else if ((external ? view.CanEditExternal(pointer.to_string()) : view.CanEdit(pointer.to_string())) && value.is_boolean())
    {
        bool edited = value.get<bool>();
        if (ImGui::Checkbox(key.c_str(), &edited))
        {
            if (external) view.SetExternalScalar(pointer.to_string(), edited);
            else view.SetScalar(pointer.to_string(), edited);
        }
    }
    else if ((external ? view.CanEditExternal(pointer.to_string()) : view.CanEdit(pointer.to_string())) && value.is_number_unsigned())
    {
        std::uint64_t edited = value.get<std::uint64_t>();
        ImGui::SetNextItemWidth(155.F);
        if (ImGui::InputScalar(key.c_str(), ImGuiDataType_U64, &edited))
        {
            if (external) view.SetExternalScalar(pointer.to_string(), edited);
            else view.SetScalar(pointer.to_string(), edited);
        }
    }
    else if ((external ? view.CanEditExternal(pointer.to_string()) : view.CanEdit(pointer.to_string())) && value.is_number_integer())
    {
        std::int64_t edited = value.get<std::int64_t>();
        ImGui::SetNextItemWidth(155.F);
        if (ImGui::InputScalar(key.c_str(), ImGuiDataType_S64, &edited))
        {
            if (external) view.SetExternalScalar(pointer.to_string(), edited);
            else view.SetScalar(pointer.to_string(), edited);
        }
    }
    else if ((external ? view.CanEditExternal(pointer.to_string()) : view.CanEdit(pointer.to_string())) && value.is_number_float())
    {
        double edited = value.get<double>();
        ImGui::SetNextItemWidth(155.F);
        if (ImGui::InputDouble(key.c_str(), &edited, 0., 0., "%.10g"))
        {
            if (external) view.SetExternalScalar(pointer.to_string(), edited);
            else view.SetScalar(pointer.to_string(), edited);
        }
    }
    else
    {
        ImGui::TextWrapped("%s  %s", key.c_str(), value.dump().c_str());
    }
    ImGui::PopID();
}
} // namespace

ScenarioView::ScenarioView(schema::Project const& project, std::vector<std::filesystem::path> const& programs)
    : project_(project)
{
    if (!project_.environmentLibrary.empty() && project_.SettingsFile())
    {
        try
        {
            auto const path = schema::FindLibrary(project_, project_.environmentLibrary, programs);
            if (path.empty()) throw std::runtime_error("Environment module not found");
            auto const provider = environment::EnvironmentLibrary::Open(path.string(), {}, project_.environmentRunProfile);
            provider->VerifyTaskContract(schema::SemanticHash(schema::LoadTaskSchema(project_.Task())));
            if (provider->HasScenarioEditor()) provider_ = provider;
        }
        catch (std::exception const& error) { providerError_ = error.what(); }
    }
    Reload();
}

bool ScenarioView::Editable() const noexcept
{
    return editingAllowed_ && provider_ && scenarioError_.empty() && scenario_.is_object();
}

bool ScenarioView::CanEdit(std::string const& pointer) const noexcept
{
    return Editable() && provider_->CanEditScenario(pointer);
}

bool ScenarioView::SetScalar(std::string const& pointer, Json value)
{
    if (!CanEdit(pointer)) return false;
    try
    {
        auto& old = scenario_.at(Json::json_pointer(pointer));
        if ((!old.is_number() && !old.is_boolean()) || old.type() != value.type()
            || (value.is_number_float() && !std::isfinite(value.get<double>()))) return false;
        old = std::move(value);
        dirty_ = scenario_ != savedScenario_;
        status_.clear();
        return true;
    }
    catch (Json::exception const&) { return false; }
}

bool ScenarioView::CanEditExternal(std::string const& pointer) const noexcept
{
    try
    {
        if (!editingAllowed_ || !externalCurriculumError_.empty() || !externalCurriculum_.is_object()
            || externalCurriculum_.value("formatVersion", 0) != 2
            || externalCurriculum_.value("contractId", std::string{}) != "triangular-v1"
            || externalCurriculum_.value("maxTeamSize", 0) != 4) return false;
        if (pointer == "/rehearsalEveryBlocks") return true;
        constexpr char prefix[] = "/gates/";
        if (pointer.compare(0, sizeof(prefix) - 1, prefix) != 0) return false;
        auto const separator = pointer.find('/', sizeof(prefix) - 1);
        if (separator == std::string::npos || separator == sizeof(prefix) - 1) return false;
        if (!std::all_of(pointer.begin() + sizeof(prefix) - 1, pointer.begin() + separator,
                [](char c) { return c >= '0' && c <= '9'; })) return false;
        auto const field = pointer.substr(separator + 1);
        if (field != "thresholdPercent" && field != "maxSeconds"
            && field != "autoAdvance" && field != "hostileZones") return false;
        return externalCurriculum_.contains(Json::json_pointer(pointer));
    }
    catch (...) { return false; }
}

bool ScenarioView::SetExternalScalar(std::string const& pointer, Json value)
{
    if (!CanEditExternal(pointer)) return false;
    try
    {
        auto& old = externalCurriculum_.at(Json::json_pointer(pointer));
        if (old.type() != value.type()) return false;
        auto const field = pointer.substr(pointer.find_last_of('/') + 1);
        if (pointer == "/rehearsalEveryBlocks")
        {
            if (!value.is_number_unsigned() || value.get<std::uint64_t>() < 2
                || value.get<std::uint64_t>() > 100)
            {
                status_ = "Rehearsal period must be between 2 and 100 blocks.";
                return false;
            }
        }
        else if (field == "thresholdPercent")
        {
            if (!value.is_number()) return false;
            double const threshold = value.get<double>();
            if (!std::isfinite(threshold) || threshold <= 0. || threshold > 100.)
            {
                status_ = "Threshold must be greater than 0 and at most 100 percent.";
                return false;
            }
        }
        else if (field == "maxSeconds")
        {
            if (!value.is_number_unsigned() || value.get<std::uint64_t>() < 1
                || value.get<std::uint64_t>() > 18000)
            {
                status_ = "Gate duration must be between 1 and 18000 seconds.";
                return false;
            }
        }
        else if (!value.is_boolean()) return false;
        old = std::move(value);
        externalDirty_ = externalCurriculum_ != savedExternalCurriculum_;
        status_.clear();
        return true;
    }
    catch (Json::exception const&) { return false; }
}

bool ScenarioView::Save()
{
    if (dirty_ && !SaveScenario()) return false;
    return !externalDirty_ || SaveExternal();
}

bool ScenarioView::SaveScenario()
{
    try
    {
        if (!Editable()) throw std::runtime_error("The native scenario editor is unavailable while a run is active");
        auto const path = *project_.SettingsFile();
        std::string error;
        auto const current = Read(path, error);
        if (!error.empty() || current != savedScenario_)
            throw std::runtime_error("Scenario changed on disk. Revert and reload before saving.");
        ValidationFolder folder;
        auto const candidate = folder.Path() / "scenario.json";
        artifacts::WriteFileAtomically(candidate, [&](std::ostream& out) { out << scenario_.dump(2) << '\n'; },
            artifacts::AtomicWriteOptions{}, "scenario candidate");
        provider_->ValidateScenario(project_.root, candidate);
        // Native validation may take time. Recheck the source before replacing it.
        if (Read(path, error) != savedScenario_ || !error.empty())
            throw std::runtime_error("Scenario changed during validation. Revert and reload before saving.");
        artifacts::WriteFileAtomically(path, [&](std::ostream& out) { out << scenario_.dump(2) << '\n'; },
            artifacts::AtomicWriteOptions{}, "scenario settings");
        savedScenario_ = scenario_;
        dirty_ = false;
        status_ = "Scenario validated by the environment and saved.";
        return true;
    }
    catch (std::exception const& error)
    {
        status_ = std::string("Not saved: ") + error.what();
        return false;
    }
}

bool ScenarioView::SaveExternal()
{
    try
    {
        if (!editingAllowed_) throw std::runtime_error("Curriculum editing is unavailable while a run is active");
        auto const path = project_.root / project_.externalCurriculum;
        std::string error;
        if (Read(path, error) != savedExternalCurriculum_ || !error.empty())
            throw std::runtime_error("Curriculum changed on disk. Discard and reload before saving.");
        artifacts::WriteFileAtomically(path, [&](std::ostream& out) { out << externalCurriculum_.dump(2) << '\n'; },
            artifacts::AtomicWriteOptions{}, "external curriculum");
        savedExternalCurriculum_ = externalCurriculum_;
        externalDirty_ = false;
        status_ = "Curriculum saved; resume will refuse state from the previous plan, so start a fresh run.";
        return true;
    }
    catch (std::exception const& error)
    {
        status_ = std::string("Not saved: ") + error.what();
        return false;
    }
}

void ScenarioView::Revert()
{
    dirty_ = false;
    status_.clear();
    Reload();
}

void ScenarioView::RevertExternal()
{
    externalDirty_ = false;
    status_.clear();
    Reload();
}

void ScenarioView::DrawEditStatus()
{
    if (!provider_)
    {
        MutedWrapped("Read-only: this provider exposes no validated scenario editing interface. "
            "The values below are the configured source, not editable controls.");
        if (!providerError_.empty()) ImGui::TextWrapped("%s", providerError_.c_str());
        return;
    }
    if (!editingAllowed_) MutedWrapped("Read-only while a process is running.");
    else MutedWrapped("Edit existing numeric/boolean settings. Save validates them with the environment. "
        "Changes apply at the next launch and may affect learned behaviour.");
    if (dirty_)
    {
        ImGui::TextUnformatted("Unsaved scenario changes");
        ImGui::BeginDisabled(!editingAllowed_);
        if (ImGui::Button("Validate and save scenario")) SaveScenario();
        ImGui::SameLine();
        if (ImGui::Button("Discard scenario changes")) Revert();
        ImGui::EndDisabled();
    }
    if (!status_.empty()) ImGui::TextWrapped("%s", status_.c_str());
}

void ScenarioView::Reload()
{
    if (auto const file = project_.SettingsFile())
    {
        if (!dirty_)
        {
            scenario_ = Read(*file, scenarioError_);
            savedScenario_ = scenario_;
        }
        curriculumError_ = scenarioError_;
        if (!project_.curriculumPointer.empty())
        {
            if (curriculumError_.empty() && !scenario_.contains(Json::json_pointer(project_.curriculumPointer)))
                curriculumError_ = "Missing curriculum at " + project_.curriculumPointer;
            ReloadPublishedState();
        }
    }
    if (!project_.learnerSettings.empty())
        learnerSettings_ = Read(project_.root / project_.learnerSettings, learnerError_);
    if (!project_.externalCurriculum.empty())
    {
        if (!externalDirty_)
        {
            externalCurriculum_ = Read(project_.root / project_.externalCurriculum, externalCurriculumError_);
            savedExternalCurriculum_ = externalCurriculum_;
        }
        externalProgress_ = nullptr;
        externalProgressError_.clear();
        if (!project_.externalCurriculumProgress.empty())
        {
            auto const progress = project_.root / project_.externalCurriculumProgress;
            std::error_code error;
            bool const exists = std::filesystem::exists(progress, error);
            if (error) externalProgressError_ = error.message();
            else if (exists) externalProgress_ = Read(progress, externalProgressError_);
        }
    }
}

void ScenarioView::ReloadPublishedState()
{
    stateUpdate_.reset();
    publishedState_ = nullptr;
    stateError_.clear();
    try
    {
        auto const generation = runtime::ReadPublicationManifest(project_);
        if (!generation) return;
        stateUpdate_ = generation->update;
        if (!generation->state.empty())
        {
            // The environment ABI also permits plain text.
            publishedState_ = Json::parse(generation->state, nullptr, false);
            if (publishedState_.is_discarded()) publishedState_ = generation->state;
            else if (publishedState_.is_object()
                && publishedState_.value("schema", std::string{}) == "ppokit.run-training-state/1")
            {
                for (char const* field : {"environment", "curriculum"})
                {
                    if (!publishedState_.contains(field) || !publishedState_.at(field).is_string()) continue;
                    Json const nested = Json::parse(publishedState_.at(field).get<std::string>(), nullptr, false);
                    if (!nested.is_discarded()) publishedState_[field] = nested;
                }
            }
        }
    }
    catch (std::exception const& error) { stateError_ = error.what(); }
}

void ScenarioView::DrawEnvironment()
{
    Heading("Environment", "Scenario settings used at the next launch.");
    ImGui::TextWrapped("Environment module  %s", project_.environmentLibrary.empty()
        ? schema::CommandExecutable(project_, schema::ProjectCommand::Inference).c_str() : project_.environmentLibrary.c_str());
    if (auto const path = project_.SettingsFile())
    {
        if (ImGui::Button("Copy scenario path"))
            ImGui::SetClipboardText(path->lexically_normal().string().c_str());
        DrawEditStatus();
        if (!provider_ || !scenarioError_.empty()) DrawSource(*path, scenario_, scenarioError_);
        else
        {
            ImGui::TextWrapped("%s", path->lexically_normal().string().c_str());
            for (auto const& item : scenario_.items())
            {
                auto const pointer = Json::json_pointer{} / item.key();
                // No task-specific field list is maintained in the Studio.
                DrawEditableValue(*this, item.key(), pointer, item.value());
            }
        }
    }
    else MutedWrapped("This project declares no separate scenario settings file. Its settings belong to the environment module.");
}

std::vector<std::string> ScenarioView::CurriculumSummary() const
{
    if (!project_.externalCurriculum.empty())
    {
        if (!externalCurriculumError_.empty()) return {externalCurriculumError_};
        std::vector<std::string> lines{"Common curriculum",
            std::filesystem::path(project_.externalCurriculum).filename().string(), "Executed by ppokit-run"};
        if (externalProgress_.is_object()
            && externalProgress_.value("schema", std::string{}) == "ppokit.curriculum-status/1")
        {
            lines.push_back("Stage " + std::to_string(externalProgress_.value("index", std::size_t{}) + 1)
                + "/" + std::to_string(externalProgress_.value("count", std::size_t{})) + " "
                + externalProgress_.value("stage", std::string{}));
            lines.push_back(externalProgress_.value("reason", std::string{}));
        }
        return lines;
    }
    if (project_.curriculumPointer.empty()) return {};
    if (!curriculumError_.empty()) return {curriculumError_};
    auto const& curriculum = scenario_.at(Json::json_pointer(project_.curriculumPointer));
    std::vector<std::string> lines{project_.settings};
    if (curriculum.is_object() && curriculum.contains("stages") && curriculum.at("stages").is_array())
    {
        auto const& stages = curriculum.at("stages");
        lines.push_back(std::to_string(stages.size()) + " named stages");
        for (auto const& stage : stages)
            if (stage.is_object() && stage.contains("name") && stage.at("name").is_string())
                lines.push_back(stage.at("name").get<std::string>());
    }
    return lines;
}

void ScenarioView::DrawCurriculum()
{
    if (!project_.externalCurriculum.empty())
    {
        Heading("Curriculum", "Common curriculum source, validation criteria and persisted progress.");
        if (ImGui::Button("Refresh curriculum")) Reload();
        auto const source = project_.root / project_.externalCurriculum;
        ImGui::SameLine();
        if (ImGui::Button("Copy curriculum path"))
            ImGui::SetClipboardText(source.lexically_normal().string().c_str());
        ImGui::SeparatorText("Configured curriculum");
        MutedWrapped("Studio, CLI and CI execute this schedule through ppokit-run. Progress belongs to the "
            "same published generation as learner and environment state; a changed plan is refused on resume.");
        MutedWrapped("Thresholds, durations, zone and advance options, and rehearsal period are editable. "
            "Gate order and team sizes follow the project contract.");
        ImGui::PushID("external-curriculum");
        if (!externalCurriculumError_.empty()) DrawSource(source, externalCurriculum_, externalCurriculumError_);
        else
        {
            ImGui::TextWrapped("%s", source.lexically_normal().string().c_str());
            for (auto const& item : externalCurriculum_.items())
                DrawEditableValue(*this, item.key(), Json::json_pointer{} / item.key(), item.value(), true);
        }
        ImGui::PopID();
        if (externalDirty_)
        {
            ImGui::TextUnformatted("Unsaved curriculum changes");
            ImGui::BeginDisabled(!editingAllowed_);
            if (ImGui::Button("Save curriculum")) SaveExternal();
            ImGui::SameLine();
            if (ImGui::Button("Discard curriculum changes")) RevertExternal();
            ImGui::EndDisabled();
        }
        if (!status_.empty()) ImGui::TextWrapped("%s", status_.c_str());
        if (!project_.externalCurriculumProgress.empty())
        {
            ImGui::SeparatorText("Curriculum progress and last decision");
            if (externalProgress_.is_null() && externalProgressError_.empty())
                MutedWrapped("No progress file has been published.");
            else
            {
                ImGui::PushID("external-progress");
                DrawSource(project_.root / project_.externalCurriculumProgress, externalProgress_, externalProgressError_);
                ImGui::PopID();
            }
        }
        return;
    }
    if (project_.curriculumPointer.empty())
    {
        MutedWrapped("The project does not declare a native curriculum source.");
        return;
    }
    if (ImGui::Button("Refresh curriculum")) Reload();
    DrawPublishedState();
    ImGui::SeparatorText("Configured curriculum");
    ImGui::TextWrapped("%s", project_.curriculumPointer.c_str());
    DrawEditStatus();
    auto const file = *project_.SettingsFile();
    if (!curriculumError_.empty()) DrawSource(file, nullptr, curriculumError_);
    else if (!provider_) DrawSource(file, scenario_.at(Json::json_pointer(project_.curriculumPointer)), {});
    else
    {
        auto const pointer = Json::json_pointer(project_.curriculumPointer);
        ImGui::TextWrapped("%s", file.string().c_str());
        DrawEditableValue(*this, "Stages and promotion settings", pointer, scenario_.at(pointer));
    }
}

void ScenarioView::DrawPublishedState() const
{
    ImGui::SeparatorText("Last published training state");
    auto const file = project_.Checkpoints() / "generation.json";
    if (!stateError_.empty()) DrawSource(file, nullptr, stateError_);
    else if (!stateUpdate_) MutedWrapped("This project has no PPOKit generation manifest.");
    else
    {
        ImGui::Text("Checkpoint update %s", std::to_string(*stateUpdate_).c_str());
        ImGui::TextWrapped("%s", file.string().c_str());
        if (publishedState_.is_null()) MutedWrapped("The environment supplied no state.");
        else if (publishedState_.is_structured())
        {
            ImGui::PushID("published-state");
            for (auto const& item : publishedState_.items()) DrawValue(item.key(), item.value());
            ImGui::PopID();
        }
        else DrawValue("State", publishedState_);
    }
}

void ScenarioView::DrawLearnerSettings() const
{
    if (!project_.learnerSettings.empty())
    {
        auto const path = project_.root / project_.learnerSettings;
        MutedWrapped("Native learner configuration (read-only in the Studio).");
        if (ImGui::Button("Copy learner settings path"))
            ImGui::SetClipboardText(path.lexically_normal().string().c_str());
        DrawSource(path, learnerSettings_, learnerError_);
    }
    else if (auto const source = project_.SettingsFile())
    {
        ImGui::TextWrapped("%s", source->lexically_normal().string().c_str());
        if (!scenarioError_.empty()) ImGui::TextWrapped("%s", scenarioError_.c_str());
        else if (scenario_.contains("training"))
            DrawValue("training", scenario_.at("training"));
        else MutedWrapped("No training section is declared in this configuration.");
    }
}
} // namespace ppokit::studio
