/**
 * @file PPOKit/studio/train_settings_view.hpp
 * @brief ImGui editor of a train.json: PPO parameters and probes, and the curriculum when the
 *        task is known.
 * @details The parameters are grouped by what they control (returns, data per update,
 * step size, exploration, reproducibility). Each row shows a widget suited to the value
 * (logarithmic slider for learning rates, halve/double for sizes), a tooltip explaining
 * it with its default and range, and a "default" button when the file sets it. Under the
 * rows, plain-language remarks say how the values fit together; a parameter a warning is
 * about is highlighted. Ctrl+S saves, Ctrl+Z / Ctrl+Y undo and redo.
 *
 * Needs a current ImGui context, nothing else (the tests run it headless).
 */
#pragma once

#include "curriculum_view.hpp"

#include "ppokit/editor/train_settings_document.hpp"

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::studio {

class ScenarioView;

class TrainSettingsView final
{
public:
    /**
     * @brief Edits `path`; a missing file is created by the first save.
     * @param task The task of the run: with it, a Curriculum tab edits the stages.
     * @param separateTeamNetworks True when the team backend optimizes actor and critic separately.
     * @throws schema::SchemaError for an invalid file (reported before the window opens).
     */
    explicit TrainSettingsView(std::filesystem::path path, std::optional<schema::TaskSchema> task = std::nullopt,
        bool separateTeamNetworks = false);
    TrainSettingsView(TrainSettingsView const&) = delete;
    TrainSettingsView& operator=(TrainSettingsView const&) = delete;

    /** @brief Draws the editor in the current ImGui window. */
    void Draw(ScenarioView* nativeCurriculum = nullptr);
    /** @brief Render the existing curriculum editor directly, with the same document and save controls. */
    void DrawCurriculum();
    /** @brief Prevent edits, including keyboard shortcuts, while a process is active. */
    void SetEditingAllowed(bool allowed) noexcept { editingAllowed_ = allowed; }

    /** @brief Part of the settings a Draw brings to the front. */
    enum class Section
    {
        Ppo,
        Curriculum,
    };
    /** @brief Brings `section` to the front at the next Draw. */
    void Show(Section section) noexcept { shown_ = section; }

    editor::TrainSettingsDocument& Document() noexcept { return document_; }
    bool Dirty() const noexcept { return document_.Dirty(); }
    /** @brief Writes the file; on failure the message is shown and false returned. */
    bool Save();
    /** @brief Reloads the file from disk, dropping unsaved changes. */
    void Revert();
    std::string const& Status() const noexcept { return status_; }

private:
    void HandleShortcuts();
    void DrawPpo();
    void DrawToolbar();
    /** @brief New file, unsaved changes, saved, or the last error. */
    void DrawStatus();
    void DrawGroup(char const* group);
    void DrawRow(core::PpoParameter parameter);
    /** @brief The value widget; true when the user changed the value. */
    bool DrawValue(core::PpoParameter parameter, double& value);
    /** @brief Probes on or off, and their interval. */
    void DrawProbes();
    void DrawAdvice();
    /** @brief Recomputes the remarks and the warned parameters (when the document changed). */
    void Analyse();

    std::filesystem::path path_;
    editor::TrainSettingsDocument document_;
    std::vector<editor::PpoAdvice> advice_;
    /** @brief Text of the document the remarks were computed for. */
    std::string analysedText_;
    std::array<bool, core::PpoParameterCount> warned_{};
    std::string status_;
    bool statusIsError_ = false;
    bool editingAllowed_ = true;
    bool separateTeamNetworks_ = false;
    CurriculumView curriculum_;
    std::optional<Section> shown_;
};

} // namespace ppokit::studio
