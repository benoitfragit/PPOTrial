/**
 * @file PPOKit/include/ppokit/editor/train_settings_document.hpp
 * @brief The PPO parameters, probes and curriculum of a train.json, as the studio edits them (no GUI).
 * @details The document holds the "ppo" and "probes" sections of the file; once a task is
 * attached it also edits the curriculum (its stages draw the task's scenario parameters).
 * Every other section is kept exactly as it was. A parameter is either set in the file or
 * left to its default: Reset removes it from the file rather than writing the default,
 * so that a later change of default reaches every run that did not choose a value.
 *
 * CheckPpoParameters explains, in plain words, how the values fit together: the horizon
 * that gamma implies, a minibatch larger than the rollout, a very large learning rate...
 */
#pragma once

#include "ppokit/core/ppo_parameters.hpp"
#include "ppokit/editor/curriculum_editing.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ppokit::editor {

/** @brief The PPO section of one train.json, with edit and save. */
class TrainSettingsDocument final
{
public:
    /**
     * @brief Document of a train.json text; an empty text starts a new file.
     * @throws schema::SchemaError for a malformed file or an invalid PPO value.
     */
    static TrainSettingsDocument Open(std::string_view text, std::string_view sourceName);
    /** @brief Reads `path`; a missing file starts a new one (written by Save). */
    static TrainSettingsDocument Load(std::filesystem::path const& path);

    core::PpoParameterValues const& Values() const noexcept { return values_; }
    /** @brief Whether the file sets this parameter (otherwise it takes its default). */
    bool IsSet(core::PpoParameter parameter) const noexcept { return set_[Index(parameter)]; }
    /** @brief Changes since the last Open or Save. */
    bool Dirty() const noexcept { return text_ != savedText_; }

    /**
     * @brief Sets a value, constrained to the parameter's range, and writes it in the file.
     * @param continuing True while the same gesture goes on (a slider being dragged): the
     *        whole gesture is undone at once.
     */
    void Set(core::PpoParameter parameter, double value, bool continuing = false);
    /** @brief Back to the default: the parameter leaves the file. */
    void Reset(core::PpoParameter parameter);
    /** @brief Every parameter back to its default. */
    void ResetAll();

    /** @brief Updates between two probed updates; 0 when the probes are off. */
    std::size_t ProbeEvery() const noexcept { return probeEvery_; }
    /** @brief Probes every `every` updates; 0 turns them off (the section leaves the file). */
    void SetProbeEvery(std::size_t every);

    /**
     * @brief Lets the document edit the curriculum, whose stages refer to the task's parameters.
     * @details Validates the original plan before converting its curriculum to editable fields.
     *          A clean document stays clean: only an edit makes it dirty.
     * @throws schema::SchemaError when the plan is incompatible with the task.
     */
    void AttachTask(schema::TaskSchema task);
    std::optional<schema::TaskSchema> const& Task() const noexcept { return task_; }
    /** @brief The curriculum being edited; absent when the file has none (or no task is attached). */
    std::optional<EditableCurriculum> const& Curriculum() const noexcept { return curriculum_; }
    /**
     * @brief Replaces the curriculum (absent: the section leaves the file).
     * @param continuing True while the same gesture goes on (see Set).
     */
    void SetCurriculum(std::optional<EditableCurriculum> curriculum, bool continuing = false);
    /** @brief What training would say about the curriculum, or empty when it is valid. */
    std::string CurriculumProblem() const;

    bool CanUndo() const noexcept { return !undo_.empty(); }
    bool CanRedo() const noexcept { return !redo_.empty(); }
    void Undo();
    void Redo();

    /** @brief The train.json the document would write. */
    std::string const& Text() const noexcept { return text_; }
    /**
     * @brief Validates and writes Text() to `path` atomically; the document is then clean.
     * @throws schema::SchemaError if the training parser refuses the settings.
     */
    void Save(std::filesystem::path const& path);

private:
    /** @brief What an undo restores. */
    struct State final
    {
        core::PpoParameterValues values;
        std::array<bool, core::PpoParameterCount> set{};
        std::size_t probeEvery = 0;
        std::optional<EditableCurriculum> curriculum;
    };

    static std::size_t Index(core::PpoParameter parameter) noexcept { return static_cast<std::size_t>(parameter); }
    /** @brief Keeps the current state for Undo, before a change. */
    void Remember();
    void Restore(State const& state);
    /** @brief Rebuilds text_ from the kept sections and the current values. */
    void Render();

    /** @brief The file's root without its "ppo" section. */
    nlohmann::ordered_json others_;
    core::PpoParameterValues values_;
    std::array<bool, core::PpoParameterCount> set_{};
    std::size_t probeEvery_ = 0;
    std::optional<schema::TaskSchema> task_;
    std::optional<EditableCurriculum> curriculum_;
    std::string text_;
    std::string savedText_;
    std::vector<State> undo_;
    std::vector<State> redo_;
};

/** @brief How much a remark matters. */
enum class AdviceLevel : std::uint8_t
{
    /** @brief What the values mean, for a reader who is learning PPO. */
    Info,
    /** @brief A combination that usually trains badly or slowly. */
    Warning,
};

/** @brief One remark on the parameters, shown next to them in the studio. */
struct PpoAdvice final
{
    AdviceLevel level = AdviceLevel::Info;
    /** @brief The parameters the remark is about, highlighted by the editor. */
    std::vector<core::PpoParameter> parameters;
    std::string message;
};

/** @brief Plain-language remarks on how the values fit together, most important first. */
std::vector<PpoAdvice> CheckPpoParameters(core::PpoParameterValues const& values);

} // namespace ppokit::editor
