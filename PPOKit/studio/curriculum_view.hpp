/**
 * @file PPOKit/studio/curriculum_view.hpp
 * @brief ImGui editor of the curriculum of a train.json: stages, their draw rules, a preview.
 * @details Left, the stages in training order (add, duplicate, remove, reorder); right, the
 * selected stage: its name, its promotion rule, and one row per scenario parameter of
 * task.json with its draw rule (task default, fixed, uniform, normal, choice) and the
 * widgets that rule needs. Above, the promotion window, threshold and rehearsal share, and
 * whether training would accept the curriculum (the training parser's own message). Below,
 * a preview: histograms of scenarios drawn from the selected stage by the curriculum engine.
 *
 * Edits go through the TrainSettingsDocument, so Save, Undo and Redo of the settings view
 * apply to them. Needs a current ImGui and ImPlot context.
 */
#pragma once

#include "ppokit/editor/curriculum_editing.hpp"
#include "ppokit/editor/train_settings_document.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace ppokit::studio {

class CurriculumView final
{
public:
    /** @param document A document with a task attached; it must outlive the view. */
    explicit CurriculumView(editor::TrainSettingsDocument& document) noexcept : document_(document) {}

    void Draw();

    void SelectStage(std::size_t stage) noexcept { selected_ = stage; }

private:
    void DrawEmpty();
    void DrawStatus();
    void DrawPromotion();
    void DrawStageList();
    void DrawStage();
    /** @brief Draw rule and values of one parameter; true when changed. */
    bool DrawSampler(schema::ScenarioParameter const& parameter, editor::EditableSampler& sampler);
    void DrawPreview();
    /** @brief Writes an edited copy back to the document, one undo step. */
    void Commit(editor::EditableCurriculum curriculum);
    /** @brief True while the widget being edited was already held last frame (one undo step per gesture). */
    bool Continuing() const noexcept;

    editor::TrainSettingsDocument& document_;
    std::size_t selected_ = 0;
    /** @brief Status and preview are recomputed only when the text or the stage changes. */
    std::string checkedText_;
    std::string problem_;
    std::string previewText_;
    std::size_t previewStage_ = 0;
    std::vector<editor::ParameterPreview> preview_;
    bool activeLastFrame_ = false;
};

} // namespace ppokit::studio
