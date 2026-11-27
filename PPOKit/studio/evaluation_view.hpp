/**
 * @file PPOKit/studio/evaluation_view.hpp
 * @brief ImGui editor of the evaluation suites of a project (eval/<suite>.json), and their results.
 * @details Left, the suites (add, duplicate, remove). Right, the selected suite:
 * - its campaign: the seeds (a first seed and a count), the episodes per seed;
 * - its gates: one row per condition, a metric chosen among the built-in ones and the
 *   episode metrics of task.json (m.rate for a flag, m.mean for a number), ">=" or "<=",
 *   a threshold;
 * - during training: whether a campaign runs every N updates (and after the last one), and
 *   its goal: further conditions, in the gates' form, for the training to be finished;
 * - BEST: the ordered criteria that decide whether a passing campaign replaces the BEST;
 * - whether ppokit-run would accept the suite (its parser's own message);
 * - its results: on-demand evaluations, campaigns of the training journal and the BEST of the suite.
 * Save writes the suites; a suite training would refuse is never saved.
 *
 * Needs a current ImGui context, nothing else (the tests run it headless).
 */
#pragma once

#include "ppokit/editor/evaluation_document.hpp"
#include "ppokit/eval/best_store.hpp"
#include "ppokit/runtime/run_log.hpp"
#include "ppokit/schema/policy_reference.hpp"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::studio {

class EvaluationView final
{
public:
    /**
     * @param folder The eval/ folder of the project (created by the first save).
     * @param checkpoints The checkpoints folder, where the BEST manifests are.
     * @param metrics Episode metrics of the task.
     * @param references Archived actors supplied by the project catalog.
     * @throws schema::SchemaError for a suite file that does not read.
     */
    EvaluationView(std::filesystem::path folder, std::filesystem::path checkpoints,
        std::vector<schema::EpisodeMetric> metrics, std::vector<schema::PolicyReference> const& references = {});

    /** @brief Draws the editor; `journal` gives the campaigns of the run (null without one). */
    void Draw(runtime::RunLog const* journal);

    editor::EvaluationDocument& Document() noexcept { return document_; }
    bool Dirty() const { return document_.Dirty(); }
    /** @brief Writes the suites; on failure the message is shown and false returned. */
    bool Save();
    void Revert();
    /** @brief Reread the last completed evaluation and the verified BEST on the next frame. */
    void RefreshResults() noexcept;
    std::string const& Status() const noexcept { return status_; }
    std::size_t Selected() const noexcept { return selected_; }
    void Select(std::size_t suite) noexcept { selected_ = suite; }

private:
    void DrawToolbar();
    void DrawSuiteList();
    void DrawSuite(editor::EditableSuite& suite);
    void DrawCampaign(editor::EditableSuite& suite);
    /** @brief Rows of conditions (gates or goal): metric, comparison, threshold, remove. */
    void DrawConditions(char const* id, std::vector<editor::EditableGate>& conditions);
    void DrawGates(editor::EditableSuite& suite);
    void DrawTraining(editor::EditableSuite& suite);
    void DrawRanking(editor::EditableSuite& suite);
    void DrawVerdict();
    void DrawResults(editor::EditableSuite const& suite, runtime::RunLog const* journal);
    /** @brief Display the last on-demand report, reading its file only after a change. */
    void DrawSavedEvaluation(std::filesystem::path const& file, std::string const& suite);
    void DrawBest(std::string const& suite);
    /** @brief Rereads the BEST manifest of `suite` when it is another suite or the file changed. */
    void RefreshBest(std::string const& suite);
    /** @brief A combo of the metric names; true when changed. */
    bool MetricCombo(char const* id, std::string& metric) const;

    editor::EvaluationDocument document_;
    std::filesystem::path checkpoints_;
    std::size_t selected_ = 0;
    std::string status_;
    bool statusIsError_ = false;
    /** @brief The BEST manifest last read, its suite and its date: not reread every frame. */
    std::string bestSuite_;
    std::filesystem::file_time_type bestTime_{};
    std::optional<eval::BestRecord> best_;
    std::string bestError_;
    std::optional<schema::PolicyReference> historicalBest_;
    std::filesystem::path bestFile_;
    std::filesystem::file_time_type bestFileTime_{};
    std::filesystem::path evaluationFile_;
    std::filesystem::file_time_type evaluationTime_{};
    std::optional<eval::SavedEvaluation> savedEvaluation_;
    std::string evaluationError_;
};

} // namespace ppokit::studio
