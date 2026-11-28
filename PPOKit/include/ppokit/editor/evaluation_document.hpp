/**
 * @file PPOKit/include/ppokit/editor/evaluation_document.hpp
 * @brief The evaluation suites of a project (eval/<suite>.json), as the studio edits them (no GUI).
 * @details One editable suite per file. A suite is played on a range of seeds (or a list
 * the file already had), a number of episodes per seed, gates on the metrics (the built-in
 * ones and the m.rate / m.mean of the task's episode metrics), and, optionally, a rhythm
 * during training and the criteria that rank the BEST policy. Each suite is checked by the
 * parser of ppokit-run itself (eval::ParseEvaluationSuite), so the studio never saves a
 * suite that training would refuse. Saving writes eval/<name>.json and removes the file
 * of a suite deleted or renamed in the editor.
 */
#pragma once

#include "ppokit/eval/evaluation.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ppokit::editor {

/** @brief A gate as the editor shows it: metric, comparison, threshold. */
struct EditableGate final
{
    std::string metric;
    /** @brief True for ">=", false for "<=". */
    bool atLeast = true;
    double threshold = 0.0;
};

/** @brief A BEST criterion as the editor shows it. */
struct EditableRank final
{
    std::string metric;
    bool higher = true;
};

/** @brief One suite being edited. */
struct EditableSuite final
{
    std::string name;
    std::uint32_t firstSeed = 1000;
    std::size_t seedCount = 10;
    /** @brief Seeds of a file that are not a range; empty when the suite uses the range above. */
    std::vector<std::uint32_t> seedList;
    std::size_t episodesPerSeed = 1;
    /** @brief Keep one environment across the campaign, resetting it explicitly on each seed. */
    bool campaignEnvironment = false;
    /** @brief Runs during training, every `everyUpdates` updates and after the last one. */
    bool duringTraining = false;
    std::size_t everyUpdates = 10;
    std::vector<EditableGate> gates;
    /** @brief Further conditions for the training to be finished (train --until-passed). */
    std::vector<EditableGate> goal;
    std::vector<EditableRank> ranking;

    /** @brief The seeds of a campaign. */
    std::vector<std::uint32_t> Seeds() const;
};

/** @brief The eval/ folder of a project, edited suite by suite. */
class EvaluationDocument final
{
public:
    /**
     * @brief Reads every eval/<suite>.json of `folder` (it may not exist yet).
     * @param metrics Episode metrics of the task (task.json "metrics").
     * @throws schema::SchemaError for a file that does not read as a suite.
     */
    EvaluationDocument(std::filesystem::path folder, std::vector<schema::EpisodeMetric> metrics);

    std::vector<EditableSuite> const& Suites() const noexcept { return suites_; }
    /** @brief Suite `index` to edit in place; Problem and Dirty see the change at once. */
    EditableSuite& Suite(std::size_t index) { return suites_.at(index); }
    /** @brief Names a gate or a criterion may use, built-in ones first. */
    std::vector<std::string> const& MetricNames() const noexcept { return metricNames_; }
    std::vector<schema::EpisodeMetric> const& Metrics() const noexcept { return metrics_; }

    /** @brief Adds a new suite (a free name, ten seeds, success_rate >= 0.9); returns its index. */
    std::size_t Add();
    /** @brief Adds a copy of suite `index` under a free name; returns its index. */
    std::size_t Duplicate(std::size_t index);
    void Remove(std::size_t index);

    /** @brief Why suite `index` would be refused (the parser's own message), or empty. */
    std::string Problem(std::size_t index) const;
    /** @brief The file text of suite `index`. */
    std::string Text(std::size_t index) const;
    /** @brief Changes since the last load or save. */
    bool Dirty() const;
    /**
     * @brief Writes every suite and removes the files of the suites that are gone.
     * @return False, with the reason in `error`, when a suite is invalid or two share a name;
     *         nothing is written then.
     */
    bool Save(std::string& error);
    /** @brief Back to the files. */
    void Revert();

private:
    void Load();
    /** @brief The suites as their file texts, by name, for Dirty. */
    std::vector<std::string> Snapshot() const;
    std::string FreeName(std::string const& stem) const;

    std::filesystem::path folder_;
    std::vector<schema::EpisodeMetric> metrics_;
    std::vector<std::string> metricNames_;
    std::vector<EditableSuite> suites_;
    /** @brief File names of the suites as last loaded or saved. */
    std::vector<std::filesystem::path> files_;
    std::vector<std::string> saved_;
};

/** @brief An editable copy of a parsed suite. */
EditableSuite ToEditable(eval::EvaluationSuite const& suite);

} // namespace ppokit::editor
