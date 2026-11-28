/**
 * @file PPOKit/src/editor/evaluation_document.cpp
 * @brief Reading, checking and writing the evaluation suites of a project.
 */
#include "ppokit/editor/evaluation_document.hpp"

#include "ppokit/artifacts/atomic_file.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>
#include <utility>

namespace ppokit::editor {
namespace {

/** @brief Whether the seeds follow each other (a range the editor can show). */
bool Consecutive(std::vector<std::uint32_t> const& seeds) noexcept
{
    for (std::size_t index = 1; index < seeds.size(); ++index)
    {
        if (seeds[index] != seeds[index - 1] + 1)
        {
            return false;
        }
    }
    return !seeds.empty();
}

/** @brief Editable conditions as the parser's gates. */
std::vector<eval::Gate> ToGates(std::vector<EditableGate> const& conditions)
{
    std::vector<eval::Gate> gates;
    for (EditableGate const& gate : conditions)
    {
        eval::Gate written;
        written.metric = gate.metric;
        written.comparison = gate.atLeast ? eval::Comparison::GreaterOrEqual : eval::Comparison::LessOrEqual;
        written.threshold = gate.threshold;
        gates.push_back(written);
    }
    return gates;
}

std::vector<EditableGate> ToEditableGates(std::vector<eval::Gate> const& gates)
{
    std::vector<EditableGate> conditions;
    for (eval::Gate const& gate : gates)
    {
        conditions.push_back({gate.metric, gate.comparison == eval::Comparison::GreaterOrEqual, gate.threshold});
    }
    return conditions;
}

/** @brief The editable suite as the parser's structure, for SuiteToJson. */
eval::EvaluationSuite ToSuite(EditableSuite const& editable)
{
    eval::EvaluationSuite suite;
    suite.name = editable.name;
    suite.seeds = editable.Seeds();
    suite.episodesPerSeed = editable.episodesPerSeed;
    suite.campaignEnvironment = editable.campaignEnvironment;
    suite.gates = ToGates(editable.gates);
    suite.goal = ToGates(editable.goal);
    if (editable.duringTraining)
    {
        suite.everyUpdates = editable.everyUpdates;
    }
    for (EditableRank const& rank : editable.ranking)
    {
        suite.best.push_back({rank.metric, rank.higher});
    }
    return suite;
}

/** @brief Whether a suite already has this name. */
bool NameTaken(std::vector<EditableSuite> const& suites, std::string const& name) noexcept
{
    for (EditableSuite const& suite : suites)
    {
        if (suite.name == name)
        {
            return true;
        }
    }
    return false;
}

} // namespace

std::vector<std::uint32_t> EditableSuite::Seeds() const
{
    if (!seedList.empty())
    {
        return seedList;
    }
    std::vector<std::uint32_t> seeds;
    for (std::size_t index = 0; index < seedCount; ++index)
    {
        seeds.push_back(firstSeed + static_cast<std::uint32_t>(index));
    }
    return seeds;
}

EditableSuite ToEditable(eval::EvaluationSuite const& suite)
{
    EditableSuite editable;
    editable.name = suite.name;
    if (Consecutive(suite.seeds))
    {
        editable.firstSeed = suite.seeds.front();
        editable.seedCount = suite.seeds.size();
    }
    else
    {
        editable.seedList = suite.seeds;
    }
    editable.episodesPerSeed = suite.episodesPerSeed;
    editable.campaignEnvironment = suite.campaignEnvironment;
    editable.gates = ToEditableGates(suite.gates);
    editable.goal = ToEditableGates(suite.goal);
    editable.duringTraining = suite.everyUpdates != 0;
    if (editable.duringTraining)
    {
        editable.everyUpdates = suite.everyUpdates;
    }
    for (eval::RankKey const& key : suite.best)
    {
        editable.ranking.push_back({key.metric, key.higher});
    }
    return editable;
}

EvaluationDocument::EvaluationDocument(std::filesystem::path folder, std::vector<schema::EpisodeMetric> metrics)
    : folder_(std::move(folder)), metrics_(std::move(metrics)), metricNames_(eval::MetricNames(metrics_))
{
    Load();
}

void EvaluationDocument::Load()
{
    suites_.clear();
    files_.clear();
    std::vector<std::filesystem::path> paths;
    std::error_code error;
    for (std::filesystem::directory_entry const& entry : std::filesystem::directory_iterator(folder_, error))
    {
        if (entry.path().extension() == ".json")
        {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    for (std::filesystem::path const& path : paths)
    {
        suites_.push_back(ToEditable(eval::LoadEvaluationSuite(path, metrics_)));
        files_.push_back(path.filename());
    }
    saved_ = Snapshot();
}

std::vector<std::string> EvaluationDocument::Snapshot() const
{
    std::vector<std::string> texts;
    for (std::size_t index = 0; index < suites_.size(); ++index)
    {
        texts.push_back(Text(index));
    }
    return texts;
}

std::string EvaluationDocument::FreeName(std::string const& stem) const
{
    std::string name = stem;
    for (std::size_t number = 2;; ++number)
    {
        if (!NameTaken(suites_, name))
        {
            return name;
        }
        name = stem + "_" + std::to_string(number);
    }
}

std::size_t EvaluationDocument::Add()
{
    EditableSuite suite;
    suite.name = FreeName("evaluation");
    suite.gates.push_back({"success_rate", true, 0.9});
    suites_.push_back(std::move(suite));
    return suites_.size() - 1;
}

std::size_t EvaluationDocument::Duplicate(std::size_t index)
{
    EditableSuite copy = suites_.at(index);
    copy.name = FreeName(copy.name + "_copy");
    suites_.insert(suites_.begin() + static_cast<std::ptrdiff_t>(index) + 1, std::move(copy));
    return index + 1;
}

void EvaluationDocument::Remove(std::size_t index)
{
    suites_.erase(suites_.begin() + static_cast<std::ptrdiff_t>(index));
}

std::string EvaluationDocument::Text(std::size_t index) const
{
    return eval::SuiteToJson(ToSuite(suites_.at(index)));
}

std::string EvaluationDocument::Problem(std::size_t index) const
{
    EditableSuite const& suite = suites_.at(index);
    if (suite.gates.empty())
    {
        return "A suite needs at least one gate: it is how a campaign passes or fails.";
    }
    try
    {
        eval::ParseEvaluationSuite(Text(index), suite.name + ".json", metrics_);
    }
    catch (std::exception const& error)
    {
        return error.what();
    }
    return {};
}

bool EvaluationDocument::Dirty() const
{
    return Snapshot() != saved_;
}

bool EvaluationDocument::Save(std::string& error)
{
    std::set<std::string> names;
    for (std::size_t index = 0; index < suites_.size(); ++index)
    {
        std::string const problem = Problem(index);
        if (!problem.empty())
        {
            error = "suite '" + suites_[index].name + "': " + problem;
            return false;
        }
        if (!names.insert(suites_[index].name).second)
        {
            error = "two suites are named '" + suites_[index].name + "'";
            return false;
        }
    }
    std::filesystem::create_directories(folder_);
    std::vector<std::filesystem::path> written;
    for (std::size_t index = 0; index < suites_.size(); ++index)
    {
        std::filesystem::path const file = suites_[index].name + ".json";
        std::string const text = Text(index);
        artifacts::WriteFileAtomically(
            folder_ / file, [&text](std::ostream& out) { out << text; }, artifacts::AtomicWriteOptions{},
            "evaluation suite");
        written.push_back(file);
    }
    // The files of suites removed or renamed in the editor go with them.
    for (std::filesystem::path const& old : files_)
    {
        if (std::find(written.begin(), written.end(), old) == written.end())
        {
            std::error_code ignored;
            std::filesystem::remove(folder_ / old, ignored);
        }
    }
    files_ = written;
    saved_ = Snapshot();
    error.clear();
    return true;
}

void EvaluationDocument::Revert()
{
    Load();
}

} // namespace ppokit::editor
