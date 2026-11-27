/**
 * @file PPOKit/studio/evaluation_view.cpp
 * @brief Drawing of the evaluation suites editor and of their campaigns.
 */
#include "evaluation_view.hpp"

#include "ppokit/artifacts/sha256.hpp"
#include "studio_theme.hpp"

#include <imgui.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace ppokit::studio {
namespace {

constexpr ImVec4 ValidText{0.45F, 0.85F, 0.45F, 1.F};
constexpr ImVec4 ErrorText{1.F, 0.45F, 0.4F, 1.F};
constexpr float SuiteListWidthInEms = 14.F;
/** @brief A suite name: letters, digits and '_', as the suite file requires. */
constexpr std::size_t NameCapacity = 64;
/** @brief Campaigns of the journal shown for a suite, most recent first. */
constexpr std::size_t ShownCampaigns = 12;

/** @brief File change marker used to avoid reading the policy or its manifest in every frame. */
std::filesystem::file_time_type FileTime(std::filesystem::path const& file) noexcept
{
    if (file.empty()) return std::filesystem::file_time_type::min();
    std::error_code error;
    return std::filesystem::last_write_time(file, error);
}

/** @brief The text of a value: "n/a" for NaN (a metric that never applied). */
std::string ValueText(double value)
{
    if (!std::isfinite(value))
    {
        return "n/a";
    }
    char text[32];
    std::snprintf(text, sizeof(text), "%.4g", value);
    return text;
}

/** @brief An unsigned whole number; true when changed. Negative entries are refused. */
bool CountInput(char const* label, std::size_t& value, std::size_t minimum)
{
    int shown = static_cast<int>(value);
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8.F);
    if (!ImGui::InputInt(label, &shown))
    {
        return false;
    }
    value = static_cast<std::size_t>(std::max(shown, static_cast<int>(minimum)));
    return true;
}

/** @brief Swaps elements `index` and `index + offset` when both exist. */
template<typename Item>
void Move(std::vector<Item>& items, std::size_t index, int offset)
{
    std::size_t const other = index + static_cast<std::size_t>(offset);
    if (offset < 0 && index == 0)
    {
        return;
    }
    if (other < items.size())
    {
        std::swap(items[index], items[other]);
    }
}

} // namespace

EvaluationView::EvaluationView(std::filesystem::path folder, std::filesystem::path checkpoints,
    std::vector<schema::EpisodeMetric> metrics, std::vector<schema::PolicyReference> const& references)
    : document_(std::move(folder), std::move(metrics)), checkpoints_(std::move(checkpoints))
{
    for (auto const& ref : references)
        if (ref.role == "HISTORICAL_BEST") historicalBest_ = ref;
}

bool EvaluationView::Save()
{
    if (!editingAllowed_)
    {
        status_ = "Stop the active run before saving evaluation suites.";
        statusIsError_ = true;
        return false;
    }
    std::string error;
    bool const saved = document_.Save(error);
    statusIsError_ = !saved;
    status_ = saved ? "Suites saved." : error;
    return saved;
}

void EvaluationView::Revert()
{
    if (!editingAllowed_)
    {
        status_ = "Stop the active run before reverting evaluation suites.";
        statusIsError_ = true;
        return;
    }
    document_.Revert();
    RefreshResults();
    selected_ = std::min(selected_, document_.Suites().empty() ? 0 : document_.Suites().size() - 1);
    status_ = "Reloaded from the files.";
    statusIsError_ = false;
}

void EvaluationView::RefreshResults() noexcept
{
    bestSuite_.clear();
    evaluationFile_.clear();
}

void EvaluationView::Draw(runtime::RunLog const* journal)
{
    DrawToolbar();
    ImGui::Separator();
    DrawSuiteList();
    ImGui::SameLine();
    ImGui::BeginChild("evaluation-suite", ImVec2(0.F, 0.F));
    if (document_.Suites().empty())
    {
        Heading("No evaluation suite yet", "A suite tests a policy on independent seeds and checks its metrics.");
        MutedWrapped("To get started, select Add in the left panel. Then review the campaign and gates, "
                     "save the suite, and use Evaluate at the top of the window.");
    }
    else
    {
        selected_ = std::min(selected_, document_.Suites().size() - 1);
        editor::EditableSuite& suite = document_.Suite(selected_);
        ImGui::BeginDisabled(!editingAllowed_);
        DrawSuite(suite);
        ImGui::EndDisabled();
        DrawResults(suite, journal);
    }
    ImGui::EndChild();
}

void EvaluationView::DrawToolbar()
{
    ImGui::TextUnformatted("Evaluation suites");
    ImGui::SameLine();
    ImGui::TextDisabled("eval/<suite>.json");
    MutedWrapped("Select a suite on the left to edit it. Save changes before running Evaluate.");
    ImGui::BeginDisabled(!document_.Dirty() || !editingAllowed_);
    if (ImGui::Button("Save##suites"))
    {
        Save();
    }
    ImGui::SetItemTooltip("Write the suites; a suite ppokit-run would refuse is not saved.");
    ImGui::SameLine();
    if (ImGui::Button("Revert##suites"))
    {
        Revert();
    }
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("Drop the unsaved changes and reload the files");
    if (!status_.empty())
    {
        ImGui::SameLine();
        ImGui::TextColored(statusIsError_ ? ErrorText : MutedText, "%s", status_.c_str());
    }
}

void EvaluationView::DrawSuiteList()
{
    ImGui::BeginChild("evaluation-suites", ImVec2(ImGui::GetFontSize() * SuiteListWidthInEms, 0.F),
        ImGuiChildFlags_Borders);
    for (std::size_t index = 0; index < document_.Suites().size(); ++index)
    {
        editor::EditableSuite const& suite = document_.Suites()[index];
        std::string const label = suite.name + (suite.duringTraining ? "  (training)" : "") + "##suite"
            + std::to_string(index);
        if (ImGui::Selectable(label.c_str(), index == selected_))
        {
            selected_ = index;
        }
    }
    ImGui::Separator();
    ImGui::BeginDisabled(!editingAllowed_);
    if (ImGui::Button("Add"))
    {
        selected_ = document_.Add();
    }
    ImGui::SetItemTooltip("A new suite: ten seeds, one episode each, success_rate >= 0.9.");
    ImGui::BeginDisabled(document_.Suites().empty());
    ImGui::SameLine();
    if (ImGui::Button("Duplicate"))
    {
        selected_ = document_.Duplicate(selected_);
    }
    ImGui::SameLine();
    if (ImGui::Button("Remove"))
    {
        document_.Remove(selected_);
        selected_ = selected_ == 0 ? 0 : selected_ - 1;
    }
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    ImGui::EndChild();
}

void EvaluationView::DrawSuite(editor::EditableSuite& suite)
{
    char name[NameCapacity];
    std::snprintf(name, sizeof(name), "%s", suite.name.c_str());
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 14.F);
    if (ImGui::InputText("Name", name, sizeof(name)))
    {
        suite.name = name;
    }
    ImGui::SetItemTooltip("The file is eval/<name>.json; letters, digits and '_'.");
    DrawVerdict();
    DrawCampaign(suite);
    DrawGates(suite);
    DrawTraining(suite);
    DrawRanking(suite);
}

void EvaluationView::DrawVerdict()
{
    std::string const problem = document_.Problem(selected_);
    if (problem.empty())
    {
        ImGui::TextColored(ValidText, "ppokit-run accepts this suite.");
    }
    else
    {
        ImGui::PushTextWrapPos(0.F);
        ImGui::TextColored(ErrorText, "%s", problem.c_str());
        ImGui::PopTextWrapPos();
    }
}

void EvaluationView::DrawCampaign(editor::EditableSuite& suite)
{
    Heading("Campaign", "Every campaign plays the same episodes: same seeds, deterministic actions.");
    if (!suite.seedList.empty())
    {
        ImGui::Text("%zu seeds listed in the file.", suite.seedList.size());
        ImGui::SameLine();
        if (ImGui::SmallButton("Use a range"))
        {
            suite.firstSeed = suite.seedList.front();
            suite.seedCount = suite.seedList.size();
            suite.seedList.clear();
        }
        ImGui::SetItemTooltip("Replace the list by consecutive seeds from its first one.");
    }
    else
    {
        int first = static_cast<int>(std::min<std::uint32_t>(suite.firstSeed, INT32_MAX));
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 8.F);
        if (ImGui::InputInt("First seed", &first))
        {
            suite.firstSeed = static_cast<std::uint32_t>(std::max(first, 0));
        }
        ImGui::SetItemTooltip("Seeds training never uses: the episodes are new to the policy.");
        ImGui::SameLine();
        CountInput("Seeds", suite.seedCount, 1);
    }
    CountInput("Episodes per seed", suite.episodesPerSeed, 1);
    ImGui::SetItemTooltip("The environment is created once per seed; each episode knows its place in the "
                          "campaign, so the starts can be spread over it.");
    ImGui::TextDisabled("%zu episodes per campaign.", suite.Seeds().size() * suite.episodesPerSeed);
}

bool EvaluationView::MetricCombo(char const* id, std::string& metric) const
{
    bool changed = false;
    ImGui::SetNextItemWidth(ImGui::GetFontSize() * 12.F);
    if (ImGui::BeginCombo(id, metric.c_str()))
    {
        for (std::string const& name : document_.MetricNames())
        {
            if (ImGui::Selectable(name.c_str(), name == metric))
            {
                metric = name;
                changed = true;
            }
        }
        ImGui::EndCombo();
    }
    return changed;
}

void EvaluationView::DrawConditions(char const* id, std::vector<editor::EditableGate>& conditions)
{
    ImGui::PushID(id);
    std::size_t removed = conditions.size();
    for (std::size_t index = 0; index < conditions.size(); ++index)
    {
        editor::EditableGate& gate = conditions[index];
        ImGui::PushID(static_cast<int>(index));
        MetricCombo("##metric", gate.metric);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 4.F);
        if (ImGui::BeginCombo("##comparison", gate.atLeast ? ">=" : "<="))
        {
            if (ImGui::Selectable(">=", gate.atLeast))
            {
                gate.atLeast = true;
            }
            if (ImGui::Selectable("<=", !gate.atLeast))
            {
                gate.atLeast = false;
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 7.F);
        ImGui::InputDouble("##threshold", &gate.threshold, 0.0, 0.0, "%.4g");
        ImGui::SameLine();
        if (ImGui::SmallButton("Remove"))
        {
            removed = index;
        }
        ImGui::PopID();
    }
    if (removed < conditions.size())
    {
        conditions.erase(conditions.begin() + static_cast<std::ptrdiff_t>(removed));
    }
    if (ImGui::SmallButton("Add a condition"))
    {
        conditions.push_back({document_.MetricNames().back(), true, 0.9});
    }
    ImGui::SetItemTooltip("m.rate: share of episodes where the flag m is set; m.mean: mean of m over the "
                          "episodes where it applies.");
    ImGui::PopID();
}

void EvaluationView::DrawGates(editor::EditableSuite& suite)
{
    Heading("Gates", "The campaign passes when every condition holds; a passing campaign may become the BEST.");
    DrawConditions("gates", suite.gates);
}

void EvaluationView::DrawTraining(editor::EditableSuite& suite)
{
    Heading("During training", "A campaign every N updates and after the last one, on the policy just saved.");
    ImGui::Checkbox("Run during training", &suite.duringTraining);
    if (suite.duringTraining)
    {
        ImGui::SameLine();
        CountInput("Every updates", suite.everyUpdates, 1);
        ImGui::TextUnformatted("Goal");
        ImGui::SameLine();
        ImGui::TextDisabled("(further conditions for the training to be finished; none: the gates are enough)");
        DrawConditions("goal", suite.goal);
        ImGui::TextDisabled("ppokit-run train --until-passed %s stops at the first campaign that passes and "
                            "reaches the goal.",
            suite.name.c_str());
    }
}

void EvaluationView::DrawRanking(editor::EditableSuite& suite)
{
    Heading("BEST", "A passing campaign replaces the BEST when it wins on the first criterion that differs.");
    std::size_t removed = suite.ranking.size();
    for (std::size_t index = 0; index < suite.ranking.size(); ++index)
    {
        editor::EditableRank& rank = suite.ranking[index];
        ImGui::PushID(static_cast<int>(index) + 1000);
        ImGui::Text("%zu.", index + 1);
        ImGui::SameLine();
        MetricCombo("##metric", rank.metric);
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::GetFontSize() * 6.F);
        if (ImGui::BeginCombo("##prefer", rank.higher ? "higher" : "lower"))
        {
            if (ImGui::Selectable("higher", rank.higher))
            {
                rank.higher = true;
            }
            if (ImGui::Selectable("lower", !rank.higher))
            {
                rank.higher = false;
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Up"))
        {
            Move(suite.ranking, index, -1);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Down"))
        {
            Move(suite.ranking, index, 1);
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Remove"))
        {
            removed = index;
        }
        ImGui::PopID();
    }
    if (removed < suite.ranking.size())
    {
        suite.ranking.erase(suite.ranking.begin() + static_cast<std::ptrdiff_t>(removed));
    }
    if (ImGui::SmallButton("Add a criterion"))
    {
        suite.ranking.push_back({"success_rate", true});
    }
    if (suite.ranking.empty())
    {
        ImGui::TextDisabled("Without criteria, the suite keeps no BEST.");
    }
}

void EvaluationView::DrawResults(editor::EditableSuite const& suite, runtime::RunLog const* journal)
{
    auto const file = eval::EvaluationReportPath(checkpoints_, suite.name);
    if (file.empty()) return;
    ImGui::SeparatorText("Results");
    if (ImGui::SmallButton("Refresh results")) RefreshResults();
    DrawSavedEvaluation(file, suite.name);
    DrawBest(suite.name);
    std::vector<runtime::EvaluationRecord const*> campaigns;
    if (journal != nullptr)
    {
        for (auto record = journal->evaluations.rbegin(); record != journal->evaluations.rend(); ++record)
        {
            if (record->suite == suite.name && campaigns.size() < ShownCampaigns)
            {
                campaigns.push_back(&*record);
            }
        }
    }
    if (campaigns.empty())
    {
        if (!savedEvaluation_) ImGui::TextDisabled("No training campaign recorded for this suite.");
        return;
    }
    int const columns = 4 + static_cast<int>(campaigns.front()->gates.size());
    if (!ImGui::BeginTable("evaluation-campaigns", columns, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg))
    {
        return;
    }
    ImGui::TableSetupColumn("Update");
    ImGui::TableSetupColumn("Verdict");
    ImGui::TableSetupColumn("BEST");
    ImGui::TableSetupColumn("Goal");
    for (runtime::EvaluationGate const& gate : campaigns.front()->gates)
    {
        ImGui::TableSetupColumn((gate.metric + " " + gate.comparison + " " + ValueText(gate.threshold)).c_str());
    }
    ImGui::TableHeadersRow();
    for (runtime::EvaluationRecord const* campaign : campaigns)
    {
        ImGui::TableNextRow();
        ImGui::TableNextColumn();
        ImGui::Text("%llu", static_cast<unsigned long long>(campaign->update));
        ImGui::TableNextColumn();
        ImGui::TextColored(campaign->passed ? ValidText : ErrorText, campaign->passed ? "passed" : "not passed");
        ImGui::TableNextColumn();
        ImGui::TextUnformatted(campaign->promoted ? "new BEST" : "");
        ImGui::TableNextColumn();
        if (!campaign->goal.empty())
        {
            ImGui::TextColored(campaign->goalReached ? ValidText : MutedText, campaign->goalReached ? "reached" : "not yet");
        }
        for (runtime::EvaluationGate const& gate : campaign->gates)
        {
            ImGui::TableNextColumn();
            ImGui::TextColored(gate.passed ? ValidText : ErrorText, "%s", ValueText(gate.value).c_str());
        }
    }
    ImGui::EndTable();
}

void EvaluationView::DrawSavedEvaluation(std::filesystem::path const& file, std::string const& suite)
{
    std::error_code error;
    auto const time = std::filesystem::last_write_time(file, error);
    if (evaluationFile_ != file || evaluationTime_ != time)
    {
        evaluationFile_ = file;
        evaluationTime_ = time;
        savedEvaluation_.reset();
        evaluationError_.clear();
        if (!error)
        {
            try
            {
                auto result = eval::ReadSavedEvaluation(file);
                if (result.report.suite != suite)
                    throw std::runtime_error(file.string() + " belongs to suite " + result.report.suite);
                savedEvaluation_ = std::move(result);
            }
            catch (std::exception const& failure) { evaluationError_ = failure.what(); }
        }
        else if (error != std::errc::no_such_file_or_directory) evaluationError_ = file.string() + ": " + error.message();
    }
    if (!evaluationError_.empty()) ImGui::TextColored(ErrorText, "%s", evaluationError_.c_str());
    if (!savedEvaluation_) return;
    auto const& result = *savedEvaluation_;
    ImGui::Text("Last completed evaluation  %s  (%zu episodes)", result.report.Passed() ? "passed" : "failed",
        result.episodeCount);
    if (!result.policyRole.empty()) ImGui::Text("Policy  %s", result.policyRole.c_str());
    if (result.update) ImGui::Text("update %llu", static_cast<unsigned long long>(*result.update));
    ImGui::Text("Mean return  %s", ValueText(result.report.meanReturn).c_str());
    for (auto const& gate : result.report.gates)
        ImGui::TextColored(gate.passed ? ValidText : ErrorText, "%s  %s  (%s %s)", gate.gate.metric.c_str(),
            ValueText(gate.value).c_str(), gate.gate.comparison == eval::Comparison::GreaterOrEqual ? ">=" : "<=",
            ValueText(gate.gate.threshold).c_str());
    if (!result.report.policySha256.empty()) ImGui::TextWrapped("Evaluated SHA-256  %s", result.report.policySha256.c_str());
}

void EvaluationView::RefreshBest(std::string const& suite)
{
    auto const time = FileTime(eval::BestManifest(checkpoints_, suite));
    bool const hasHistorical = historicalBest_ && historicalBest_->suite == suite;
    if (suite == bestSuite_ && time == bestTime_ && FileTime(bestFile_) == bestFileTime_)
    {
        return;
    }
    if (suite != bestSuite_) bestFile_.clear();
    bestSuite_ = suite;
    bestTime_ = time;
    bestError_.clear();
    try
    {
        best_ = eval::ReadBest(checkpoints_, suite);
        if (best_) bestFile_ = best_->policy;
        else if (hasHistorical)
        {
            bestFile_ = historicalBest_->file;
            if (artifacts::Sha256OfFile(bestFile_) != historicalBest_->sha256)
                throw std::runtime_error(bestFile_.string() + ": HISTORICAL_BEST SHA-256 differs");
        }
        else bestFile_.clear();
    }
    catch (std::exception const& failure)
    {
        best_.reset();
        bestError_ = failure.what();
    }
    bestFileTime_ = FileTime(bestFile_);
}

void EvaluationView::DrawBest(std::string const& suite)
{
    RefreshBest(suite);
    if (!bestError_.empty())
    {
        ImGui::TextColored(ErrorText, "BEST: %s", bestError_.c_str());
        return;
    }
    if (!best_)
    {
        if (historicalBest_ && historicalBest_->suite == suite)
        {
            ImGui::Text("Packaged BEST  %s", historicalBest_->role.c_str());
            if (historicalBest_->sourceUpdate)
                ImGui::Text("update %llu", static_cast<unsigned long long>(*historicalBest_->sourceUpdate));
            ImGui::TextWrapped("%s", historicalBest_->file.string().c_str());
        }
        else ImGui::TextDisabled("No BEST policy yet.");
        return;
    }
    std::string values;
    for (eval::NamedValue const& aggregate : best_->report.aggregates)
    {
        values += "  " + aggregate.name + " " + ValueText(aggregate.value);
    }
    ImGui::Text("BEST: update %llu, success rate %s%s", static_cast<unsigned long long>(best_->update),
        ValueText(best_->report.successRate).c_str(), values.c_str());
    ImGui::SetItemTooltip("%s", best_->policy.string().c_str());
}

} // namespace ppokit::studio
