/**
 * @file PPOKit/src/runtime/curriculum.cpp
 * @brief Promotion windows and persistence of the curriculum stage.
 */
#include "ppokit/runtime/curriculum.hpp"

#include "ppokit/artifacts/atomic_file.hpp"

#include <fstream>
#include <iomanip>
#include <iterator>
#include <cmath>
#include <limits>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ppokit::runtime {
namespace {

using Json = nlohmann::ordered_json;

std::string SourceName(PromotionSource source)
{
    switch (source)
    {
    case PromotionSource::Episode: return "episode";
    case PromotionSource::Evaluation: return "evaluation";
    case PromotionSource::Updates: return "updates";
    }
    return "unknown";
}

/** @brief Locale-independent round-trip representation used by plan fingerprints. */
std::string ExactDouble(double value)
{
    std::ostringstream text;
    text << std::setprecision(std::numeric_limits<double>::max_digits10) << value;
    return text.str();
}

PromotionSource ReadSource(std::string const& source)
{
    if (source == "episode") return PromotionSource::Episode;
    if (source == "evaluation") return PromotionSource::Evaluation;
    if (source == "updates") return PromotionSource::Updates;
    throw std::invalid_argument("PPOKit curriculum: unknown promotion source '" + source + "'");
}

double MetricValue(std::vector<std::pair<std::string, double>> const& metrics, std::string const& name,
    bool passed)
{
    if (name.empty()) return passed ? 1.0 : 0.0;
    auto const found = std::find_if(metrics.begin(), metrics.end(), [&](auto const& value) { return value.first == name; });
    if (found == metrics.end())
        throw std::invalid_argument("PPOKit curriculum: evaluation has no metric '" + name + "'");
    return found->second;
}

std::uint64_t HashText(std::uint64_t hash, std::string const& text) noexcept
{
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (unsigned char value : text)
    {
        hash ^= value;
        hash *= prime;
    }
    return hash;
}

void ValidatePlan(OrchestrationPlan const& plan)
{
    if (plan.stages.empty()) throw std::invalid_argument("PPOKit curriculum: at least one stage is required");
    for (std::size_t index = 0; index < plan.stages.size(); ++index)
    {
        OrchestrationStage const& stage = plan.stages[index];
        if (stage.name.empty()) throw std::invalid_argument("PPOKit curriculum: a stage name is empty");
        auto const duplicate = std::find_if(plan.stages.begin(), plan.stages.begin() + index,
            [&](OrchestrationStage const& other) { return other.name == stage.name; });
        if (duplicate != plan.stages.begin() + index)
            throw std::invalid_argument("PPOKit curriculum: duplicate stage '" + stage.name + "'");
        PromotionCriterion const& criterion = stage.promotion;
        if (criterion.source == PromotionSource::Updates && criterion.updates == 0)
            throw std::invalid_argument("PPOKit curriculum: stage '" + stage.name + "' has no update budget");
        if (criterion.source == PromotionSource::Episode
            && (criterion.window == 0 || !std::isfinite(criterion.threshold)
                || criterion.threshold < 0.0 || criterion.threshold > 1.0))
            throw std::invalid_argument("PPOKit curriculum: invalid episode criterion of stage '" + stage.name + "'");
        if (criterion.source == PromotionSource::Evaluation
            && (!std::isfinite(criterion.threshold) || criterion.threshold < 0.0))
            throw std::invalid_argument("PPOKit curriculum: invalid evaluation criterion of stage '" + stage.name + "'");
        if (!Json::parse(stage.payload, nullptr, false).is_object())
            throw std::invalid_argument("PPOKit curriculum: payload of stage '" + stage.name + "' is not an object");
        if (!std::isfinite(stage.maximumSeconds) || stage.maximumSeconds < 0.0)
            throw std::invalid_argument("PPOKit curriculum: invalid time budget of stage '" + stage.name + "'");
        if (!stage.rehearsalWeights.empty())
        {
            if (stage.rehearsalWeights.size() != index + 1 || stage.rehearsalWeights.back() == 0)
                throw std::invalid_argument("PPOKit curriculum: invalid rehearsal weights of stage '" + stage.name + "'");
            if (plan.blockUpdates == 0)
                throw std::invalid_argument("PPOKit curriculum: weighted rehearsals require a nonzero block size");
        }
    }
    if ((plan.primaryUpdates == 0) != (plan.priorUpdates == 0))
        throw std::invalid_argument("PPOKit curriculum: primary and prior rehearsal quotas must be set together");
    if (plan.rehearsalEveryBlocks != 0 && plan.blockUpdates == 0)
        throw std::invalid_argument("PPOKit curriculum: a rehearsal block size cannot be zero");
}

} // namespace

CurriculumEngine::CurriculumEngine(OrchestrationPlan plan) : plan_(std::move(plan))
{
    ValidatePlan(plan_);
}

CurriculumDecision CurriculumEngine::Promote(std::string reason)
{
    lastReason_ = std::move(reason);
    window_.Clear();
    if (AtFinalStage())
    {
        completed_ = true;
        activeStage_ = stage_;
        return {false, true, lastReason_};
    }
    ++stage_;
    activeStage_ = stage_;
    stageStartUpdate_ = lastUpdate_;
    return {true, false, lastReason_};
}

CurriculumDecision CurriculumEngine::RecordEpisode(bool eligible, bool success)
{
    if (completed_ || Criterion().source != PromotionSource::Episode)
        return {false, completed_, lastReason_};
    if (!eligible)
    {
        lastReason_ = "Episode ignored by the stage eligibility rule.";
        return {false, false, lastReason_};
    }
    PromotionCriterion const criterion = Criterion();
    std::size_t const decisionEpisodes = window_.Episodes() + 1;
    std::size_t const decisionSuccesses = window_.Successes() + (success ? 1 : 0);
    bool const promoted = window_.Record(success, criterion.window, criterion.threshold);
    if (promoted)
    {
        std::ostringstream reason;
        if (plan_.stages[stage_].autoAdvance)
        {
            reason << "Promoted after " << criterion.window << " eligible episodes at or above "
                   << criterion.threshold * 100.0 << "% success.";
            return Promote(reason.str());
        }
        reason << "Maintained after " << decisionSuccesses << "/" << decisionEpisodes
               << " successes: automatic promotion is disabled.";
        lastReason_ = reason.str();
        return {false, false, lastReason_};
    }
    std::ostringstream reason;
    if (decisionEpisodes == criterion.window)
        reason << "Maintained after a completed window: " << decisionSuccesses << "/" << decisionEpisodes;
    else
        reason << "Maintained: " << window_.Successes() << "/" << window_.Episodes();
    reason << " successes; validation requires " << criterion.threshold * 100.0 << "% over "
           << criterion.window << " episodes.";
    lastReason_ = reason.str();
    return {false, false, lastReason_};
}

void CurriculumEngine::SelectActiveStage(std::uint64_t update)
{
    activeStage_ = stage_;
    if (stage_ == 0) return;
    std::uint64_t const offset = update >= stageStartUpdate_ ? update - stageStartUpdate_ : 0;
    std::vector<std::uint32_t> const& weights = plan_.stages[stage_].rehearsalWeights;
    if (!weights.empty())
    {
        std::uint64_t const block = offset / plan_.blockUpdates;
        std::uint64_t total = 0;
        for (std::uint32_t const weight : weights) total += weight;
        std::vector<std::int64_t> credit(weights.size(), 0);
        for (std::uint64_t position = 0; position <= block % total; ++position)
        {
            std::size_t selected = 0;
            for (std::size_t index = 0; index < weights.size(); ++index)
            {
                credit[index] += weights[index];
                if (credit[index] > credit[selected]) selected = index;
            }
            credit[selected] -= static_cast<std::int64_t>(total);
            activeStage_ = selected;
        }
        return;
    }
    if (plan_.primaryUpdates != 0 && plan_.priorUpdates != 0)
    {
        std::uint64_t const cycle = plan_.primaryUpdates + plan_.priorUpdates;
        std::uint64_t const cycleIndex = offset / cycle;
        if (offset % cycle >= plan_.primaryUpdates)
            activeStage_ = static_cast<std::size_t>(cycleIndex % stage_);
        return;
    }
    if (plan_.rehearsalEveryBlocks != 0 && plan_.blockUpdates != 0)
    {
        std::uint64_t const block = offset / plan_.blockUpdates;
        if ((block + 1) % plan_.rehearsalEveryBlocks == 0)
            activeStage_ = static_cast<std::size_t>((block / plan_.rehearsalEveryBlocks) % stage_);
    }
}

CurriculumDecision CurriculumEngine::RecordUpdate(std::uint64_t update)
{
    if (update < lastUpdate_)
        throw std::invalid_argument("PPOKit curriculum: update counter moved backwards");
    lastUpdate_ = update;
    if (!completed_ && Criterion().source == PromotionSource::Updates
        && update - stageStartUpdate_ >= Criterion().updates && plan_.stages[stage_].autoAdvance)
    {
        std::string const reason = "Promoted after the configured budget of "
            + std::to_string(Criterion().updates) + " updates.";
        CurriculumDecision decision = Promote(reason);
        SelectActiveStage(update);
        return decision;
    }
    SelectActiveStage(update);
    if (!completed_ && Criterion().source == PromotionSource::Updates)
    {
        lastReason_ = "Maintained: " + std::to_string(update - stageStartUpdate_) + "/"
            + std::to_string(Criterion().updates) + " stage updates completed.";
    }
    return {false, completed_, lastReason_};
}

CurriculumDecision CurriculumEngine::RecordEvaluation(std::string const& suite,
    std::vector<std::pair<std::string, double>> const& metrics, bool passed)
{
    if (completed_ || Criterion().source != PromotionSource::Evaluation)
        return {false, completed_, lastReason_};
    if (!Criterion().suite.empty() && Criterion().suite != suite)
    {
        lastReason_ = "Maintained: evaluation '" + suite + "' is not the criterion suite '" + Criterion().suite + "'.";
        return {false, false, lastReason_};
    }
    double const value = MetricValue(metrics, Criterion().metric, passed);
    std::ostringstream evaluation;
    evaluation << suite << ": " << (Criterion().metric.empty() ? "passed" : Criterion().metric)
               << "=" << value << ", required=" << Criterion().threshold;
    lastEvaluation_ = evaluation.str();
    if (value >= Criterion().threshold && plan_.stages[stage_].autoAdvance)
        return Promote("Promoted by independent evaluation: " + lastEvaluation_ + ".");
    lastReason_ = "Maintained after independent evaluation: " + lastEvaluation_ + ".";
    return {false, false, lastReason_};
}

void CurriculumEngine::Restore(std::size_t stage, std::size_t episodes, std::size_t successes,
    std::uint64_t stageStartUpdate, std::uint64_t lastUpdate, bool completed)
{
    if (stage >= plan_.stages.size()) throw std::invalid_argument("PPOKit curriculum: saved stage is out of range");
    if (stageStartUpdate > lastUpdate) throw std::invalid_argument("PPOKit curriculum: stage starts after last update");
    if (completed && stage + 1 != plan_.stages.size())
        throw std::invalid_argument("PPOKit curriculum: only the final stage can be completed");
    PromotionCriterion const& criterion = plan_.stages[stage].promotion;
    if ((completed || criterion.source != PromotionSource::Episode) && (episodes != 0 || successes != 0))
        throw std::invalid_argument("PPOKit curriculum: saved stage cannot have an episode window");
    if (criterion.source == PromotionSource::Episode && episodes >= criterion.window)
        throw std::invalid_argument("PPOKit curriculum: saved episode window already reached its decision boundary");
    stage_ = stage;
    stageStartUpdate_ = stageStartUpdate;
    lastUpdate_ = lastUpdate;
    completed_ = completed;
    window_.Restore(episodes, successes);
    SelectActiveStage(lastUpdate_);
}

std::optional<std::size_t> CurriculumEngine::StageIndex(std::string const& name) const
{
    for (std::size_t index = 0; index < plan_.stages.size(); ++index)
        if (plan_.stages[index].name == name) return index;
    return std::nullopt;
}

std::uint64_t CurriculumEngine::Fingerprint() const
{
    std::uint64_t hash = 1469598103934665603ULL;
    auto field = [&hash](std::string const& value) {
        hash = HashText(hash, std::to_string(value.size()));
        hash = HashText(hash, ":");
        hash = HashText(hash, value);
    };
    field(plan_.id);
    field(std::to_string(plan_.rehearsalEveryBlocks));
    field(std::to_string(plan_.blockUpdates));
    field(std::to_string(plan_.primaryUpdates));
    field(std::to_string(plan_.priorUpdates));
    for (OrchestrationStage const& stage : plan_.stages)
    {
        field(stage.name);
        field(stage.payload);
        field(SourceName(stage.promotion.source));
        field(stage.promotion.metric);
        field(stage.promotion.suite);
        field(std::to_string(stage.promotion.window));
        field(ExactDouble(stage.promotion.threshold));
        field(std::to_string(stage.promotion.updates));
        field(stage.autoAdvance ? "1" : "0");
        field(ExactDouble(stage.maximumSeconds));
        for (std::uint32_t const weight : stage.rehearsalWeights)
            field(std::to_string(weight));
    }
    return hash;
}

std::string CurriculumEngine::StateJson() const
{
    Json const state{{"schema", "ppokit.curriculum-state/2"}, {"plan", Fingerprint()},
        {"stage", StageName()}, {"active_stage", ActiveStageName()}, {"stage_start_update", stageStartUpdate_},
        {"last_update", lastUpdate_}, {"episodes", WindowEpisodes()}, {"successes", WindowSuccesses()},
        {"completed", completed_}, {"reason", lastReason_}, {"evaluation", lastEvaluation_}};
    return state.dump();
}

void CurriculumEngine::RestoreJson(std::string const& text)
{
    Json const state = Json::parse(text);
    if (!state.is_object() || state.value("schema", std::string()) != "ppokit.curriculum-state/2")
        throw std::invalid_argument("PPOKit curriculum: not a curriculum state v2");
    if (state.at("plan").get<std::uint64_t>() != Fingerprint())
        throw std::invalid_argument("PPOKit curriculum: saved state belongs to another curriculum plan");
    auto const stage = StageIndex(state.at("stage").get<std::string>());
    if (!stage) throw std::invalid_argument("PPOKit curriculum: saved stage is absent from the plan");
    Restore(*stage, state.at("episodes").get<std::size_t>(), state.at("successes").get<std::size_t>(),
        state.at("stage_start_update").get<std::uint64_t>(), state.at("last_update").get<std::uint64_t>(),
        state.at("completed").get<bool>());
    if (state.at("active_stage").get<std::string>() != ActiveStageName())
        throw std::invalid_argument("PPOKit curriculum: saved active rehearsal stage is inconsistent");
    lastReason_ = state.value("reason", std::string{});
    lastEvaluation_ = state.value("evaluation", std::string{});
}

std::string CurriculumEngine::StatusJson() const
{
    PromotionCriterion const& criterion = Criterion();
    Json criteria{{"source", SourceName(criterion.source)}, {"threshold", criterion.threshold},
        {"window", criterion.window}, {"updates", criterion.updates}, {"suite", criterion.suite},
        {"metric", criterion.metric}, {"auto_advance", plan_.stages[stage_].autoAdvance},
        {"max_seconds", plan_.stages[stage_].maximumSeconds}};
    Json const status{{"schema", "ppokit.curriculum-status/1"}, {"plan", plan_.id}, {"stage", StageName()},
        {"active_stage", ActiveStageName()}, {"index", stage_}, {"count", plan_.stages.size()},
        {"completed", completed_}, {"window", {{"episodes", WindowEpisodes()}, {"successes", WindowSuccesses()}}},
        {"updates", {{"stage_start", stageStartUpdate_}, {"last", lastUpdate_}}}, {"criterion", criteria},
        {"last_evaluation", lastEvaluation_}, {"reason", lastReason_}};
    return status.dump(2) + '\n';
}

OrchestrationPlan LoadOrchestrationPlan(std::filesystem::path const& path)
{
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("PPOKit curriculum: cannot open '" + path.string() + "'");
    Json const root = Json::parse(stream);
    OrchestrationPlan plan;
    if (root.value("schema", std::string()) == "ppokit.curriculum/1")
    {
        plan.id = root.value("id", path.stem().string());
        plan.rehearsalEveryBlocks = root.value("rehearsal_every_blocks", std::size_t{});
        plan.blockUpdates = root.value("block_updates", std::uint64_t{20});
        for (Json const& value : root.at("stages"))
        {
            OrchestrationStage stage;
            stage.name = value.at("name").get<std::string>();
            stage.payload = value.value("payload", Json::object()).dump();
            Json const& promotion = value.at("promotion");
            stage.promotion.source = ReadSource(promotion.at("source").get<std::string>());
            stage.promotion.window = promotion.value("window", std::size_t{1});
            stage.promotion.threshold = promotion.value("threshold", 1.0);
            stage.promotion.updates = promotion.value("updates", std::uint64_t{});
            stage.promotion.suite = promotion.value("suite", std::string{});
            stage.promotion.metric = promotion.value("metric", std::string{});
            stage.autoAdvance = value.value("auto_advance", true);
            stage.maximumSeconds = value.value("max_seconds", 0.0);
            if (value.contains("rehearsal_weights"))
            {
                Json const& weights = value.at("rehearsal_weights");
                if (!weights.is_object())
                    throw std::invalid_argument("PPOKit curriculum: rehearsal_weights must be an object");
                stage.rehearsalWeights.resize(plan.stages.size() + 1, 0);
                for (std::size_t index = 0; index < plan.stages.size(); ++index)
                    stage.rehearsalWeights[index] = weights.value(plan.stages[index].name, std::uint32_t{});
                stage.rehearsalWeights.back() = weights.value(stage.name, std::uint32_t{});
            }
            plan.stages.push_back(std::move(stage));
        }
    }
    else if (root.contains("gates"))
    {
        plan.id = root.value("contractId", std::string("adaptive-gates"));
        plan.rehearsalEveryBlocks = root.value("rehearsalEveryBlocks", std::size_t{});
        for (Json const& gate : root.at("gates"))
        {
            OrchestrationStage stage;
            stage.name = gate.at("id").get<std::string>();
            Json payload{{"blue", gate.at("blue")}, {"red", gate.at("red")},
                {"hostileZones", gate.at("hostileZones")}};
            stage.payload = payload.dump();
            stage.promotion.source = PromotionSource::Evaluation;
            stage.promotion.suite = "validation";
            stage.promotion.metric = "mission_success.rate";
            stage.promotion.threshold = gate.at("thresholdPercent").get<double>() / 100.0;
            stage.autoAdvance = gate.value("autoAdvance", true);
            stage.maximumSeconds = gate.value("maxSeconds", 0.0);
            plan.stages.push_back(std::move(stage));
        }
    }
    else if (root.contains("stages") && root.contains("totalUpdates"))
    {
        plan.id = "blue-quota-v" + std::to_string(root.value("formatVersion", 0));
        if (root.contains("rehearsal"))
        {
            plan.primaryUpdates = root.at("rehearsal").at("primaryUpdates").get<std::uint64_t>();
            plan.priorUpdates = root.at("rehearsal").at("priorUpdates").get<std::uint64_t>();
        }
        for (Json const& value : root.at("stages"))
        {
            OrchestrationStage stage;
            stage.name = value.at("id").get<std::string>();
            stage.payload = Json{{"blue", value.at("blue")}, {"red", value.at("red")},
                {"hostileZones", value.at("hostileZones")}}.dump();
            stage.promotion.source = PromotionSource::Updates;
            stage.promotion.updates = value.at("updates").get<std::uint64_t>();
            plan.stages.push_back(std::move(stage));
        }
    }
    else
        throw std::invalid_argument("PPOKit curriculum: unsupported configuration '" + path.string() + "'");
    ValidatePlan(plan);
    return plan;
}

Curriculum::Curriculum(schema::CurriculumPlan plan)
    : plan_(std::move(plan)), progress_([this]() {
          OrchestrationPlan runtime;
          runtime.id = "training.json";
          for (schema::CurriculumStage const& stage : plan_.stages)
          {
              OrchestrationStage item;
              item.name = stage.name;
              item.promotion.source = PromotionSource::Episode;
              item.promotion.window = stage.window;
              item.promotion.threshold = stage.promotionRate;
              runtime.stages.push_back(std::move(item));
          }
          return runtime;
      }())
{
}

bool Curriculum::Record(ScenarioDraw const& draw, bool success)
{
    // Rehearsals and episodes drawn before a promotion say nothing about the current stage.
    if (draw.rehearsal || draw.stage != progress_.Stage() || AtFinalStage())
    {
        return false;
    }
    return progress_.RecordEpisode(true, success).changed;
}

void Curriculum::Restore(std::size_t stage) noexcept
{
    progress_.Restore(std::min(stage, plan_.stages.size() - 1), 0, 0);
}

void Curriculum::Restore(std::size_t stage, std::size_t episodes, std::size_t successes)
{
    progress_.Restore(std::min(stage, plan_.stages.size() - 1), episodes, successes);
    if (AtFinalStage() && episodes != 0)
    {
        throw std::invalid_argument("PPOKit curriculum: the final stage cannot have a promotion window");
    }
}

std::optional<std::size_t> Curriculum::StageIndex(std::string const& name) const
{
    return progress_.StageIndex(name);
}

void SaveCurriculumStage(std::filesystem::path const& path, Curriculum const& curriculum)
{
    Json const state{{"schema", "ppokit.curriculum-state/1"}, {"stage", curriculum.StageName()},
        {"episodes", curriculum.WindowEpisodes()}, {"successes", curriculum.WindowSuccesses()}};
    artifacts::WriteFileAtomically(path, [&state](std::ostream& stream) { stream << state.dump(2) << '\n'; },
        artifacts::AtomicWriteOptions{}, "PPOKit curriculum state");
}

bool LoadCurriculumStage(std::filesystem::path const& path, Curriculum& curriculum)
{
    std::ifstream stream(path);
    if (!stream)
    {
        return false;
    }
    std::string const text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    std::string name;
    std::size_t episodes = 0;
    std::size_t successes = 0;
    Json const state = Json::parse(text, nullptr, false);
    if (state.is_object())
    {
        if (state.value("schema", std::string()) != "ppokit.curriculum-state/1"
            || !state.contains("stage") || !state.at("stage").is_string()
            || !state.contains("episodes") || !state.at("episodes").is_number_unsigned()
            || !state.contains("successes") || !state.at("successes").is_number_unsigned())
        {
            throw std::runtime_error("PPOKit curriculum: invalid state in '" + path.string() + "'");
        }
        name = state.at("stage").get<std::string>();
        episodes = state.at("episodes").get<std::size_t>();
        successes = state.at("successes").get<std::size_t>();
    }
    else
    {
        std::istringstream legacy(text);
        std::string extra;
        if (!(legacy >> name) || (legacy >> extra))
        {
            throw std::runtime_error("PPOKit curriculum: '" + path.string()
                + "' must hold a curriculum state or exactly one legacy stage name");
        }
    }
    auto const index = curriculum.StageIndex(name);
    if (!index)
    {
        throw std::runtime_error("PPOKit curriculum: saved stage '" + name + "' is not in train.json");
    }
    curriculum.Restore(*index, episodes, successes);
    return true;
}

} // namespace ppokit::runtime
