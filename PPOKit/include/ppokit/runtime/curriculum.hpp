/**
 * @file PPOKit/include/ppokit/runtime/curriculum.hpp
 * @brief Runs a curriculum from train.json: draws episode scenarios and promotes stages.
 * @details
 * @code
 * ppokit::runtime::Curriculum curriculum(*plan.curriculum);
 * std::vector<double> scenario;                       // reused for every episode
 * auto const draw = curriculum.Sample(rng, scenario); // values in task.json order
 * // ... play one episode with this scenario ...
 * if (curriculum.Record(draw, success)) { log("promoted to " + curriculum.StageName()); }
 * @endcode
 * Promotion (PromotionWindow, shared with FlightControl3D): once a window of episodes of the current stage is
 * complete, the stage is promoted if their success rate reaches the stage threshold, and a
 * new empty window starts either way. Rehearsal episodes replay an earlier stage and never
 * count for promotion. The last stage is the full task: it is never left.
 */
#pragma once

#include "ppokit/runtime/promotion_window.hpp"
#include "ppokit/schema/train_schema.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace ppokit::runtime {

/** @brief Event that can validate a curriculum stage. */
enum class PromotionSource : std::uint8_t
{
    Episode,
    Evaluation,
    Updates,
};

/** @brief Project-neutral promotion criterion owned by the common curriculum engine. */
struct PromotionCriterion final
{
    PromotionSource source = PromotionSource::Episode;
    std::size_t window = 1;
    double threshold = 1.0;
    std::uint64_t updates = 0;
    std::string suite;
    std::string metric;
};

/** @brief One common stage; payload is interpreted only by the environment adapter. */
struct OrchestrationStage final
{
    std::string name;
    std::string payload = "{}";
    PromotionCriterion promotion;
    bool autoAdvance = true;
    double maximumSeconds = 0.0;
    /** @brief Optional smooth weighted-round-robin shares, indexed by plan stage. */
    std::vector<std::uint32_t> rehearsalWeights;
};

/** @brief Ordered stages plus deterministic rehearsal scheduling. */
struct OrchestrationPlan final
{
    std::string id;
    std::vector<OrchestrationStage> stages;
    std::size_t rehearsalEveryBlocks = 0;
    std::uint64_t blockUpdates = 20;
    std::uint64_t primaryUpdates = 0;
    std::uint64_t priorUpdates = 0;
};

/** @brief Result of a curriculum event, including the reason shown by CLI and Studio. */
struct CurriculumDecision final
{
    bool changed = false;
    bool completed = false;
    std::string reason;
};

/**
 * @brief Single state machine used by episode, evaluation and update-budget curricula.
 * @details It owns promotion, rehearsal selection, reasons and exact persistence. A project
 *          adapter only applies StagePayload() and reports domain outcomes.
 */
class CurriculumEngine final
{
public:
    /** @throws std::invalid_argument for an invalid plan. */
    explicit CurriculumEngine(OrchestrationPlan plan);

    std::size_t Stage() const noexcept { return stage_; }
    std::size_t ActiveStage() const noexcept { return activeStage_; }
    std::size_t Stages() const noexcept { return plan_.stages.size(); }
    std::string const& StageName() const noexcept { return plan_.stages[stage_].name; }
    std::string const& ActiveStageName() const noexcept { return plan_.stages[activeStage_].name; }
    std::string const& StagePayload() const noexcept { return plan_.stages[activeStage_].payload; }
    std::string const& Payload(std::size_t stage) const { return plan_.stages.at(stage).payload; }
    PromotionCriterion const& Criterion() const noexcept { return plan_.stages[stage_].promotion; }
    double MaximumSeconds() const noexcept { return plan_.stages[stage_].maximumSeconds; }
    bool AtFinalStage() const noexcept { return stage_ + 1 == plan_.stages.size(); }
    bool Completed() const noexcept { return completed_; }
    std::size_t WindowEpisodes() const noexcept { return window_.Episodes(); }
    std::size_t WindowSuccesses() const noexcept { return window_.Successes(); }
    std::uint64_t StageStartUpdate() const noexcept { return stageStartUpdate_; }
    std::string const& LastReason() const noexcept { return lastReason_; }
    std::string const& LastEvaluation() const noexcept { return lastEvaluation_; }

    /** @brief Records an eligible or ignored training episode. */
    CurriculumDecision RecordEpisode(bool eligible, bool success);
    /** @brief Records an optimizer boundary and selects the next deterministic rehearsal block. */
    CurriculumDecision RecordUpdate(std::uint64_t update);
    /** @brief Records an independent evaluation campaign and its named aggregates. */
    CurriculumDecision RecordEvaluation(std::string const& suite,
        std::vector<std::pair<std::string, double>> const& metrics, bool passed);

    /** @brief Starts at an exact stage/window/update state. */
    void Restore(std::size_t stage, std::size_t episodes, std::size_t successes,
        std::uint64_t stageStartUpdate = 0, std::uint64_t lastUpdate = 0, bool completed = false);
    std::optional<std::size_t> StageIndex(std::string const& name) const;
    /** @brief Versioned state used in the publication generation. */
    std::string StateJson() const;
    /** @brief Restores StateJson and rejects a state belonging to another plan. */
    void RestoreJson(std::string const& state);
    /** @brief Human-readable, versioned snapshot consumed by the Studio. */
    std::string StatusJson() const;

private:
    CurriculumDecision Promote(std::string reason);
    void SelectActiveStage(std::uint64_t update);
    std::uint64_t Fingerprint() const;

    OrchestrationPlan plan_;
    std::size_t stage_ = 0;
    std::size_t activeStage_ = 0;
    PromotionWindow window_;
    std::uint64_t stageStartUpdate_ = 0;
    std::uint64_t lastUpdate_ = 0;
    bool completed_ = false;
    std::string lastReason_ = "Curriculum has not received a result yet.";
    std::string lastEvaluation_;
};

/**
 * @brief Loads the canonical curriculum schema and the two historical Blue formats.
 * @param path Configuration file (`ppokit.curriculum/1`, adaptive gates v2, or V2/V3 quotas).
 */
OrchestrationPlan LoadOrchestrationPlan(std::filesystem::path const& path);

/** @brief Which stage produced a scenario, to be handed back to Curriculum::Record. */
struct ScenarioDraw final
{
    std::size_t stage = 0;
    /** @brief True when an earlier stage was replayed; such episodes do not count for promotion. */
    bool rehearsal = false;
};

/** @brief Curriculum state: current stage and promotion window. */
class Curriculum final
{
public:
    /** @throws std::invalid_argument for a plan without stages. */
    explicit Curriculum(schema::CurriculumPlan plan);

    std::size_t Stage() const noexcept { return progress_.Stage(); }
    std::size_t Stages() const noexcept { return plan_.stages.size(); }
    std::string const& StageName() const noexcept { return plan_.stages[progress_.Stage()].name; }
    bool AtFinalStage() const noexcept { return progress_.AtFinalStage(); }
    std::size_t WindowEpisodes() const noexcept { return progress_.WindowEpisodes(); }
    std::size_t WindowSuccesses() const noexcept { return progress_.WindowSuccesses(); }

    /**
     * @brief Draws the scenario of the next episode.
     * @param values Receives one value per scenario parameter, in task.json order (resized once).
     */
    template<typename Engine>
    ScenarioDraw Sample(Engine& rng, std::vector<double>& values) const
    {
        ScenarioDraw draw;
        draw.stage = progress_.Stage();
        if (draw.stage > 0 && plan_.rehearsal > 0 && std::bernoulli_distribution(plan_.rehearsal)(rng))
        {
            draw.stage = std::uniform_int_distribution<std::size_t>(0, draw.stage - 1)(rng);
            draw.rehearsal = true;
        }
        auto const& samplers = plan_.stages[draw.stage].samplers;
        values.resize(samplers.size());
        for (std::size_t index = 0; index < samplers.size(); ++index)
        {
            values[index] = Draw(samplers[index], rng);
        }
        return draw;
    }

    /**
     * @brief Records the outcome of an episode drawn by Sample.
     * @return True when this episode completes a window that promotes the stage.
     */
    bool Record(ScenarioDraw const& draw, bool success);

    /** @brief Restores a stage (e.g. after resuming training), with an empty window; clamped to the last stage. */
    void Restore(std::size_t stage) noexcept;

    /** @brief Restores an exact stage and its incomplete promotion window. */
    void Restore(std::size_t stage, std::size_t episodes, std::size_t successes);

    /** @brief Index of a stage by name. @return Nothing for an unknown name. */
    std::optional<std::size_t> StageIndex(std::string const& name) const;

private:
    /** @brief One draw of one parameter; whole numbers for int, bool and enum parameters. */
    template<typename Engine>
    static double Draw(schema::ParameterSampler const& sampler, Engine& rng)
    {
        switch (sampler.kind)
        {
        case schema::SamplerKind::Fixed:
            return sampler.value;
        case schema::SamplerKind::Uniform:
            if (sampler.integral)
            {
                auto const low = static_cast<long long>(sampler.low);
                auto const high = static_cast<long long>(sampler.high);
                return static_cast<double>(std::uniform_int_distribution<long long>(low, high)(rng));
            }
            return std::uniform_real_distribution<double>(sampler.low, sampler.high)(rng);
        case schema::SamplerKind::Normal:
        {
            double const drawn = std::normal_distribution<double>(sampler.mean, sampler.deviation)(rng);
            double const clamped = std::clamp(drawn, sampler.minimum, sampler.maximum);
            return sampler.integral ? std::round(clamped) : clamped;
        }
        case schema::SamplerKind::Choice:
            break;
        }
        std::vector<double> weights;
        weights.reserve(sampler.choices.size());
        for (auto const& choice : sampler.choices)
        {
            weights.push_back(choice.weight);
        }
        std::discrete_distribution<std::size_t> pick(weights.begin(), weights.end());
        return sampler.choices[pick(rng)].value;
    }

    schema::CurriculumPlan plan_;
    CurriculumEngine progress_;
};

/**
 * @brief Saves the stage name and incomplete promotion window atomically.
 * @details Stage names remain robust to stages added later. Sampling owns no RNG; its caller's
 *          checkpointed engine is the random state required for exact continuation.
 */
void SaveCurriculumStage(std::filesystem::path const& path, Curriculum const& curriculum);

/**
 * @brief Restores the stage and promotion window saved by SaveCurriculumStage.
 * @details Historical files containing only a stage name remain accepted with an empty window.
 * @return False when the file does not exist (a new run starts at the first stage).
 * @throws std::runtime_error when the saved stage is not a stage of this curriculum.
 */
bool LoadCurriculumStage(std::filesystem::path const& path, Curriculum& curriculum);

} // namespace ppokit::runtime
