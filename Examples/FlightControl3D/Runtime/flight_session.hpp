#pragma once
#include "flight_environment.hpp"
#include "flight_learner.hpp"
#include "flight_model_store.hpp"

namespace ppo::examples::flight {

enum class TrainingBudget { TotalUpdates, AdditionalUpdates };

/** Value-only render snapshot; the renderer never reads FDM or network objects. */
struct FlightSnapshot final
{
    Runway runway{};
    FlightPlan plan{};
    AircraftState aircraft{};
    MissionState mission{}, lastEpisode{};
    Guidance guidance{};
    Controls controls{};
    StepResult lastStep{};
    LearningStats learning{};
    std::uint64_t episodes{}, touchdowns{}, stops{}, crashes{}, timeouts{};
    double glideDegrees{}, ilsRange{}, stepSeconds{};
    /** @brief Physical extension threshold used by the camera GEAR indicator. */
    double gearDownMinimum{};
    bool training{}, finished{}, failed{};
    /** @brief Current limit and last duration in simulated, not wall-clock, seconds. */
    double elapsedSeconds{},maximumEpisodeSeconds{},lastEpisodeSeconds{};
    ValidationMetrics validation{},lastValidation{};
    std::uint64_t validationCampaigns{};
    double bestScore{};
    bool validating{},hasBest{};
    double localizerCaptureDegrees{},glideCaptureDegrees{};
    std::uint64_t trainingUpdateLimit{};
    bool resumed{};
    CurriculumStage curriculumStage{CurriculumStage::Joint};
    std::size_t gateSuccesses{},gateEvaluated{};
};

/** True when a captured beam is also stable in heading, route and slope. */
bool CaptureGateQualified(Config const&,CurriculumStage,MissionState const&,
    AircraftState const&,Guidance const&) noexcept;

/** Shared orchestration for the current desktop framework and CPU headless runner. */
class FlightSession final
{
public:
    FlightSession(Config config, bool training, std::filesystem::path modelDirectory,
        std::optional<std::size_t> inferenceEpisodes = std::nullopt,
        TrainingBudget budget = TrainingBudget::TotalUpdates);
    void Tick();
    void ResetEpisode();
    /** Extend a completed desktop block without reinitializing the learned policy. */
    void ContinueTraining();
    FlightSnapshot Snapshot() const noexcept;
    bool Finished() const noexcept { return finished_; }
    FlightLearner const& Learner() const noexcept { return learner_; }
private:
    Config config_;
    bool training_{};
    FlightModelStore models_;
    FlightEnvironment environment_;
    FlightLearner learner_;
    Controls controls_{};
    StepResult lastStep_{};
    MissionState lastEpisode_{};
    std::uint64_t episodes_{}, touchdowns_{}, stops_{}, crashes_{}, timeouts_{};
    bool pendingReset_{}, finished_{};
    std::uint64_t trainingUpdateLimit_{};
    bool resumed_{};
    std::unique_ptr<FlightEnvironment> validationEnvironment_;
    ValidationMetrics validation_{},lastValidation_{};
    std::optional<ValidationMetrics> best_;
    std::uint64_t validationCampaigns_{};
    std::size_t inferenceEpisodeLimit_{};
    bool validationPendingReset_{};
    double validationHoldSeconds_{};
    bool validationEpisodeSucceeded_{};
    std::size_t validationGateSuccesses_{},lastGateSuccesses_{},lastGateEvaluated_{};
    void StartValidation();
    void TickValidation();
};
} // namespace ppo::examples::flight
