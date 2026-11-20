#pragma once
#include "flight_curriculum.hpp"
#include "flight_environment.hpp"
#include "flight_learner.hpp"
#include "flight_model_store.hpp"
#include "flight_reward_tuning.hpp"

namespace ppo::examples::flight {

enum class TrainingBudget { TotalUpdates, AdditionalUpdates };

/** Fixed-capacity convergence diagnostics; populated only when enabled in JSON. */
struct FlightDiagnostics final
{
    bool enabled{};
    std::uint64_t samples{};
    Observation observation{};
    PolicyVector policy{};
    LogStandardDeviations logStandardDeviations{};
    std::array<double,ContinuousActions> rawAction{},appliedAction{};
    std::array<double,ObservationSize> observationAbsoluteSums{},observationMaximumAbsolute{};
    std::array<std::uint64_t,ObservationSize> observationSaturations{};
    std::array<double,PolicyOutputs> policyAbsoluteSums{},policyMaximumAbsolute{};
    LogStandardDeviations logStandardDeviationAbsoluteSums{},
        logStandardDeviationMaximumAbsolute{};
    std::array<double,ContinuousActions> rawAbsoluteSums{},appliedAbsoluteSums{};
    std::array<std::uint64_t,ContinuousActions> rawSaturations{},appliedSaturations{};
    std::array<double,RewardCount> rewardSums{};
    double rewardSum{},rewardMinimum{},rewardMaximum{};
};

/** Value-only render snapshot; the renderer never reads FDM or network objects. */
struct FlightSnapshot final
{
    Runway runway{};
    AircraftState aircraft{};
    MissionState mission{}, lastEpisode{};
    Guidance guidance{};
    Controls controls{};
    StepResult lastStep{};
    LearningStats learning{};
    std::uint64_t episodes{}, touchdowns{}, stops{}, crashes{}, timeouts{};
    double glideDegrees{}, stepSeconds{};
    /** @brief Touchdown aim point, in metres past the threshold. */
    double aimPointAlongMeters{};
    /** @brief Physical extension threshold used by the camera GEAR indicator. */
    double gearDownMinimum{};
    bool training{}, finished{}, failed{};
    /** @brief Current limit and last duration in simulated, not wall-clock, seconds. */
    double elapsedSeconds{},maximumEpisodeSeconds{},lastEpisodeSeconds{};
    ValidationMetrics validation{},lastValidation{};
    std::uint64_t validationCampaigns{};
    double bestScore{};
    /** @brief Curriculum difficulty and PPO update of the BEST validation campaign. */
    double bestDifficulty{};
    std::uint64_t bestUpdate{};
    bool validating{},hasBest{};
    std::uint64_t trainingUpdateLimit{};
    bool resumed{};
    /** @brief Stops and episodes of the last complete independent validation. */
    std::size_t gateSuccesses{},gateEvaluated{};
    /** @brief Training curriculum level, level count and current promotion window. */
    std::size_t curriculumLevel{},curriculumLevels{},curriculumWindowStops{},curriculumWindowEpisodes{};
    FlightDiagnostics diagnostics{},lastDiagnostics{};
    /** @brief Sea level in the local world frame, in metres. */
    double seaHeightMeters{};
    /** @brief Latest crash event and its physical pose, retained across episode resets. */
    std::uint64_t crashEvent{};
    AircraftState crashAircraft{};
    Runway crashRunway{};
    /** @brief Most recent crash details, including non-finite state failures. */
    CrashDiagnostic crashDiagnostic{};
    bool hasCrashDiagnostic{};
    /** @brief Whether the snapshot holds a finite position for the approach profile. */
    bool approachProfileValid{};
    /** @brief Requested values, applied at the next PPO boundary when pending. */
    RewardSettings rewardSettings{};
    /** @brief Reward weights currently applied to collected transitions. */
    RewardSettings activeRewardSettings{};
    bool rewardTuningPending{};
    std::array<char,192> rewardTuningStatus{};
    /** @brief Per-control acknowledgements, including rejected edits. */
    std::array<std::uint64_t,RewardTuningCount> rewardTuningRevisions{};
    /** @brief Sum of each reward during the currently displayed episode. */
    std::array<double,RewardCount> episodeRewardSums{};
};

/** Shared orchestration for the current desktop framework and CPU headless runner. */
class FlightSession final
{
public:
    /**
     * @brief Opens a training or evaluation session on one model directory.
     * @param config Validated scenario.
     * @param training True to train and validate; false to evaluate the validated BEST.
     * @param modelDirectory Directory holding latest and BEST checkpoints.
     * @param inferenceEpisodes Evaluation budget, defaulting to the configured validation episodes.
     * @param budget Whether maxUpdates is a total or an additional update budget.
     * @param scenarioPath Scenario file updated by the reward controls; empty disables them.
     */
    FlightSession(Config config, bool training, std::filesystem::path modelDirectory,
        std::optional<std::size_t> inferenceEpisodes = std::nullopt,
        TrainingBudget budget = TrainingBudget::TotalUpdates,
        std::filesystem::path scenarioPath = {});
    void Tick();
    void ResetEpisode();
    /** Extend a completed desktop block without reinitializing the learned policy. */
    void ContinueTraining();
    FlightSnapshot Snapshot() const noexcept;
    bool Finished() const noexcept { return finished_; }
    FlightLearner const& Learner() const noexcept { return learner_; }
    /** @brief Save an existing reward coefficient and queue it for a safe boundary. */
    bool SetRewardTuning(std::size_t index,double value) noexcept;
private:
    Config config_;
    bool training_{};
    std::filesystem::path modelDirectory_;
    std::filesystem::path scenarioPath_;
    std::optional<Config> pendingRewardConfig_;
    std::array<char,192> rewardTuningStatus_{};
    std::array<std::uint64_t,RewardTuningCount> rewardTuningRevisions_{};
    FlightModelStore models_;
    FlightCurriculum curriculum_;
    std::unique_ptr<FlightEnvironment> environment_;
    FlightLearner learner_;
    Controls controls_{};
    StepResult lastStep_{};
    MissionState lastEpisode_{};
    std::uint64_t episodes_{}, touchdowns_{}, stops_{}, crashes_{}, timeouts_{};
    std::uint64_t crashEvent_{};
    AircraftState crashAircraft_{};
    Runway crashRunway_{};
    CrashDiagnostic crashDiagnostic_{};
    bool hasCrashDiagnostic_{};
    bool pendingReset_{}, finished_{};
    std::uint64_t trainingUpdateLimit_{};
    bool resumed_{};
    std::unique_ptr<FlightEnvironment> validationEnvironment_;
    ValidationMetrics validation_{},lastValidation_{};
    std::optional<ValidationMetrics> best_;
    std::uint64_t validationCampaigns_{};
    std::size_t inferenceEpisodeLimit_{};
    bool validationPendingReset_{};
    FlightDiagnostics diagnostics_{},lastDiagnostics_{};
    std::array<double,RewardCount> episodeRewardSums_{},validationEpisodeRewardSums_{};
    /**
     * @brief Retains terminal crash details and any finite physical pose across episode resets.
     * @param environment Active training or validation environment after its terminal step.
     */
    void CaptureCrash(FlightEnvironment const& environment) noexcept;
    /** @brief Apply saved settings without mixing reward contracts within a PPO rollout. */
    void ApplyPendingRewardTuning();
    /** @brief Freezes the learner and starts an independent validation campaign. */
    void StartValidation();
    /** @brief Starts the next reproducible validation episode of the curriculum-level mix. */
    void ResetValidationEpisode();
    /** @brief Starts the next reproducible inference episode of the full-task mix. */
    void ResetInferenceEpisode();
    void TickValidation();
    /** @brief Records a completed validation, promotes BEST and decides whether training ends. */
    void FinishValidation();
    /** @brief Feeds a finished airborne training episode to the curriculum and applies a promotion. */
    void AdvanceCurriculum(MissionState const& episode);
    /** @brief Saves latest and its curriculum level together. */
    void SaveLatest();
    /** @brief Runs one PPO update and its periodic checkpoint and validation. */
    void OptimizeAndCheckpoint();
    /** @brief Rebuilds the training environment with a deterministic update-specific seed. */
    void CreateTrainingEnvironment();
};
} // namespace ppo::examples::flight
