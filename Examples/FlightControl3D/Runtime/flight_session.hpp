/**
 * @file flight_session.hpp
 * @brief Orchestration of training, validation and inference for the desktop and headless runners.
 *
 * Role in the chain: environment + learner + curriculum + model store -> **session** ->
 * snapshot -> renderer. One Tick() is one policy action:
 * - training: sample a stochastic action, step the environment, record the transition,
 *   feed finished airborne episodes to the curriculum, run a PPO update when the rollout is
 *   full, checkpoint, and periodically start a validation campaign;
 * - validation: the learner is frozen (deterministic actions, no Record, no RNG use) and a
 *   separate environment plays evaluation.episodes seeded episodes; the campaign may be
 *   promoted to BEST;
 * - inference: deterministic actions on the validated BEST, on the full-task evaluation mix.
 *
 * Reward weights edited from the desktop are saved to the scenario JSON at once but only
 * applied at a PPO rollout boundary, so a rollout never mixes two reward contracts.
 *
 * Not here: the physics (Simulator/), PPO math (Training/), drawing (Renderer/). The
 * renderer only reads the value-only FlightSnapshot.
 *
 * Read next: flight_session.cpp, flight_headless.cpp (log and CSV formats),
 * flight_example.cpp (desktop ECS bridge).
 */
#pragma once

#include "flight_curriculum.hpp"
#include "flight_environment.hpp"
#include "flight_learner.hpp"
#include "flight_model_store.hpp"
#include "flight_reward_tuning.hpp"

namespace ppo::examples::flight {

/** @brief Whether TrainingSettings::maxUpdates is a total or an additional update budget. */
enum class TrainingBudget { TotalUpdates, AdditionalUpdates };

/** @brief Fixed-capacity convergence diagnostics; populated only when enabled in JSON. */
struct FlightDiagnostics final
{
    bool enabled{};
    /** @brief Actions recorded in the current window. */
    std::uint64_t samples{};
    /** @brief Live values of the last action, shown by the renderer. */
    Observation observation{};
    PolicyVector policy{};
    LogStandardDeviations logStandardDeviations{};
    std::array<double, ContinuousActions> rawAction{};
    std::array<double, ContinuousActions> appliedAction{};
    /** @brief Window statistics per observation slot. */
    std::array<double, ObservationSize> observationAbsoluteSums{};
    std::array<double, ObservationSize> observationMaximumAbsolute{};
    std::array<std::uint64_t, ObservationSize> observationSaturations{};
    /** @brief Window statistics per actor output. */
    std::array<double, PolicyOutputs> policyAbsoluteSums{};
    std::array<double, PolicyOutputs> policyMaximumAbsolute{};
    LogStandardDeviations logStandardDeviationAbsoluteSums{};
    LogStandardDeviations logStandardDeviationMaximumAbsolute{};
    /** @brief Window statistics per action axis, raw (latent) and applied (filtered physical). */
    std::array<double, ContinuousActions> rawAbsoluteSums{};
    std::array<double, ContinuousActions> appliedAbsoluteSums{};
    std::array<std::uint64_t, ContinuousActions> rawSaturations{};
    std::array<std::uint64_t, ContinuousActions> appliedSaturations{};
    /** @brief Window reward statistics. */
    std::array<double, RewardCount> rewardSums{};
    double rewardSum{};
    double rewardMinimum{};
    double rewardMaximum{};
};

/** @brief Value-only render snapshot; the renderer never reads FDM or network objects. */
struct FlightSnapshot final
{
    Runway runway{};
    AircraftState aircraft{};
    /** @brief Current and last finished episode states. */
    MissionState mission{};
    MissionState lastEpisode{};
    Guidance guidance{};
    /** @brief Filtered controls of the last action. */
    Controls controls{};
    StepResult lastStep{};
    LearningStats learning{};
    /** @brief Episode counters of the training or inference stream. */
    std::uint64_t episodes{};
    std::uint64_t touchdowns{};
    std::uint64_t stops{};
    std::uint64_t crashes{};
    std::uint64_t timeouts{};
    /** @brief Glide slope, in degrees, and action duration, in seconds. */
    double glideDegrees{};
    double stepSeconds{};
    /** @brief Touchdown aim point, in metres past the threshold. */
    double aimPointAlongMeters{};
    /** @brief Physical extension threshold used by the camera GEAR indicator. */
    double gearDownMinimum{};
    bool training{};
    bool finished{};
    /** @brief Set by the desktop bridge when the session threw. */
    bool failed{};
    /** @brief Current limit and last duration in simulated, not wall-clock, seconds. */
    double elapsedSeconds{};
    double maximumEpisodeSeconds{};
    double lastEpisodeSeconds{};
    /** @brief Campaign in progress and last completed campaign. */
    ValidationMetrics validation{};
    ValidationMetrics lastValidation{};
    std::uint64_t validationCampaigns{};
    /** @brief Score of the BEST generation, in percent. */
    double bestScore{};
    /** @brief Curriculum difficulty and PPO update of the BEST validation campaign. */
    double bestDifficulty{};
    std::uint64_t bestUpdate{};
    bool validating{};
    bool hasBest{};
    std::uint64_t trainingUpdateLimit{};
    bool resumed{};
    /** @brief Stops and episodes of the last complete independent validation. */
    std::size_t gateSuccesses{};
    std::size_t gateEvaluated{};
    /** @brief Training curriculum level, level count and current promotion window. */
    std::size_t curriculumLevel{};
    std::size_t curriculumLevels{};
    std::size_t curriculumWindowStops{};
    std::size_t curriculumWindowEpisodes{};
    /** @brief Current and last logged diagnostic windows. */
    FlightDiagnostics diagnostics{};
    FlightDiagnostics lastDiagnostics{};
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
    /** @brief Null-terminated status line of the last reward edit. */
    std::array<char, 192> rewardTuningStatus{};
    /** @brief Per-control acknowledgements, including rejected edits. */
    std::array<std::uint64_t, RewardTuningCount> rewardTuningRevisions{};
    /** @brief Sum of each reward during the currently displayed episode. */
    std::array<double, RewardCount> episodeRewardSums{};
};

/** @brief Shared orchestration for the current desktop framework and CPU headless runner. */
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
    /** @brief Advances the session by one policy action (training, validation or inference). */
    void Tick();
    /** @brief Starts the next training or inference episode. */
    void ResetEpisode();
    /** @brief Extend a completed desktop block without reinitializing the learned policy. */
    void ContinueTraining();
    FlightSnapshot Snapshot() const noexcept;
    bool Finished() const noexcept
    {
        return finished_;
    }
    FlightLearner const& Learner() const noexcept
    {
        return learner_;
    }
    /** @brief Save an existing reward coefficient and queue it for a safe boundary. */
    bool SetRewardTuning(std::size_t index, double value) noexcept;

private:
    Config config_;
    bool training_{};
    std::filesystem::path scenarioPath_;
    /** @brief Saved reward edit waiting for a rollout boundary. */
    std::optional<Config> pendingRewardConfig_;
    std::array<char, 192> rewardTuningStatus_{};
    std::array<std::uint64_t, RewardTuningCount> rewardTuningRevisions_{};
    FlightModelStore models_;
    FlightCurriculum curriculum_;
    /** @brief Training or inference environment. */
    std::unique_ptr<FlightEnvironment> environment_;
    FlightLearner learner_;
    Controls controls_{};
    StepResult lastStep_{};
    MissionState lastEpisode_{};
    std::uint64_t episodes_{};
    std::uint64_t touchdowns_{};
    std::uint64_t stops_{};
    std::uint64_t crashes_{};
    std::uint64_t timeouts_{};
    std::uint64_t crashEvent_{};
    AircraftState crashAircraft_{};
    Runway crashRunway_{};
    CrashDiagnostic crashDiagnostic_{};
    bool hasCrashDiagnostic_{};
    bool pendingReset_{};
    bool finished_{};
    std::uint64_t trainingUpdateLimit_{};
    bool resumed_{};
    /** @brief Environment of the validation campaign in progress, null otherwise. */
    std::unique_ptr<FlightEnvironment> validationEnvironment_;
    ValidationMetrics validation_{};
    ValidationMetrics lastValidation_{};
    std::optional<ValidationMetrics> best_;
    std::uint64_t validationCampaigns_{};
    std::size_t inferenceEpisodeLimit_{};
    bool validationPendingReset_{};
    FlightDiagnostics diagnostics_{};
    FlightDiagnostics lastDiagnostics_{};
    std::array<double, RewardCount> episodeRewardSums_{};
    std::array<double, RewardCount> validationEpisodeRewardSums_{};

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
    /** @brief One deterministic action of the validation campaign; the learner stays frozen. */
    void TickValidation();
    /** @brief Records a completed validation, promotes BEST and decides whether training ends. */
    void FinishValidation();
    /** @brief Counts a finished training or inference episode and feeds the curriculum. */
    void FinishEpisode();
    /** @brief Feeds a finished airborne training episode to the curriculum and applies a promotion. */
    void AdvanceCurriculum(MissionState const& episode);
    /** @brief Saves latest and its curriculum level together. */
    void SaveLatest();
    /** @brief Runs one PPO update and its periodic checkpoint and validation. */
    void OptimizeAndCheckpoint();
    /** @brief Seed of the training or inference environment. */
    std::uint32_t EnvironmentSeed() const;
    /** @brief Rebuilds the training environment with a deterministic update-specific seed. */
    void CreateTrainingEnvironment();
};

} // namespace ppo::examples::flight
