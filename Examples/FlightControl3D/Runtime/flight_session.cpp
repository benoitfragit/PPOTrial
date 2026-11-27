/**
 * @file flight_session.cpp
 * @brief Training and validation cycles, diagnostics windows and the render snapshot.
 *
 * Cycle of a training update:
 *   Tick x rolloutSteps (collect) -> Optimize -> [FLIGHT_UPDATE] log -> diagnostics window
 *   -> pending reward edit -> SaveLatest every checkpointEvery updates -> validation every
 *   evaluation.everyUpdates updates and after the last update.
 * Cycle of a validation:
 *   StartValidation (new environment seeded with evaluation.seed) -> evaluation.episodes
 *   seeded episodes with EvaluationStart plans at the curriculum difficulty -> Promote ->
 *   finished when the update budget is spent.
 *
 * The [FLIGHT_*] log lines are parsed by tools/flight/curriculum_ci.py and the CI: their
 * text, keys and order are a compatibility contract.
 *
 * Read next: flight_session.hpp, flight_headless.cpp, Training/flight_model_store.cpp.
 */
#include "flight_session.hpp"
#include "flight_task_layout.hpp"
#include "ppokit/schema/project.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace ppo::examples::flight {
namespace {
/** @brief Name of the curriculum frontier for the logs: "approach" or the stage name. */
std::string CurriculumStageName(CurriculumSettings const& settings, double difficulty)
{
    auto const position = ResolveCurriculum(settings, difficulty);
    if (!position.staged)
    {
        return "approach";
    }
    return settings.stages[position.stage].name;
}

/** @brief Largest accepted inference budget, in episodes. */
constexpr std::size_t MaximumInferenceEpisodes = 1000000;
/** @brief Updates between two probed updates (layer statistics for the studio's Probes tab). */
constexpr std::uint64_t ProbeEveryUpdates = 10;
std::uint64_t AdditionalUpdateLimit(std::uint64_t updates, std::size_t additional)
{
    if (updates > std::numeric_limits<std::uint64_t>::max() - additional)
    {
        throw std::overflow_error("FlightControl3D: training update budget overflow");
    }
    return updates + additional;
}

/** @brief Emits one complete rollout's losses and weighted reward-channel means. */
void LogUpdate(LearningStats const& stats)
{
    std::cout << "[FLIGHT_UPDATE] update=" << stats.updates
        << " mean_reward=" << stats.meanReward
        << " policy_loss=" << stats.policyLoss
        << " value_loss=" << stats.valueLoss
        << " entropy=" << stats.entropy
        << " max_minibatch_approx_kl=" << stats.maximumApproximateKl
        << " max_minibatch_sampled_kl=" << stats.maximumSampledKl
        << " actor_steps=" << stats.actorSteps << " critic_steps=" << stats.criticSteps
        << " actor_kl_skips=" << stats.actorKlSkips
        << " explained_variance=" << stats.explainedVariance;
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
    {
        std::cout << ' ' << RewardNames[channel] << '=' << stats.meanRewards[channel];
    }
    std::cout << std::endl;
}

/** @brief Reward weights in channel order, as the journal and the ledger take them. */
std::vector<double> WeightVector(RewardSettings const& reward)
{
    return {reward.weights.begin(), reward.weights.end()};
}

/** @brief `start` event of the journal: task, channels with labels and weights, hyper-parameters. */
ppokit::runtime::RunHeader JournalHeader(Config const& config)
{
    using Layout = task::TaskLayout;
    ppokit::runtime::RunHeader header;
    header.task = Layout::Name;
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
    {
        header.channels.push_back({Layout::Reward::Names[channel], Layout::Reward::Labels[channel],
            config.reward.weights[channel]});
    }
    auto const& training = config.training;
    header.settings = {{"gamma", training.gamma}, {"lambda", training.lambda}, {"clip", training.clip},
        {"actor_learning_rate", training.actorRate}, {"critic_learning_rate", training.criticRate},
        {"entropy_coefficient", training.entropy}, {"epochs", static_cast<double>(training.epochs)},
        {"minibatch", static_cast<double>(training.minibatch)},
        {"rollout_steps", static_cast<double>(training.rolloutSteps)}, {"target_kl", training.targetKl},
        {"hidden_width", static_cast<double>(training.hiddenWidth)},
        {"seed", static_cast<double>(training.seed)}, {"probe_every", static_cast<double>(ProbeEveryUpdates)}};
    return header;
}

/** @brief Flight PPO statistics under the journal's stable names. */
ppokit::runtime::NamedValues JournalMetrics(LearningStats const& stats)
{
    return {{"policy_loss", stats.policyLoss}, {"value_loss", stats.valueLoss}, {"entropy", stats.entropy},
        {"mean_reward", stats.meanReward}, {"max_approx_kl", stats.maximumApproximateKl},
        {"max_sampled_kl", stats.maximumSampledKl}, {"actor_kl_skips", static_cast<double>(stats.actorKlSkips)},
        {"explained_variance", stats.explainedVariance}};
}

void UpdateLiveDiagnostics(FlightDiagnostics& diagnostics, Observation const& observation,
    PolicyAction const& action, Controls const& controls) noexcept
{
    if (!diagnostics.enabled)
    {
        return;
    }
    diagnostics.observation = observation;
    diagnostics.policy = action.policy.output;
    diagnostics.logStandardDeviations = action.policy.logStandardDeviations;
    diagnostics.rawAction = action.raw;
    diagnostics.appliedAction = Applied(controls);
}

void AccumulateObservationDiagnostics(FlightDiagnostics& diagnostics, Observation const& observation,
    double threshold) noexcept
{
    for (std::size_t slot = 0; slot < ObservationSize; ++slot)
    {
        double const absolute = std::abs(static_cast<double>(observation[slot]));
        diagnostics.observationAbsoluteSums[slot] += absolute;
        diagnostics.observationMaximumAbsolute[slot] = std::max(diagnostics.observationMaximumAbsolute[slot],
            absolute);
        if (absolute >= threshold * ObservationBound)
        {
            ++diagnostics.observationSaturations[slot];
        }
    }
}

void AccumulatePolicyDiagnostics(FlightDiagnostics& diagnostics, PolicyAction const& action,
    double threshold) noexcept
{
    for (std::size_t output = 0; output < PolicyOutputs; ++output)
    {
        double const absolute = std::abs(action.policy.output[output]);
        diagnostics.policyAbsoluteSums[output] += absolute;
        diagnostics.policyMaximumAbsolute[output] = std::max(diagnostics.policyMaximumAbsolute[output],
            absolute);
    }
    for (std::size_t axis = 0; axis < ContinuousActions; ++axis)
    {
        double const logStandardDeviationAbsolute = std::abs(action.policy.logStandardDeviations[axis]);
        diagnostics.logStandardDeviationAbsoluteSums[axis] += logStandardDeviationAbsolute;
        diagnostics.logStandardDeviationMaximumAbsolute[axis] = std::max(
            diagnostics.logStandardDeviationMaximumAbsolute[axis], logStandardDeviationAbsolute);
        double const raw = action.raw[axis];
        double const applied = diagnostics.appliedAction[axis];
        diagnostics.rawAbsoluteSums[axis] += std::abs(raw);
        diagnostics.appliedAbsoluteSums[axis] += std::abs(applied);
        bool const active = ActiveAxis(axis, action.brakeActive);
        if (active && RawSaturated(axis, raw, threshold))
        {
            ++diagnostics.rawSaturations[axis];
        }
        if (active && AppliedSaturated(axis, applied, threshold))
        {
            ++diagnostics.appliedSaturations[axis];
        }
    }
}

void AccumulateRewardDiagnostics(FlightDiagnostics& diagnostics, StepResult const& step) noexcept
{
    diagnostics.rewardSum += step.reward;
    if (diagnostics.samples == 1)
    {
        diagnostics.rewardMinimum = step.reward;
        diagnostics.rewardMaximum = step.reward;
    }
    else
    {
        diagnostics.rewardMinimum = std::min(diagnostics.rewardMinimum, step.reward);
        diagnostics.rewardMaximum = std::max(diagnostics.rewardMaximum, step.reward);
    }
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
    {
        diagnostics.rewardSums[channel] += step.rewards[channel];
    }
}

void RecordDiagnostics(FlightDiagnostics& diagnostics, Observation const& observation,
    PolicyAction const& action, Controls const& controls, StepResult const& step, double threshold) noexcept
{
    if (!diagnostics.enabled)
    {
        return;
    }
    UpdateLiveDiagnostics(diagnostics, observation, action, controls);
    ++diagnostics.samples;
    AccumulateObservationDiagnostics(diagnostics, observation, threshold);
    AccumulatePolicyDiagnostics(diagnostics, action, threshold);
    AccumulateRewardDiagnostics(diagnostics, step);
}

void LogObservationDiagnostics(FlightDiagnostics const& diagnostics, std::uint64_t update, double samples)
{
    std::cout << "[FLIGHT_OBSERVATION_DIAG] update=" << update << " samples=" << diagnostics.samples;
    for (std::size_t slot = 0; slot < ObservationSize; ++slot)
    {
        std::cout << ' ' << ObservationNames[slot] << "_mean_abs="
            << diagnostics.observationAbsoluteSums[slot] / samples
            << ' ' << ObservationNames[slot] << "_max_abs=" << diagnostics.observationMaximumAbsolute[slot]
            << ' ' << ObservationNames[slot] << "_sat=" << diagnostics.observationSaturations[slot];
    }
    std::cout << std::endl;
}

void LogPolicyDiagnostics(FlightDiagnostics const& diagnostics, std::uint64_t update, double samples)
{
    std::cout << "[FLIGHT_POLICY_DIAG] update=" << update;
    for (std::size_t output = 0; output < PolicyOutputs; ++output)
    {
        std::cout << ' ' << PolicyOutputNames[output] << "_abs="
            << diagnostics.policyAbsoluteSums[output] / samples
            << ' ' << PolicyOutputNames[output] << "_max_abs=" << diagnostics.policyMaximumAbsolute[output];
    }
    for (std::size_t axis = 0; axis < ContinuousActions; ++axis)
    {
        std::cout << ' ' << ActionNames[axis] << "_log_std_mean_abs="
            << diagnostics.logStandardDeviationAbsoluteSums[axis] / samples
            << ' ' << ActionNames[axis] << "_log_std_max_abs="
            << diagnostics.logStandardDeviationMaximumAbsolute[axis]
            << ' ' << ActionNames[axis] << "_raw_mean_abs=" << diagnostics.rawAbsoluteSums[axis] / samples
            << ' ' << ActionNames[axis] << "_applied_mean_abs="
            << diagnostics.appliedAbsoluteSums[axis] / samples
            << ' ' << ActionNames[axis] << "_raw_sat=" << diagnostics.rawSaturations[axis]
            << ' ' << ActionNames[axis] << "_applied_sat=" << diagnostics.appliedSaturations[axis];
    }
    std::cout << std::endl;
}

void LogRewardDiagnostics(FlightDiagnostics const& diagnostics, std::uint64_t update, double samples)
{
    std::cout << "[FLIGHT_REWARD_DIAG] update=" << update
        << " mean=" << diagnostics.rewardSum / samples
        << " minimum=" << diagnostics.rewardMinimum << " maximum=" << diagnostics.rewardMaximum;
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
    {
        std::cout << ' ' << RewardNames[channel] << "_mean=" << diagnostics.rewardSums[channel] / samples;
    }
    std::cout << std::endl;
}

/** @brief Emits the three diagnostic log lines of a finished window. */
void LogDiagnostics(FlightDiagnostics const& diagnostics, std::uint64_t update)
{
    if (!diagnostics.enabled || diagnostics.samples == 0)
    {
        return;
    }
    double const samples = static_cast<double>(diagnostics.samples);
    LogObservationDiagnostics(diagnostics, update, samples);
    LogPolicyDiagnostics(diagnostics, update, samples);
    LogRewardDiagnostics(diagnostics, update, samples);
}

/** @brief Starts a new diagnostic window while keeping the live values shown by the renderer. */
FlightDiagnostics NextDiagnosticsWindow(FlightDiagnostics const& previous) noexcept
{
    FlightDiagnostics next{};
    next.enabled = true;
    next.observation = previous.observation;
    next.policy = previous.policy;
    next.logStandardDeviations = previous.logStandardDeviations;
    next.rawAction = previous.rawAction;
    next.appliedAction = previous.appliedAction;
    return next;
}

/** @brief Adds the channel rewards of one action to an episode total. */
void AddEpisodeRewards(std::array<double, RewardCount>& sums, StepResult const& step) noexcept
{
    for (std::size_t channel = 0; channel < RewardCount; ++channel)
    {
        sums[channel] += step.rewards[channel];
    }
}
} // namespace

FlightSession::FlightSession(Config config, bool training, std::filesystem::path modelDirectory,
    std::optional<std::size_t> inferenceEpisodes, TrainingBudget budget, std::filesystem::path scenarioPath, bool legacyInbox)
    : config_(std::move(config)), training_(training), scenarioPath_(std::move(scenarioPath)),
      models_(config_, std::move(modelDirectory)),
      curriculum_(config_.training.curriculum), learner_(config_),
      rewardLedger_(WeightVector(config_.reward), 1)
{
    diagnostics_.enabled = config_.diagnostics.enabled;
    lastDiagnostics_.enabled = config_.diagnostics.enabled;
    if (inferenceEpisodes && (training_ || *inferenceEpisodes == 0
        || *inferenceEpisodes > MaximumInferenceEpisodes))
    {
        throw std::invalid_argument("FlightControl3D: invalid inference episode budget");
    }
    // Inference episode i uses seed evaluation.seed + i, which must not overflow.
    if (inferenceEpisodes
        && *inferenceEpisodes - 1U > std::numeric_limits<std::uint32_t>::max() - config_.evaluation.seed)
    {
        throw std::invalid_argument("FlightControl3D: inference episode seeds overflow");
    }
    inferenceEpisodeLimit_ = inferenceEpisodes.value_or(config_.evaluation.episodes);
    if (!training_)
    {
        models_.LoadBest(learner_);
    }
    else if (config_.training.resume && std::filesystem::exists(models_.Latest()))
    {
        learner_.Load(models_.Latest());
        resumed_ = true;
        if (auto const level = LoadCurriculumLevel(models_.CurriculumLevel()))
        {
            curriculum_.Restore(*level);
        }
    }
    best_ = models_.BestMetrics(training_);
    if (budget == TrainingBudget::AdditionalUpdates)
    {
        trainingUpdateLimit_ = AdditionalUpdateLimit(learner_.Stats().updates, config_.training.maxUpdates);
    }
    else
    {
        trainingUpdateLimit_ = config_.training.maxUpdates;
    }
    if (training_)
    {
        OpenJournal(legacyInbox);
    }
    CreateTrainingEnvironment();
    if (training_ && learner_.Stats().updates >= trainingUpdateLimit_)
    {
        StartValidation();
    }
}

void FlightSession::ContinueTraining()
{
    if (!training_ || !finished_)
    {
        return;
    }
    trainingUpdateLimit_ = AdditionalUpdateLimit(learner_.Stats().updates, config_.training.maxUpdates);
    finished_ = false;
    pendingReset_ = true;
}

void FlightSession::ResetEpisode()
{
    if (!finished_)
    {
        if (training_)
        {
            environment_->Reset();
        }
        else
        {
            ResetInferenceEpisode();
        }
        episodeRewardSums_.fill(0);
    }
    pendingReset_ = false;
}

void FlightSession::CaptureCrash(FlightEnvironment const& environment) noexcept
{
    if (environment.Mission().outcome != Outcome::Crash)
    {
        return;
    }
    crashDiagnostic_ = environment.CrashDetails();
    hasCrashDiagnostic_ = true;
    if (!IsFinite(environment.Aircraft()))
    {
        return;
    }
    crashAircraft_ = environment.Aircraft();
    crashRunway_ = environment.Settings().runway;
    ++crashEvent_;
}

void FlightSession::Tick()
{
    if (pendingRewardConfig_ && !validationEnvironment_ && learner_.Collected() == 0)
    {
        ApplyPendingRewardTuning();
    }
    if (finished_)
    {
        return;
    }
    if (validationEnvironment_)
    {
        TickValidation();
        return;
    }
    if (pendingReset_)
    {
        ResetEpisode();
    }
    // One policy action: stochastic in training, deterministic in inference.
    auto const observation = environment_->ObservationNow();
    auto const action = learner_.Choose(observation, !training_);
    controls_ = ToControls(action, config_.initial);
    lastStep_ = environment_->Step(controls_);
    AddEpisodeRewards(episodeRewardSums_, lastStep_);
    controls_ = environment_->Mission().filteredControls;
    RecordDiagnostics(diagnostics_, observation, action, controls_, lastStep_,
        config_.diagnostics.saturationThreshold);
    if (training_)
    {
        learner_.Record(observation, action, lastStep_, environment_->ObservationNow());
        rewardLedger_.RecordWeighted(0, lastStep_.rewards.data(), lastStep_.reward);
    }
    if (lastStep_.terminated || lastStep_.truncated)
    {
        FinishEpisode();
    }
    if (training_ && learner_.Ready())
    {
        OptimizeAndCheckpoint();
    }
    if (!training_ && episodes_ >= inferenceEpisodeLimit_)
    {
        finished_ = true;
    }
}

void FlightSession::FinishEpisode()
{
    CaptureCrash(*environment_);
    lastEpisode_ = environment_->Mission();
    ++episodes_;
    if (lastEpisode_.touchedDown)
    {
        ++touchdowns_;
    }
    if (lastEpisode_.outcome == Outcome::Stopped)
    {
        ++stops_;
    }
    if (lastEpisode_.outcome == Outcome::Crash)
    {
        ++crashes_;
    }
    if (lastEpisode_.outcome == Outcome::Timeout)
    {
        ++timeouts_;
    }
    if (training_)
    {
        rewardLedger_.EndEpisode(0);
    }
    // Only airborne episodes measure the curriculum difficulty.
    if (training_ && environment_->StartedAirborne())
    {
        AdvanceCurriculum(lastEpisode_);
    }
    pendingReset_ = true;
}

void FlightSession::AdvanceCurriculum(MissionState const& episode)
{
    if (!curriculum_.Record(episode.outcome == Outcome::Stopped))
    {
        return;
    }
    environment_->SetDifficulty(curriculum_.Difficulty());
    std::cout << "[FLIGHT_CURRICULUM] level=" << curriculum_.Level() << "/" << curriculum_.Levels() - 1
        << " difficulty=" << curriculum_.Difficulty()
        << " stage=" << CurriculumStageName(config_.training.curriculum, curriculum_.Difficulty())
        << " update=" << learner_.Stats().updates << std::endl;
}

void FlightSession::SaveLatest()
{
    learner_.Save(models_.Latest());
    SaveCurriculumLevel(models_.CurriculumLevel(), curriculum_.Level());
}

void FlightSession::OpenJournal(bool legacyInbox)
{
    auto const path = models_.RunJournal();
    std::filesystem::create_directories(path.parent_path());
    auto header = JournalHeader(config_);
    // The studio can change weights only where they can be saved: a scenario file.
    header.acceptsWeightRequests = !scenarioPath_.empty();
    // A resumed latest continues its journal; a new policy starts a new one.
    journal_ = resumed_ ? ppokit::runtime::RunLogWriter::Resume(path, header, learner_.Stats().updates, &rewardLedger_)
                        : ppokit::runtime::RunLogWriter(path, header);
    journalStart_ = std::chrono::steady_clock::now();
    if (header.acceptsWeightRequests && legacyInbox)
    {
        std::vector<std::string> names;
        for (auto const& channel : header.channels)
        {
            names.push_back(channel.name);
        }
        weightRequests_.emplace(path, std::move(names));
    }
}

void FlightSession::TakeWeightRequest()
{
    if (!weightRequests_)
    {
        return;
    }
    try
    {
        auto const request = weightRequests_->Take();
        if (!request)
        {
            return;
        }
        weightRequests_->Answer(ApplyWeightRequest(*request));
    }
    catch (std::exception const& error)
    {
        // A studio file problem must never stop the training.
        std::cerr << "[FLIGHT_WEIGHTS] weight request ignored: " << error.what() << std::endl;
    }
}

ppokit::runtime::WeightAnswer FlightSession::ApplyWeightRequest(ppokit::runtime::WeightRequest const& request)
{
    if (scenarioPath_.empty())
    {
        return {request.id, false, "Refused: this session has no scenario file to save the weights in"};
    }
    // Validate every weight before applying any: a request is applied whole or not at all.
    Config candidate = pendingRewardConfig_ ? *pendingRewardConfig_ : config_;
    try
    {
        for (auto const& [channel, weight] : request.weights)
        {
            candidate = WithRewardTuning(candidate, channel, weight);
        }
    }
    catch (std::exception const& error)
    {
        return {request.id, false, std::string("Refused: ") + error.what()};
    }
    bool applied = true;
    for (auto const& [channel, weight] : request.weights)
    {
        applied = SetRewardTuning(channel, weight) && applied;
    }
    std::uint64_t const from = learner_.Stats().updates + 1U;
    // Right after an update the change takes effect at once (ApplyPendingRewardTuning checks it).
    ApplyPendingRewardTuning();
    std::string const status = rewardTuningStatus_.data();
    std::string const message = applied ? "Applied from update " + std::to_string(from) + ". " + status : status;
    return {request.id, applied, message};
}

void FlightSession::JournalUpdate()
{
    if (!journal_)
    {
        return;
    }
    ppokit::runtime::UpdateRecord record;
    record.update = learner_.Stats().updates;
    record.elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - journalStart_).count();
    record.ppo = JournalMetrics(learner_.Stats());
    record.rewards = rewardLedger_.Summarize();
    journal_->WriteUpdate(record);
}

void FlightSession::JournalProbes()
{
    auto const update = learner_.Stats().updates;
    journal_->WriteProbes({update, "actor", learner_.LayerProbes().actor});
    journal_->WriteProbes({update, "critic", learner_.LayerProbes().critic});
}

void FlightSession::OptimizeAndCheckpoint()
{
    // Every ProbeEveryUpdates-th update is probed for the studio (read-only: same weights).
    bool const probed = journal_ && (learner_.Stats().updates + 1U) % ProbeEveryUpdates == 0U;
    learner_.ProbeLayers(probed);
    learner_.Optimize();
    learner_.ProbeLayers(false);
    LogUpdate(learner_.Stats());
    JournalUpdate();
    if (probed)
    {
        JournalProbes();
    }
    auto const updates = learner_.Stats().updates;
    bool const final = updates >= trainingUpdateLimit_;
    // Diagnostics are logged per window of logEveryUpdates updates, and after the last one.
    if (config_.diagnostics.enabled && (final || updates % config_.diagnostics.logEveryUpdates == 0))
    {
        lastDiagnostics_ = diagnostics_;
        LogDiagnostics(lastDiagnostics_, updates);
        diagnostics_ = NextDiagnosticsWindow(diagnostics_);
    }
    // The buffer is empty now: a weight request of the studio or a pending reward edit can take effect.
    TakeWeightRequest();
    if (pendingRewardConfig_)
    {
        ApplyPendingRewardTuning();
    }
    bool const validate = final || updates % config_.evaluation.everyUpdates == 0;
    if (validate || updates % config_.training.checkpointEvery == 0)
    {
        SaveLatest();
    }
    if (validate)
    {
        StartValidation();
    }
}

FlightSnapshot FlightSession::Snapshot() const noexcept
{
    // The renderer follows the validation environment while a campaign runs.
    FlightEnvironment const* activeEnvironment = environment_.get();
    if (validationEnvironment_)
    {
        activeEnvironment = validationEnvironment_.get();
    }
    auto const& active = *activeEnvironment;
    FlightSnapshot state{};
    // Current episode.
    state.runway = active.Settings().runway;
    state.aircraft = active.Aircraft();
    state.mission = active.Mission();
    state.lastEpisode = lastEpisode_;
    state.guidance = active.CurrentGuidance();
    state.controls = controls_;
    state.lastStep = lastStep_;
    state.learning = learner_.Stats();
    // Episode counters and scenario references.
    state.episodes = episodes_;
    state.touchdowns = touchdowns_;
    state.stops = stops_;
    state.crashes = crashes_;
    state.timeouts = timeouts_;
    state.glideDegrees = config_.mission.glideDegrees;
    state.aimPointAlongMeters = AimPointAlongMeters(active.Settings());
    state.stepSeconds = config_.StepSeconds();
    state.gearDownMinimum = config_.mission.gearDownMinimum;
    state.training = training_;
    state.finished = finished_;
    state.elapsedSeconds = static_cast<double>(state.mission.steps) * active.Settings().mission.physicsDt;
    state.maximumEpisodeSeconds = active.Settings().mission.maxSeconds;
    state.lastEpisodeSeconds = static_cast<double>(state.lastEpisode.steps) * config_.mission.physicsDt;
    // Validation, BEST and curriculum.
    state.validation = validation_;
    state.lastValidation = lastValidation_;
    state.validationCampaigns = validationCampaigns_;
    state.validating = static_cast<bool>(validationEnvironment_);
    state.hasBest = best_.has_value();
    if (best_)
    {
        state.bestScore = best_->Score();
        state.bestDifficulty = best_->difficulty;
        state.bestUpdate = best_->update;
    }
    state.trainingUpdateLimit = trainingUpdateLimit_;
    state.resumed = resumed_;
    state.gateSuccesses = lastValidation_.stops;
    state.gateEvaluated = lastValidation_.episodes;
    state.curriculumLevel = curriculum_.Level();
    state.curriculumLevels = curriculum_.Levels();
    state.curriculumWindowStops = curriculum_.WindowStops();
    state.curriculumWindowEpisodes = curriculum_.WindowEpisodes();
    // Diagnostics and crash details.
    state.diagnostics = diagnostics_;
    state.lastDiagnostics = lastDiagnostics_;
    state.seaHeightMeters = active.Settings().runwayRandomization.seaHeightMeters;
    state.deck = active.CurrentDeck();
    state.crashEvent = crashEvent_;
    state.crashAircraft = crashAircraft_;
    state.crashRunway = crashRunway_;
    state.crashDiagnostic = crashDiagnostic_;
    state.hasCrashDiagnostic = hasCrashDiagnostic_;
    // Reward tuning.
    state.rewardSettings = config_.reward;
    if (pendingRewardConfig_)
    {
        state.rewardSettings = pendingRewardConfig_->reward;
    }
    state.activeRewardSettings = config_.reward;
    state.rewardTuningPending = pendingRewardConfig_.has_value();
    state.rewardTuningStatus = rewardTuningStatus_;
    state.rewardTuningRevisions = rewardTuningRevisions_;
    state.episodeRewardSums = episodeRewardSums_;
    if (validationEnvironment_)
    {
        state.episodeRewardSums = validationEpisodeRewardSums_;
    }
    state.approachProfileValid = IsFinite(state.aircraft) && std::isfinite(state.guidance.along)
        && std::isfinite(state.guidance.height);
    return state;
}

bool FlightSession::SetRewardTuning(std::size_t index, double value) noexcept
{
    try
    {
        if (!training_ || scenarioPath_.empty())
        {
            throw std::runtime_error("Reward editing is unavailable for this session");
        }
        auto const& current = pendingRewardConfig_ ? *pendingRewardConfig_ : config_;
        if (index >= RewardTuningCount)
        {
            throw std::out_of_range("Unknown reward setting");
        }
        // An unchanged value is acknowledged without touching the scenario file.
        if (RewardTuningValue(current.reward, index) == value)
        {
            ++rewardTuningRevisions_[index];
            return true;
        }
        auto next = WithRewardTuning(current, index, value);
        SaveRewardTuning(scenarioPath_, index, value);
        pendingRewardConfig_ = std::move(next);
        std::snprintf(rewardTuningStatus_.data(), rewardTuningStatus_.size(),
            "Saved to JSON; waiting for PPO rollout boundary");
        if (!validationEnvironment_ && learner_.Collected() == 0)
        {
            ApplyPendingRewardTuning();
        }
        ++rewardTuningRevisions_[index];
        return true;
    }
    catch (std::exception const& error)
    {
        std::snprintf(rewardTuningStatus_.data(), rewardTuningStatus_.size(), "Reward edit rejected: %s",
            error.what());
        if (index < RewardTuningCount)
        {
            ++rewardTuningRevisions_[index];
        }
        return false;
    }
}

void FlightSession::ApplyPendingRewardTuning()
{
    if (!pendingRewardConfig_ || validationEnvironment_ || learner_.Collected() != 0)
    {
        return;
    }
    auto const& next = *pendingRewardConfig_;
    if (journal_)
    {
        auto const previous = WeightVector(config_.reward);
        auto const weights = WeightVector(next.reward);
        if (weights != previous)
        {
            // The buffer is empty: the next update is the first one collected under these weights.
            journal_->WriteWeights({learner_.Stats().updates + 1U, previous, weights});
        }
        rewardLedger_.SetWeights(weights);
    }
    learner_.SetRewardContract(next.reward, next.checkpointSignature);
    models_.SetRewardContract(next.reward, next.checkpointSignature);
    environment_->SetRewardSettings(next.reward);
    config_.reward = next.reward;
    config_.checkpointSignature = next.checkpointSignature;
    pendingRewardConfig_.reset();
    // The previous BEST was scored under another reward contract.
    best_.reset();
    lastValidation_ = {};
    validationCampaigns_ = 0;
    std::snprintf(rewardTuningStatus_.data(), rewardTuningStatus_.size(),
        "Saved and active; BEST requires revalidation");
}

void FlightSession::StartValidation()
{
    validation_ = {};
    validation_.update = learner_.Stats().updates;
    // The campaign follows the curriculum: it evaluates the mix being trained.
    validation_.difficulty = curriculum_.Difficulty();
    validationPendingReset_ = false;
    lastStep_ = {};
    controls_ = {};
    validationEnvironment_ = std::make_unique<FlightEnvironment>(config_, DefaultDataRoot(),
        config_.evaluation.seed);
    ResetValidationEpisode();
}

void FlightSession::ResetValidationEpisode()
{
    auto const seed = config_.evaluation.seed + static_cast<std::uint32_t>(validation_.episodes);
    auto const start = EvaluationStart(validation_.episodes, config_.evaluation.episodes,
        config_.training.curriculum.rolloutStartShare, validation_.difficulty);
    validationEnvironment_->Reset(seed, start);
    validationEpisodeRewardSums_.fill(0);
    std::cout << "[FLIGHT_EVALUATION_EPISODE] episode=" << (validation_.episodes + 1U) << " seed=" << seed
        << " start=" << (start.rolling ? "rolling" : "airborne") << " difficulty=" << start.difficulty
            << std::endl;
}

void FlightSession::ResetInferenceEpisode()
{
    // Inference evaluates the full curriculum mix, from rolling starts to the hardest approaches.
    auto const seed = config_.evaluation.seed + static_cast<std::uint32_t>(episodes_);
    environment_->Reset(seed, EvaluationStart(episodes_, inferenceEpisodeLimit_,
        config_.training.curriculum.rolloutStartShare, FullTaskDifficulty));
}

void FlightSession::TickValidation()
{
    auto& evaluation = *validationEnvironment_;
    if (validationPendingReset_)
    {
        ResetValidationEpisode();
        validationPendingReset_ = false;
    }
    auto const observation = evaluation.ObservationNow();
    auto const action = learner_.Choose(observation, true);
    controls_ = ToControls(action, config_.initial);
    lastStep_ = evaluation.Step(controls_);
    AddEpisodeRewards(validationEpisodeRewardSums_, lastStep_);
    controls_ = evaluation.Mission().filteredControls;
    UpdateLiveDiagnostics(diagnostics_, observation, action, controls_);
    // No Record/Optimize here: training rollout, optimizer and RNG remain frozen.
    if (!lastStep_.terminated && !lastStep_.truncated)
    {
        return;
    }
    CaptureCrash(evaluation);
    validation_.Record(evaluation.Mission());
    if (validation_.episodes < config_.evaluation.episodes)
    {
        validationPendingReset_ = true;
    }
    else
    {
        FinishValidation();
    }
}

void FlightSession::FinishValidation()
{
    models_.Promote(learner_, validation_);
    best_ = models_.BestMetrics(true);
    lastValidation_ = validation_;
    ++validationCampaigns_;
    validationEnvironment_.reset();
    lastStep_ = {};
    controls_ = {};
    finished_ = learner_.Stats().updates >= trainingUpdateLimit_;
    if (pendingRewardConfig_)
    {
        ApplyPendingRewardTuning();
    }
}

std::uint32_t FlightSession::EnvironmentSeed() const
{
    if (!training_)
    {
        return config_.evaluation.seed;
    }
    if (config_.training.randomizeEnvironmentSeed)
    {
        return std::random_device{}();
    }
    return TrainingSeed(config_.training.seed, learner_.Stats().updates);
}

void FlightSession::CreateTrainingEnvironment()
{
    auto const seed = EnvironmentSeed();
    std::cout << "[FLIGHT_DYNAMICS] mode=" << (training_ ? "training" : "inference")
        << " aircraft=" << config_.aircraft << " environment_seed=" << seed << std::endl;
    environment_ = std::make_unique<FlightEnvironment>(config_, DefaultDataRoot(), seed);
    double difficulty = FullTaskDifficulty;
    if (training_)
    {
        difficulty = curriculum_.Difficulty();
    }
    environment_->SetDifficulty(difficulty);
    episodeRewardSums_.fill(0);
    if (training_)
    {
        environment_->Reset();
    }
    else
    {
        ResetInferenceEpisode();
    }
}
std::filesystem::path ModelDirectoryFor(std::filesystem::path const& scenarioPath, Config const& config)
{
    std::filesystem::path const absolute = std::filesystem::absolute(scenarioPath);
    if (std::optional<std::filesystem::path> const project = ppokit::schema::ProjectFolderOf(absolute))
    {
        return *project / "checkpoints";
    }
    // A scenario outside a project keeps its models beside it.
    return (absolute.parent_path() / config.training.checkpointDirectory).lexically_normal();
}

} // namespace ppo::examples::flight
