#include "flight_session.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <random>
#include <utility>

namespace ppo::examples::flight {
namespace {
std::uint64_t AdditionalUpdateLimit(std::uint64_t updates,std::size_t additional)
{
    if(updates>std::numeric_limits<std::uint64_t>::max()-additional)
        throw std::overflow_error("FlightControl3D: training update budget overflow");
    return updates+additional;
}
std::uint32_t TrainingSeed(std::uint32_t base,std::uint64_t updates) noexcept
{
    // SplitMix64 keeps resumed update blocks deterministic but distinct.
    std::uint64_t value=static_cast<std::uint64_t>(base)^((updates+1U)*0x9e3779b97f4a7c15ULL);
    value=(value^(value>>30U))*0xbf58476d1ce4e5b9ULL;
    value=(value^(value>>27U))*0x94d049bb133111ebULL;
    value^=value>>31U;
    return static_cast<std::uint32_t>(value^(value>>32U));
}
/** @brief Emits one complete rollout's losses and weighted reward-channel means. */
void LogUpdate(LearningStats const& stats)
{
    std::cout<<"[FLIGHT_UPDATE] update="<<stats.updates
        <<" mean_reward="<<stats.meanReward
        <<" policy_loss="<<stats.policyLoss
        <<" value_loss="<<stats.valueLoss
        <<" entropy="<<stats.entropy
        <<" max_minibatch_approx_kl="<<stats.maximumApproximateKl
        <<" max_minibatch_sampled_kl="<<stats.maximumSampledKl
        <<" actor_steps="<<stats.actorSteps<<" critic_steps="<<stats.criticSteps
        <<" actor_kl_skips="<<stats.actorKlSkips;
    for(std::size_t channel=0;channel<RewardCount;++channel)
        std::cout<<' '<<RewardNames[channel]<<'='<<stats.meanRewards[channel];
    std::cout<<std::endl;
}
std::array<double,ContinuousActions> Applied(Controls const& controls) noexcept
{
    return {controls.elevator,controls.aileron,controls.rudder,controls.throttle,controls.brake};
}
void UpdateLiveDiagnostics(FlightDiagnostics& diagnostics,Observation const& observation,
    PolicyAction const& action,Controls const& controls) noexcept
{
    if(!diagnostics.enabled) return;
    diagnostics.observation=observation;
    diagnostics.policy=action.policy.output;
    diagnostics.logStandardDeviations=action.policy.logStandardDeviations;
    diagnostics.rawAction=action.raw;
    diagnostics.appliedAction=Applied(controls);
}
void RecordDiagnostics(FlightDiagnostics& diagnostics,Observation const& observation,
    PolicyAction const& action,Controls const& controls,StepResult const& step,double threshold) noexcept
{
    if(!diagnostics.enabled) return;
    UpdateLiveDiagnostics(diagnostics,observation,action,controls);
    ++diagnostics.samples;
    for(std::size_t i=0;i<ObservationSize;++i)
    {
        double const absolute=std::abs(static_cast<double>(observation[i]));
        diagnostics.observationAbsoluteSums[i]+=absolute;
        diagnostics.observationMaximumAbsolute[i]=std::max(diagnostics.observationMaximumAbsolute[i],absolute);
        if(absolute>=threshold*ObservationBound) ++diagnostics.observationSaturations[i];
    }
    for(std::size_t i=0;i<PolicyOutputs;++i)
    {
        double const absolute=std::abs(action.policy.output[i]);
        diagnostics.policyAbsoluteSums[i]+=absolute;
        diagnostics.policyMaximumAbsolute[i]=std::max(diagnostics.policyMaximumAbsolute[i],absolute);
    }
    for(std::size_t i=0;i<ContinuousActions;++i)
    {
        double const logStandardDeviationAbsolute=std::abs(action.policy.logStandardDeviations[i]);
        diagnostics.logStandardDeviationAbsoluteSums[i]+=logStandardDeviationAbsolute;
        diagnostics.logStandardDeviationMaximumAbsolute[i]=std::max(
            diagnostics.logStandardDeviationMaximumAbsolute[i],logStandardDeviationAbsolute);
        double const raw=action.raw[i],applied=diagnostics.appliedAction[i];
        diagnostics.rawAbsoluteSums[i]+=std::abs(raw);
        diagnostics.appliedAbsoluteSums[i]+=std::abs(applied);
        bool const brake=i==BrakeAction;
        bool const rawSaturated=brake ? raw<=0 || raw>=threshold : std::abs(raw)>=threshold;
        // Throttle and brake are one-sided physical ranges.
        bool const appliedSaturated=i>=3 ? applied<=1-threshold || applied>=threshold
            : std::abs(applied)>=threshold;
        if(ActiveAxis(i,action.brakeActive) && rawSaturated) ++diagnostics.rawSaturations[i];
        if(ActiveAxis(i,action.brakeActive) && appliedSaturated) ++diagnostics.appliedSaturations[i];
    }
    diagnostics.rewardSum+=step.reward;
    if(diagnostics.samples==1)
        diagnostics.rewardMinimum=diagnostics.rewardMaximum=step.reward;
    else
    {
        diagnostics.rewardMinimum=std::min(diagnostics.rewardMinimum,step.reward);
        diagnostics.rewardMaximum=std::max(diagnostics.rewardMaximum,step.reward);
    }
    for(std::size_t i=0;i<RewardCount;++i) diagnostics.rewardSums[i]+=step.rewards[i];
}
void LogDiagnostics(FlightDiagnostics const& diagnostics,std::uint64_t update)
{
    if(!diagnostics.enabled || diagnostics.samples==0) return;
    double const samples=static_cast<double>(diagnostics.samples);
    std::cout<<"[FLIGHT_OBSERVATION_DIAG] update="<<update<<" samples="<<diagnostics.samples;
    for(std::size_t i=0;i<ObservationSize;++i)
        std::cout<<' '<<ObservationNames[i]<<"_mean_abs="<<diagnostics.observationAbsoluteSums[i]/samples
            <<' '<<ObservationNames[i]<<"_max_abs="<<diagnostics.observationMaximumAbsolute[i]
            <<' '<<ObservationNames[i]<<"_sat="<<diagnostics.observationSaturations[i];
    std::cout<<std::endl;
    std::cout<<"[FLIGHT_POLICY_DIAG] update="<<update;
    for(std::size_t i=0;i<PolicyOutputs;++i)
        std::cout<<' '<<PolicyOutputNames[i]<<"_abs="<<diagnostics.policyAbsoluteSums[i]/samples
            <<' '<<PolicyOutputNames[i]<<"_max_abs="<<diagnostics.policyMaximumAbsolute[i];
    for(std::size_t i=0;i<ContinuousActions;++i)
        std::cout<<' '<<ActionNames[i]<<"_log_std_mean_abs="<<diagnostics.logStandardDeviationAbsoluteSums[i]/samples
            <<' '<<ActionNames[i]<<"_log_std_max_abs="<<diagnostics.logStandardDeviationMaximumAbsolute[i]
            <<' '<<ActionNames[i]<<"_raw_mean_abs="<<diagnostics.rawAbsoluteSums[i]/samples
            <<' '<<ActionNames[i]<<"_applied_mean_abs="<<diagnostics.appliedAbsoluteSums[i]/samples
            <<' '<<ActionNames[i]<<"_raw_sat="<<diagnostics.rawSaturations[i]
            <<' '<<ActionNames[i]<<"_applied_sat="<<diagnostics.appliedSaturations[i];
    std::cout<<std::endl;
    std::cout<<"[FLIGHT_REWARD_DIAG] update="<<update
        <<" mean="<<diagnostics.rewardSum/samples
        <<" minimum="<<diagnostics.rewardMinimum<<" maximum="<<diagnostics.rewardMaximum;
    for(std::size_t i=0;i<RewardCount;++i)
        std::cout<<' '<<RewardNames[i]<<"_mean="<<diagnostics.rewardSums[i]/samples;
    std::cout<<std::endl;
}
/** @brief Starts a new diagnostic window while keeping the live values shown by the renderer. */
FlightDiagnostics NextDiagnosticsWindow(FlightDiagnostics const& previous) noexcept
{
    FlightDiagnostics next{};
    next.enabled=true;
    next.observation=previous.observation;
    next.policy=previous.policy;
    next.logStandardDeviations=previous.logStandardDeviations;
    next.rawAction=previous.rawAction;
    next.appliedAction=previous.appliedAction;
    return next;
}
} // namespace

FlightSession::FlightSession(Config config, bool training, std::filesystem::path modelDirectory,
    std::optional<std::size_t> inferenceEpisodes,TrainingBudget budget,std::filesystem::path scenarioPath)
    : config_(std::move(config)),training_(training),modelDirectory_(std::move(modelDirectory)),
      scenarioPath_(std::move(scenarioPath)),models_(config_,modelDirectory_),
      curriculum_(config_.training.curriculum),learner_(config_)
{
    diagnostics_.enabled=config_.diagnostics.enabled;
    lastDiagnostics_.enabled=config_.diagnostics.enabled;
    if(inferenceEpisodes && (training_ || *inferenceEpisodes==0 || *inferenceEpisodes>1000000))
        throw std::invalid_argument("FlightControl3D: invalid inference episode budget");
    if(inferenceEpisodes && *inferenceEpisodes-1U>std::numeric_limits<std::uint32_t>::max()-config_.evaluation.seed)
        throw std::invalid_argument("FlightControl3D: inference episode seeds overflow");
    inferenceEpisodeLimit_=inferenceEpisodes.value_or(config_.evaluation.episodes);
    if(!training_) models_.LoadBest(learner_);
    else if(config_.training.resume && std::filesystem::exists(models_.Latest()))
    {
        learner_.Load(models_.Latest());
        resumed_=true;
        if(auto const level=LoadCurriculumLevel(models_.CurriculumLevel())) curriculum_.Restore(*level);
    }
    best_=models_.BestMetrics(training_);
    trainingUpdateLimit_=budget==TrainingBudget::AdditionalUpdates
        ? AdditionalUpdateLimit(learner_.Stats().updates,config_.training.maxUpdates) : config_.training.maxUpdates;
    CreateTrainingEnvironment();
    if(training_ && learner_.Stats().updates>=trainingUpdateLimit_) StartValidation();
}
void FlightSession::ContinueTraining()
{
    if(!training_ || !finished_) return;
    trainingUpdateLimit_=AdditionalUpdateLimit(learner_.Stats().updates,config_.training.maxUpdates);
    finished_=false; pendingReset_=true;
}
void FlightSession::ResetEpisode()
{
    if (!finished_)
    {
        if(training_) environment_->Reset();
        else ResetInferenceEpisode();
        episodeRewardSums_.fill(0);
    }
    pendingReset_=false;
}
void FlightSession::CaptureCrash(FlightEnvironment const& environment) noexcept
{
    if(environment.Mission().outcome!=Outcome::Crash) return;
    crashDiagnostic_=environment.CrashDetails();
    hasCrashDiagnostic_=true;
    if(!IsFinite(environment.Aircraft())) return;
    crashAircraft_=environment.Aircraft();
    crashRunway_=environment.Settings().runway;
    ++crashEvent_;
}
void FlightSession::Tick()
{
    if(pendingRewardConfig_ && !validationEnvironment_ && learner_.Collected()==0)
        ApplyPendingRewardTuning();
    if (finished_) return;
    if(validationEnvironment_) { TickValidation();return; }
    if (pendingReset_) ResetEpisode();
    auto const observation=environment_->ObservationNow();
    auto const action=learner_.Choose(observation,!training_);
    controls_=ToControls(action,config_.initial);
    lastStep_=environment_->Step(controls_);
    for(std::size_t i=0;i<RewardCount;++i) episodeRewardSums_[i]+=lastStep_.rewards[i];
    controls_=environment_->Mission().filteredControls;
    RecordDiagnostics(diagnostics_,observation,action,controls_,lastStep_,config_.diagnostics.saturationThreshold);
    if (training_) learner_.Record(observation,action,lastStep_,environment_->ObservationNow());
    if (lastStep_.terminated || lastStep_.truncated)
    {
        CaptureCrash(*environment_);
        lastEpisode_=environment_->Mission(); ++episodes_;
        if (lastEpisode_.touchedDown) ++touchdowns_;
        if (lastEpisode_.outcome==Outcome::Stopped) ++stops_;
        if (lastEpisode_.outcome==Outcome::Crash) ++crashes_;
        if (lastEpisode_.outcome==Outcome::Timeout) ++timeouts_;
        if (training_ && environment_->StartedAirborne()) AdvanceCurriculum(lastEpisode_);
        pendingReset_=true;
    }
    if (training_ && learner_.Ready()) OptimizeAndCheckpoint();
    if (!training_ && episodes_>=inferenceEpisodeLimit_) finished_=true;
}
void FlightSession::AdvanceCurriculum(MissionState const& episode)
{
    if(!curriculum_.Record(episode.outcome==Outcome::Stopped)) return;
    environment_->SetDifficulty(curriculum_.Difficulty());
    std::cout<<"[FLIGHT_CURRICULUM] level="<<curriculum_.Level()<<"/"<<curriculum_.Levels()-1
        <<" difficulty="<<curriculum_.Difficulty()<<" update="<<learner_.Stats().updates<<std::endl;
}
void FlightSession::SaveLatest()
{
    learner_.Save(models_.Latest());
    SaveCurriculumLevel(models_.CurriculumLevel(),curriculum_.Level());
}
void FlightSession::OptimizeAndCheckpoint()
{
    learner_.Optimize();
    LogUpdate(learner_.Stats());
    auto const updates=learner_.Stats().updates;
    bool const final=updates>=trainingUpdateLimit_;
    if(config_.diagnostics.enabled && (final || updates%config_.diagnostics.logEveryUpdates==0))
    {
        lastDiagnostics_=diagnostics_;
        LogDiagnostics(lastDiagnostics_,updates);
        diagnostics_=NextDiagnosticsWindow(diagnostics_);
    }
    if(pendingRewardConfig_) ApplyPendingRewardTuning();
    bool const validate=final || updates%config_.evaluation.everyUpdates==0;
    if(validate || updates%config_.training.checkpointEvery==0) SaveLatest();
    if(validate) StartValidation();
}
FlightSnapshot FlightSession::Snapshot() const noexcept
{
    auto const& active=validationEnvironment_ ? *validationEnvironment_ : *environment_;
    FlightSnapshot state{};
    state.runway=active.Settings().runway;
    state.aircraft=active.Aircraft();
    state.mission=active.Mission();
    state.lastEpisode=lastEpisode_;
    state.guidance=active.CurrentGuidance();
    state.controls=controls_;
    state.lastStep=lastStep_;
    state.learning=learner_.Stats();
    state.episodes=episodes_;state.touchdowns=touchdowns_;state.stops=stops_;
    state.crashes=crashes_;state.timeouts=timeouts_;
    state.glideDegrees=config_.mission.glideDegrees;
    state.aimPointAlongMeters=AimPointAlongMeters(active.Settings());
    state.stepSeconds=config_.StepSeconds();
    state.gearDownMinimum=config_.mission.gearDownMinimum;
    state.training=training_;state.finished=finished_;
    state.elapsedSeconds=static_cast<double>(state.mission.steps)*active.Settings().mission.physicsDt;
    state.maximumEpisodeSeconds=active.Settings().mission.maxSeconds;
    state.lastEpisodeSeconds=static_cast<double>(state.lastEpisode.steps)*config_.mission.physicsDt;
    state.validation=validation_;state.lastValidation=lastValidation_;state.validationCampaigns=validationCampaigns_;
    state.validating=static_cast<bool>(validationEnvironment_);
    state.hasBest=best_.has_value();
    state.bestScore=best_ ? best_->Score() : 0;
    state.bestDifficulty=best_ ? best_->difficulty : 0;
    state.bestUpdate=best_ ? best_->update : 0;
    state.trainingUpdateLimit=trainingUpdateLimit_;state.resumed=resumed_;
    state.gateSuccesses=lastValidation_.stops;state.gateEvaluated=lastValidation_.episodes;
    state.curriculumLevel=curriculum_.Level();state.curriculumLevels=curriculum_.Levels();
    state.curriculumWindowStops=curriculum_.WindowStops();state.curriculumWindowEpisodes=curriculum_.WindowEpisodes();
    state.diagnostics=diagnostics_;state.lastDiagnostics=lastDiagnostics_;
    state.seaHeightMeters=active.Settings().runwayRandomization.seaHeightMeters;
    state.crashEvent=crashEvent_;
    state.crashAircraft=crashAircraft_;
    state.crashRunway=crashRunway_;
    state.crashDiagnostic=crashDiagnostic_;
    state.hasCrashDiagnostic=hasCrashDiagnostic_;
    state.rewardSettings=pendingRewardConfig_ ? pendingRewardConfig_->reward : config_.reward;
    state.activeRewardSettings=config_.reward;
    state.rewardTuningPending=pendingRewardConfig_.has_value();
    state.rewardTuningStatus=rewardTuningStatus_;
    state.rewardTuningRevisions=rewardTuningRevisions_;
    state.episodeRewardSums=validationEnvironment_ ? validationEpisodeRewardSums_ : episodeRewardSums_;
    state.approachProfileValid=IsFinite(state.aircraft) && std::isfinite(state.guidance.along)
        && std::isfinite(state.guidance.height);
    return state;
}
bool FlightSession::SetRewardTuning(std::size_t index,double value) noexcept
{
    try
    {
        if(!training_ || scenarioPath_.empty())
            throw std::runtime_error("Reward editing is unavailable for this session");
        auto const& current=pendingRewardConfig_ ? *pendingRewardConfig_ : config_;
        if(index>=RewardTuningCount) throw std::out_of_range("Unknown reward setting");
        if(RewardTuningValue(current.reward,index)==value)
        {
            ++rewardTuningRevisions_[index];
            return true;
        }
        auto next=WithRewardTuning(current,index,value);
        SaveRewardTuning(scenarioPath_,index,value);
        pendingRewardConfig_=std::move(next);
        std::snprintf(rewardTuningStatus_.data(),rewardTuningStatus_.size(),
            "Saved to JSON; waiting for PPO rollout boundary");
        if(!validationEnvironment_ && learner_.Collected()==0) ApplyPendingRewardTuning();
        ++rewardTuningRevisions_[index];
        return true;
    }
    catch(std::exception const& error)
    {
        std::snprintf(rewardTuningStatus_.data(),rewardTuningStatus_.size(),"Reward edit rejected: %s",error.what());
        if(index<RewardTuningCount) ++rewardTuningRevisions_[index];
        return false;
    }
}
void FlightSession::ApplyPendingRewardTuning()
{
    if(!pendingRewardConfig_ || validationEnvironment_ || learner_.Collected()!=0) return;
    auto const& next=*pendingRewardConfig_;
    learner_.SetRewardContract(next.reward,next.checkpointSignature);
    models_.SetRewardContract(next.reward,next.checkpointSignature);
    environment_->SetRewardSettings(next.reward);
    config_.reward=next.reward;
    config_.checkpointSignature=next.checkpointSignature;
    pendingRewardConfig_.reset();
    best_.reset();lastValidation_={};validationCampaigns_=0;
    std::snprintf(rewardTuningStatus_.data(),rewardTuningStatus_.size(),"Saved and active; BEST requires revalidation");
}
void FlightSession::StartValidation()
{
    validation_={};validation_.update=learner_.Stats().updates;
    // The campaign follows the curriculum: it evaluates the mix being trained.
    validation_.difficulty=curriculum_.Difficulty();
    validationPendingReset_=false;lastStep_={};controls_={};
    validationEnvironment_=std::make_unique<FlightEnvironment>(config_,DefaultDataRoot(),config_.evaluation.seed);
    ResetValidationEpisode();
}
void FlightSession::ResetValidationEpisode()
{
    auto const seed=config_.evaluation.seed+static_cast<std::uint32_t>(validation_.episodes);
    auto const start=EvaluationStart(validation_.episodes,config_.evaluation.episodes,
        config_.training.curriculum.rolloutStartShare,validation_.difficulty);
    validationEnvironment_->Reset(seed,start);
    validationEpisodeRewardSums_.fill(0);
    std::cout<<"[FLIGHT_EVALUATION_EPISODE] episode="<<(validation_.episodes+1U)<<" seed="<<seed
        <<" start="<<(start.rolling ? "rolling" : "airborne")<<" difficulty="<<start.difficulty<<std::endl;
}
void FlightSession::ResetInferenceEpisode()
{
    // Inference evaluates the full curriculum mix, from rolling starts to the hardest approaches.
    auto const seed=config_.evaluation.seed+static_cast<std::uint32_t>(episodes_);
    constexpr double FullTaskDifficulty=1.0;
    environment_->Reset(seed,EvaluationStart(episodes_,inferenceEpisodeLimit_,
        config_.training.curriculum.rolloutStartShare,FullTaskDifficulty));
}
void FlightSession::TickValidation()
{
    auto& evaluation=*validationEnvironment_;
    if(validationPendingReset_)
    {
        ResetValidationEpisode();validationPendingReset_=false;
    }
    auto const observation=evaluation.ObservationNow();
    auto const action=learner_.Choose(observation,true);
    controls_=ToControls(action,config_.initial);
    lastStep_=evaluation.Step(controls_);
    for(std::size_t i=0;i<RewardCount;++i) validationEpisodeRewardSums_[i]+=lastStep_.rewards[i];
    controls_=evaluation.Mission().filteredControls;
    UpdateLiveDiagnostics(diagnostics_,observation,action,controls_);
    // No Record/Optimize here: training rollout, optimizer and RNG remain frozen.
    if(!lastStep_.terminated && !lastStep_.truncated) return;
    CaptureCrash(evaluation);
    validation_.Record(evaluation.Mission());
    if(validation_.episodes<config_.evaluation.episodes) validationPendingReset_=true;
    else FinishValidation();
}
void FlightSession::FinishValidation()
{
    models_.Promote(learner_,validation_);
    best_=models_.BestMetrics(true);
    lastValidation_=validation_;
    ++validationCampaigns_;
    validationEnvironment_.reset();
    lastStep_={};controls_={};
    finished_=learner_.Stats().updates>=trainingUpdateLimit_;
    if(pendingRewardConfig_) ApplyPendingRewardTuning();
}
void FlightSession::CreateTrainingEnvironment()
{
    auto const seed=training_ && config_.training.randomizeEnvironmentSeed
        ? std::random_device{}()
        : training_ ? TrainingSeed(config_.training.seed,learner_.Stats().updates) : config_.evaluation.seed;
    std::cout<<"[FLIGHT_DYNAMICS] mode="<<(training_ ? "training" : "inference")
        <<" aircraft="<<config_.aircraft<<" environment_seed="<<seed<<std::endl;
    environment_=std::make_unique<FlightEnvironment>(config_,DefaultDataRoot(),seed);
    environment_->SetDifficulty(training_ ? curriculum_.Difficulty() : 1.0);
    episodeRewardSums_.fill(0);
    if(training_) environment_->Reset();
    else ResetInferenceEpisode();
}
} // namespace ppo::examples::flight
