#include "flight_session.hpp"
#include <iostream>
#include <utility>
#include <limits>

namespace ppo::examples::flight {
namespace {
std::uint64_t AdditionalUpdateLimit(std::uint64_t updates,std::size_t additional)
{
    if(updates>std::numeric_limits<std::uint64_t>::max()-additional)
        throw std::overflow_error("FlightControl3D: training update budget overflow");
    return updates+additional;
}
Config EnvironmentConfig(Config config,bool training)
{
    if(!training) config.training.locReminderProbability=0;
    // Capture lessons terminate at their objective: widening that objective
    // teaches early exits that cannot pass the nominal independent gate.
    if(!training || IsCaptureLesson(config.training.curriculumStage)) return config;
    config.mission.localizerCaptureDegrees*=config.training.ilsCaptureScale;
    config.mission.glideCaptureDegrees*=config.training.ilsCaptureScale;
    config.training.ilsCaptureScale=1.0;
    return config;
}
/** @brief Emits one complete rollout's losses and weighted reward-channel means. */
void LogUpdate(LearningStats const& stats)
{
    std::cout<<"[FLIGHT_UPDATE] update="<<stats.updates
        <<" mean_reward="<<stats.meanReward
        <<" policy_loss="<<stats.policyLoss
        <<" value_loss="<<stats.valueLoss
        <<" entropy="<<stats.entropy
        <<" approx_kl="<<stats.maximumApproximateKl<<" actor_steps="<<stats.actorSteps
        <<" kl_stop="<<stats.actorStoppedByKl
        <<" policy_kl="<<stats.acceptedPolicyKl<<" proposed_kl="<<stats.maximumProposedKl
        <<" actor_rejections="<<stats.actorStepRejections;
    for(std::size_t channel=0;channel<RewardCount;++channel)
        std::cout<<' '<<RewardNames[channel]<<'='<<stats.meanRewards[channel];
    std::cout<<std::endl;
}
} // namespace
FlightSession::FlightSession(Config config, bool training, std::filesystem::path modelDirectory,
    std::optional<std::size_t> inferenceEpisodes,TrainingBudget budget)
    : config_(std::move(config)),training_(training),models_(config_,std::move(modelDirectory)),
      environment_(EnvironmentConfig(config_,training_),DefaultDataRoot(),
          training ? config_.training.seed : config_.evaluation.seed),learner_(config_)
{
    if(inferenceEpisodes && (training_ || *inferenceEpisodes==0 || *inferenceEpisodes>1000000))
        throw std::invalid_argument("FlightControl3D: invalid inference episode budget");
    inferenceEpisodeLimit_=inferenceEpisodes.value_or(config_.evaluation.episodes);
    best_=models_.BestMetrics(training_);
    if(!training_) models_.LoadBest(learner_);
    else if(config_.training.resume && std::filesystem::exists(models_.Latest()))
    { learner_.Load(models_.Latest()); resumed_=true; }
    trainingUpdateLimit_=budget==TrainingBudget::AdditionalUpdates
        ? AdditionalUpdateLimit(learner_.Stats().updates,config_.training.maxUpdates) : config_.training.maxUpdates;
    environment_.Reset();
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
    if (!finished_) environment_.Reset();
    pendingReset_=false;
}
void FlightSession::Tick()
{
    if (finished_) return;
    if(validationEnvironment_) { TickValidation();return; }
    if (pendingReset_) ResetEpisode();
    auto const observation=environment_.ObservationNow();
    auto const action=learner_.Choose(observation,!training_);
    controls_=ToControls(action,config_.initial.throttle);
    lastStep_=environment_.Step(controls_);
    if (training_) learner_.Record(observation,action,lastStep_,environment_.ObservationNow());
    if (lastStep_.terminated || lastStep_.truncated)
    {
        lastEpisode_=environment_.Mission(); ++episodes_;
        if (lastEpisode_.touchedDown) ++touchdowns_;
        if (lastEpisode_.outcome==Outcome::Stopped) ++stops_;
        if (lastEpisode_.outcome==Outcome::Crash) ++crashes_;
        if (lastEpisode_.outcome==Outcome::Timeout) ++timeouts_;
        pendingReset_=true;
    }
    if (training_ && learner_.Ready())
    {
        learner_.Optimize();
        LogUpdate(learner_.Stats());
        bool const final=learner_.Stats().updates>=trainingUpdateLimit_;
        bool const validate=final || learner_.Stats().updates%config_.evaluation.everyUpdates==0;
        if(validate || learner_.Stats().updates%config_.training.checkpointEvery==0) learner_.Save(models_.Latest());
        if(validate) StartValidation();
    }
    if (!training_ && episodes_>=inferenceEpisodeLimit_) finished_=true;
}
FlightSnapshot FlightSession::Snapshot() const noexcept
{
    auto const& active=validationEnvironment_ ? *validationEnvironment_ : environment_;
    FlightSnapshot state{active.Settings().runway,active.Plan(),active.Aircraft(),active.Mission(),lastEpisode_,
        active.CurrentGuidance(),controls_,lastStep_,learner_.Stats(),episodes_,touchdowns_,stops_,crashes_,timeouts_,
        config_.mission.glideDegrees,config_.mission.ilsRangeMeters,config_.StepSeconds(),
        config_.mission.gearDownMinimum,training_,finished_,false};
    state.elapsedSeconds=static_cast<double>(state.mission.steps)*active.Settings().mission.physicsDt;
    state.maximumEpisodeSeconds=active.Settings().mission.maxSeconds;
    state.lastEpisodeSeconds=static_cast<double>(state.lastEpisode.steps)*config_.mission.physicsDt;
    state.validation=validation_;state.lastValidation=lastValidation_;state.validationCampaigns=validationCampaigns_;
    state.validating=static_cast<bool>(validationEnvironment_);state.hasBest=best_.has_value();state.bestScore=best_ ? best_->Score() : 0;
    state.localizerCaptureDegrees=active.Settings().mission.localizerCaptureDegrees;
    state.glideCaptureDegrees=active.Settings().mission.glideCaptureDegrees;
    state.trainingUpdateLimit=trainingUpdateLimit_;state.resumed=resumed_;
    state.curriculumStage=config_.training.curriculumStage;
    state.gateSuccesses=lastGateSuccesses_;state.gateEvaluated=lastGateEvaluated_;
    state.episodeStage=active.Settings().training.curriculumStage;
    state.locRetentionSuccesses=lastLocRetentionSuccesses_;
    state.locRetentionEvaluated=lastLocRetentionEvaluated_;
    return state;
}
void FlightSession::StartValidation()
{
    validatingLocRetention_=false;
    lastLocRetentionSuccesses_=0;lastLocRetentionEvaluated_=0;
    BeginValidationStage(config_.training.curriculumStage);
}
void FlightSession::BeginValidationStage(CurriculumStage stage)
{
    validation_={};validation_.update=learner_.Stats().updates;
    validationPendingReset_=false;lastStep_={};controls_={};
    validationGateSuccesses_=0;
    auto validationConfig=EnvironmentConfig(config_,false);
    validationConfig.training.curriculumStage=stage;
    validationEnvironment_=std::make_unique<FlightEnvironment>(validationConfig,DefaultDataRoot(),config_.evaluation.seed);
    validationEnvironment_->Reset();
}
void FlightSession::TickValidation()
{
    auto& evaluation=*validationEnvironment_;
    if(validationPendingReset_)
    {
        evaluation.Reset();validationPendingReset_=false;
    }
    auto const action=learner_.Choose(evaluation.ObservationNow(),true);
    controls_=ToControls(action,config_.initial.throttle);
    lastStep_=evaluation.Step(controls_);
    // No Record/Optimize here: training rollout, optimizer and RNG remain frozen.
    auto const stage=validatingLocRetention_ ? CurriculumStage::Loc : config_.training.curriculumStage;
    if(lastStep_.terminated || lastStep_.truncated)
    {
        if(IsCaptureLesson(stage) && evaluation.Settings().training.curriculumStage==stage
            && evaluation.Mission().outcome==Outcome::CaptureCompleted)
            ++validationGateSuccesses_;
        validation_.Record(evaluation.Mission());
        if(validation_.episodes<config_.evaluation.episodes) validationPendingReset_=true;
        else
        {
            if(validatingLocRetention_)
            {
                lastLocRetentionSuccesses_=validationGateSuccesses_;
                lastLocRetentionEvaluated_=validation_.episodes;
            }
            else
            {
                // Intermediate capture gates never promote a landing BEST.
                if(!IsCaptureLesson(stage)) models_.Promote(learner_,validation_);
                best_=models_.BestMetrics(true);lastValidation_=validation_;
                lastGateSuccesses_=IsCaptureLesson(stage) ? validationGateSuccesses_ : validation_.stops;
                lastGateEvaluated_=validation_.episodes;
                if(stage==CurriculumStage::Glide)
                {
                    // Report retention separately; LOC captures cannot inflate the Glide gate.
                    validatingLocRetention_=true;
                    BeginValidationStage(CurriculumStage::Loc);
                    return;
                }
            }
            ++validationCampaigns_;
            validationEnvironment_.reset();
            validatingLocRetention_=false;
            lastStep_={};controls_={};
            finished_=learner_.Stats().updates>=trainingUpdateLimit_;
        }
    }
}
} // namespace ppo::examples::flight
