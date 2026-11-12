#include "flight_session.hpp"
#include <utility>

namespace ppo::examples::flight {
FlightSession::FlightSession(Config config, bool training, std::filesystem::path modelDirectory,std::optional<std::size_t> inferenceEpisodes)
    : config_(std::move(config)),training_(training),models_(config_,std::move(modelDirectory)),
      environment_(config_,DefaultDataRoot(),training ? config_.training.seed : config_.evaluation.seed),learner_(config_)
{
    if(inferenceEpisodes && (training_ || *inferenceEpisodes==0 || *inferenceEpisodes>1000000))
        throw std::invalid_argument("FlightControl3D: invalid inference episode budget");
    inferenceEpisodeLimit_=inferenceEpisodes.value_or(config_.evaluation.episodes);
    best_=models_.BestMetrics();
    if(!training_) models_.LoadBest(learner_);
    else if(config_.training.resume) learner_.Load(models_.Latest());
    environment_.Reset();
    if(training_ && learner_.Stats().updates>=config_.training.maxUpdates) StartValidation();
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
    controls_=ToControls(action);
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
        bool const final=learner_.Stats().updates>=config_.training.maxUpdates;
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
        config_.mission.glideDegrees,config_.mission.ilsRangeMeters,config_.StepSeconds(),training_,finished_,false};
    state.validation=validation_;state.lastValidation=lastValidation_;state.validationCampaigns=validationCampaigns_;
    state.validating=static_cast<bool>(validationEnvironment_);state.hasBest=best_.has_value();state.bestScore=best_ ? best_->Score() : 0;
    return state;
}
void FlightSession::StartValidation()
{
    validation_={};validation_.update=learner_.Stats().updates;
    validationPendingReset_=false;lastStep_={};controls_={};
    validationEnvironment_=std::make_unique<FlightEnvironment>(config_,DefaultDataRoot(),config_.evaluation.seed);
    validationEnvironment_->Reset();
}
void FlightSession::TickValidation()
{
    auto& evaluation=*validationEnvironment_;
    if(validationPendingReset_) { evaluation.Reset();validationPendingReset_=false; }
    auto const action=learner_.Choose(evaluation.ObservationNow(),true);
    controls_=ToControls(action);
    lastStep_=evaluation.Step(controls_);
    // No Record/Optimize here: training rollout, optimizer and RNG remain frozen.
    if(lastStep_.terminated || lastStep_.truncated)
    {
        validation_.Record(evaluation.Mission());
        if(validation_.episodes<config_.evaluation.episodes) validationPendingReset_=true;
        else
        {
            models_.Promote(learner_,validation_);
            best_=models_.BestMetrics();lastValidation_=validation_;++validationCampaigns_;
            validationEnvironment_.reset();
            lastStep_={};controls_={};
            finished_=learner_.Stats().updates>=config_.training.maxUpdates;
        }
    }
}
} // namespace ppo::examples::flight
