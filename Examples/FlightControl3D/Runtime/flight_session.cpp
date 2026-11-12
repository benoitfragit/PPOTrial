#include "flight_session.hpp"
#include <utility>

namespace ppo::examples::flight {
FlightSession::FlightSession(Config config, bool training, std::filesystem::path checkpoint)
    : config_(std::move(config)),training_(training),checkpoint_(std::move(checkpoint)),
      environment_(config_,DefaultDataRoot(),training ? config_.training.seed : config_.evaluation.seed),learner_(config_)
{
    if (!training_ || config_.training.resume) learner_.Load(checkpoint_);
    environment_.Reset();
    finished_=training_ && learner_.Stats().updates>=config_.training.maxUpdates;
}
void FlightSession::ResetEpisode()
{
    if (!finished_) environment_.Reset();
    pendingReset_=false;
}
void FlightSession::Tick()
{
    if (finished_) return;
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
        finished_=learner_.Stats().updates>=config_.training.maxUpdates;
        if (finished_ || learner_.Stats().updates%config_.training.checkpointEvery==0) learner_.Save(checkpoint_);
    }
    if (!training_ && episodes_>=config_.evaluation.episodes) finished_=true;
}
FlightSnapshot FlightSession::Snapshot() const noexcept
{
    return {environment_.Settings().runway,environment_.Plan(),environment_.Aircraft(),environment_.Mission(),lastEpisode_,
        environment_.CurrentGuidance(),controls_,lastStep_,learner_.Stats(),episodes_,touchdowns_,stops_,crashes_,timeouts_,
        config_.mission.glideDegrees,config_.mission.ilsRangeMeters,config_.StepSeconds(),training_,finished_,false};
}
} // namespace ppo::examples::flight
