#include "flight_example.hpp"
#include "flight_session.hpp"
#include "environment.hpp"
#include <iostream>

namespace ppo::examples::flight {
namespace {
/** Kinematic mirror only: Entity::Run must not apply the legacy planar integrator. */
class AircraftEntity final : public ppo::model::Entity
{
public:
    explicit AircraftEntity(entt::registry* registry)
        : Entity(registry,ppo::model::eEntitySide::Friend,ppo::model::eEntityType::Mobile)
    { add<ppo::model::sEntityMetadata>(ppo::model::sEntityMetadata{"JSBSim Cessna 310","FlightControl3D"}); }
    void Run() noexcept override {}
    void Init() noexcept override {}
};

class FlightAgent final : public ppo::model::IAgent
{
    std::shared_ptr<FlightSession> session_;
    std::shared_ptr<AircraftEntity> entity_;
    Config config_;
    unsigned long long id_{ppo::model::GetNewUniqId()};
    bool failed_{};
public:
    FlightAgent(std::shared_ptr<FlightSession> session, std::shared_ptr<AircraftEntity> entity, Config config)
        : session_(std::move(session)),entity_(std::move(entity)),config_(std::move(config)) {}
    void Init() override
    {
        // Never discard a partial on-policy rollout on a desktop Stop/Play.
        // A stopped worker resumes this same session at its next transition.
        try { session_->ContinueTraining(); }
        catch(std::exception const& error)
        { failed_=true; std::cerr<<"[FLIGHT_ERROR] "<<error.what()<<'\n'; }
    }
    void ChooseAction() override
    {
        if (failed_) return;
        try { session_->Tick(); }
        catch(std::exception const& error)
        { failed_=true; std::cerr<<"[FLIGHT_ERROR] "<<error.what()<<'\n'; }
    }
    bool SetNewState() override
    {
        auto const flight=session_->Snapshot();
        auto& state=entity_->registry()->get<ppo::model::sEntityState>(entity_->uniqid());
        state.m_x=static_cast<float>(flight.aircraft.position.north);
        state.m_y=static_cast<float>(flight.aircraft.position.east);
        state.m_dx=static_cast<float>(flight.aircraft.velocity.north);
        state.m_dy=static_cast<float>(flight.aircraft.velocity.east);
        state.m_speed=static_cast<float>(flight.aircraft.airspeed);
        state.m_rotation=static_cast<float>(flight.aircraft.heading);
        state.m_destroyed=flight.mission.outcome==Outcome::Crash;
        return false; // Session owns episode resets, not the generic ECS rollout scheduler.
    }
    void ProcessCommand(ppo::model::sAgentRealTimeCommand const&) noexcept override {}
    ppo::model::sAgentStateSnaphot GetSnapshot() const noexcept override
    {
        auto const state=session_->Snapshot();
        ppo::model::sAgentStateSnaphot result{};
        result.m_name="FlightControl3D hybrid PPO"; result.m_id=id_;
        result.m_common.m_training=state.training && !state.finished && !state.validating && !failed_;
        result.m_common.m_evaluationinprogress=state.validating;
        result.m_common.m_completedEvaluations=state.validationCampaigns;
        result.m_common.m_notation=static_cast<float>(state.bestScore);
        result.m_common.m_hasLatestEvaluation=state.validationCampaigns>0;
        result.m_common.m_latestEvaluationNotation=static_cast<float>(state.lastValidation.Score());
        result.m_common.m_latestEvaluationSuccessRate=static_cast<float>(state.lastValidation.SuccessRate());
        result.m_common.m_latestEvaluationTestCount=static_cast<unsigned int>(config_.evaluation.episodes);
        result.m_common.m_evaluationTestsFinished=static_cast<unsigned int>(state.validation.episodes);
        result.m_common.m_evaluationprogress=static_cast<float>(state.validation.episodes)/static_cast<float>(config_.evaluation.episodes);
        result.m_common.m_done=state.lastStep.terminated || state.lastStep.truncated;
        result.m_common.m_version=static_cast<unsigned int>(state.learning.updates);
        result.m_common.m_step=static_cast<long long>(session_->Learner().Collected());
        result.m_common.m_maxStep=static_cast<long long>(config_.training.rolloutSteps);
        result.m_common.m_reward=static_cast<float>(state.lastStep.reward);
        result.m_common.m_actorloss=static_cast<float>(state.learning.policyLoss);
        result.m_common.m_valueloss=static_cast<float>(state.learning.valueLoss);
        result.m_common.m_entropyloss=static_cast<float>(state.learning.entropy);
        for (std::size_t i=0;i<RewardCount;++i)
        {
            result.m_rewardnames.emplace_back(RewardNames[i]);
            result.m_scalings.push_back(static_cast<float>(config_.reward.weights[i]));
            result.m_rewards.push_back(static_cast<float>(state.lastStep.rewards[i]));
        }
        return result;
    }
    bool Failed() const noexcept { return failed_; }
};

class FlightWorldRules final : public ppo::model::IWorldRules
{
    std::shared_ptr<FlightSession> session_;
    std::shared_ptr<FlightAgent> agent_;
public:
    FlightWorldRules(std::shared_ptr<FlightSession> session,std::shared_ptr<FlightAgent> agent)
        : session_(std::move(session)),agent_(std::move(agent)) {}
    void BeforeStep(ppo::model::WorldEntities&) override {}
    void Resolve(ppo::model::WorldEntities&) override {}
    void AfterStep() override {}
    void CopySnapshot(entt::registry&,entt::registry& destination) const override
    {
        auto state=session_->Snapshot(); state.failed=agent_->Failed();
        destination.ctx().emplace<FlightSnapshot>(state);
        destination.ctx().get<ppo::model::sEnvironmentState>().m_dt=static_cast<float>(state.stepSeconds);
    }
};
void RegisterTypes() {} // Direct typed construction; no legacy agent factories are needed.
bool Load(std::filesystem::path const& path,ppo::model::Environment& env,bool training)
{
    try
    {
        auto const config=LoadConfig(path);
        auto const directory=path.parent_path().parent_path().parent_path()/config.training.checkpointDirectory;
        auto session=std::make_shared<FlightSession>(config,training,directory,std::nullopt,TrainingBudget::AdditionalUpdates);
        auto const initial=session->Snapshot();
        std::cout<<"[FLIGHT_SESSION] "<<(training ? (initial.resumed ? "Resumed latest" : "New policy") : "Validated BEST")
            <<" | update="<<initial.learning.updates<<" | limit="<<initial.trainingUpdateLimit<<std::endl;
        auto entity=env.CreateEntity<AircraftEntity>();
        auto agent=env.CreateAgent<FlightAgent>(session,entity,config);
        agent->SetNewState();
        return env.SetWorldRules(std::make_unique<FlightWorldRules>(session,agent));
    }
    catch(std::exception const& error) { std::cerr<<"[FLIGHT_ERROR] "<<error.what()<<'\n'; return false; }
}
bool LoadTraining(std::filesystem::path const& path,ppo::model::Environment& env) { return Load(path,env,true); }
bool LoadInference(std::filesystem::path const& path,ppo::model::Environment& env) { return Load(path,env,false); }
}
ppo::app::ExampleDefinition Definition()
{
    return {"flight-control-3d","FLIGHT CONTROL 3D - JSBSim / PPO", "Scenarios/FlightControl3D.json",
        &RegisterTypes,&LoadTraining,"Native JSBSim C310: randomized airborne start, racetrack, moving runway ILS and hybrid PPO."};
}
ppo::app::ExampleDefinition InferenceDefinition()
{
    return {"flight-control-3d-inference","FLIGHT CONTROL 3D - JSBSim / INFERENCE", "Scenarios/FlightControl3D.json",
        &RegisterTypes,&LoadInference,"Evaluate a FlightControl3D checkpoint on independent randomized starts without training."};
}
} // namespace ppo::examples::flight
