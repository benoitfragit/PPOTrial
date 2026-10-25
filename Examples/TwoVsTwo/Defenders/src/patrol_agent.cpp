#include "patrol_agent.hpp"
#include "patrol_policy.hpp"
#include "patrol_training_ecs_agent.hpp"
#include "patrol_learner.hpp"
#include "patrol_checkpoint.hpp"
#include <iostream>
#include <sstream>
#include "initializer.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ppo::examples::twovstwo::defenders {
std::shared_ptr<PatrolPolicy> LoadPatrolPolicy(std::filesystem::path const& checkpoint)
{
    return std::make_shared<PatrolPolicy>(checkpoint);
}

std::shared_ptr<PatrolPolicy> CreateUntrainedPatrolPolicy(std::uint32_t seed)
{
    return std::make_shared<PatrolPolicy>(seed);
}

namespace {

Mobile FromEcs(ppo::model::BaseEntity& entity)
{
    auto [state] = entity.get<ppo::model::sEntityState>();
    float turnRate = 0.F;
    if (entity.has<ppo::model::sFlightModelState>())
        {
        auto [flight] = entity.get<ppo::model::sFlightModelState>();
        turnRate = flight.m_accpsi;
    }
    return {state.m_x, state.m_y, state.m_rotation, state.m_speed,
            turnRate, !state.m_destroyed};
}

} // namespace

PatrolProdAgent::PatrolProdAgent(ppo::model::BaseEntity& own,
                             ppo::model::BaseEntity& teammate,
                             std::size_t role, std::shared_ptr<PatrolPolicy> actor)
    : own_(own.registry(), own.uniqid()),
      teammate_(teammate.registry(), teammate.uniqid()),
      role_(role), actor_(std::move(actor))
{
    if (!actor_ || role_ >= 2U || own_.registry() != teammate_.registry())
        throw std::invalid_argument("Invalid Blue PPO agent arguments");
}

void PatrolProdAgent::Init()
{
    step_ = 0U;
    done_ = false;
    hasBeforeStep_ = false;
}

Parameters PatrolProdAgent::ReadParameters()
{
    Parameters p{};
    auto* registry = own_.registry();
    if (registry && registry->ctx().contains<ppo::model::sEnvironmentState>())
    {
        auto const& env = registry->ctx().get<ppo::model::sEnvironmentState>();
        p.halfNorth = 0.5F * env.m_maxNorth;
        p.halfEast = 0.5F * env.m_maxEast;
        p.dt = env.m_dt;
    }
    if (own_.has<ppo::model::sFlightModelState>())
    {
        auto const [flight] = own_.get<ppo::model::sFlightModelState>();
        p.maxSpeed = flight.m_maxSpeed;
        p.maxAcceleration = flight.m_maxAcc;
        p.maxRotationRate = flight.m_maxAccRot;
    }
    return p;
}

State PatrolProdAgent::ReadState(Parameters const& p)
{
    auto* registry = own_.registry();
    auto const& context = registry->ctx().get<InitializerContext>();
    State result{};
    result.side = context.side;
    result.arclength = context.arclength + static_cast<float>(step_) * p.dt * p.cruiseSpeed;
    result.controlledRole = role_;
    result.step = step_;
    result.guards[role_] = FromEcs(own_);
    result.guards[1U - role_] = FromEcs(teammate_);
    result.failed = false;
    return result;
}

void PatrolProdAgent::ChooseAction()
{
    if (done_ || !own_.has<ppo::model::sEntityState, ppo::model::sSNAControlState>())
        return;
    Parameters const p = ReadParameters();
    if (!Valid(p))
        return;
    beforeStep_ = ReadState(p);
    hasBeforeStep_ = true;
    auto const observation = Observe(p, beforeStep_);
    auto const action = actor_->Predict(observation);
    auto [control] = own_.get<ppo::model::sSNAControlState>();
    control.m_rotSpeedCmd = action[0];
    control.m_accCmd = action[1];
}

bool PatrolProdAgent::SetNewState()
{
    if (done_)
        return true;
    ++step_;
    auto const p = ReadParameters();
    if (!Valid(p))
        return done_ = true;
    State const after = ReadState(p);
    done_ = step_ >= p.episodeSteps
        || (hasBeforeStep_ && GuardsCollided(p, beforeStep_, after));
    for (auto const& car : after.guards)
        done_ |= !car.available || !std::isfinite(car.north) || !std::isfinite(car.east)
            || std::abs(car.north) >= p.halfNorth || std::abs(car.east) >= p.halfEast;
    return done_;
}

ppo::model::sAgentStateSnaphot PatrolProdAgent::GetSnapshot() const noexcept
{
    ppo::model::sAgentStateSnaphot snapshot{};
    snapshot.m_name = role_ == 0U ? "Patrol 1 PPO" : "Patrol 2 PPO";
    snapshot.m_name += actor_->IsTrained() ? " (checkpoint)" : " (untrained)";
    snapshot.m_id = id_;
    snapshot.m_common.m_step = static_cast<long long>(step_);
    snapshot.m_common.m_maxStep = 4096;
    snapshot.m_common.m_done = done_;
    snapshot.m_common.m_training = false;
    return snapshot;
}
} // namespace ppo::examples::twovstwo::defenders

// Only this translation unit instantiates the RLtools CPU backend for the GUI.
namespace ppo::examples::twovstwo::defenders {

PatrolTrainingAgent::PatrolTrainingAgent(
    ppo::model::BaseEntity& own, ppo::model::BaseEntity& teammate,
    std::size_t role, std::shared_ptr<PatrolPolicy> policy,
    Parameters parameters, std::filesystem::path checkpoint,
    std::uint32_t seed, std::uint64_t maxUpdates)
    : PatrolProdAgent(own, teammate, role, policy),
      policy_(std::move(policy)), parameters_(parameters),
      checkpoint_(std::move(checkpoint)), seed_(seed), maxUpdates_(maxUpdates)
{
    if (!Valid(parameters_) || !policy_ || checkpoint_.empty() || maxUpdates_ == 0)
        throw std::invalid_argument("Invalid patrol PPO training agent configuration");
}

PatrolTrainingAgent::~PatrolTrainingAgent()
{
    Save();
}

void PatrolTrainingAgent::Init()
{
    // Reset physical episode state only. Keep optimizer and unfinished rollout.
    PatrolProdAgent::Init();
}

void PatrolTrainingAgent::ChooseAction()
{
    if (finished_)
    {
        PatrolProdAgent::ChooseAction();
        return;
    }
    if (done_ || !own_.has<ppo::model::sEntityState, ppo::model::sSNAControlState>())
        return;
    auto const parameters = ReadParameters();
    if (!Valid(parameters))
        return;

    try
    {
        if (!learner_)
            learner_ = std::make_unique<PatrolLearner>(parameters_, seed_);

        beforeStep_ = ReadState(parameters);
        hasBeforeStep_ = true;
        // This is the actual action that the ECS/SNA world will execute.
        lastAction_ = learner_->ChooseAction(Observe(parameters, beforeStep_));
        auto [control] = own_.get<ppo::model::sSNAControlState>();
        control.m_rotSpeedCmd = Clamp(lastAction_[0], -1.F, 1.F);
        control.m_accCmd = Clamp(lastAction_[1], -1.F, 1.F);
    }
    catch (std::exception const& e)
    {
        std::cerr << "[PATROL] ECS PPO action failed: " << e.what() << '\n';
        finished_ = true;
        PatrolProdAgent::ChooseAction();
    }
}

void PatrolTrainingAgent::Save() noexcept
{
    if (!learner_ || updates_ == 0U)
        return;
    try
    {
        learner_->WithActor([&](auto const& actor) {
            SavePatrolPolicy(checkpoint_, actor);
        });
    }
    catch (std::exception const& e)
    {
        std::cerr << "[PATROL] Checkpoint save failed: " << e.what() << '\n';
    }
}

bool PatrolTrainingAgent::SetNewState()
{
    // The generic Environment scheduler has already applied ECS physics and
    // world rules for both blue cars. Never advance a duplicate training world.
    bool const resetRequested = PatrolProdAgent::SetNewState();
    if (finished_ || !learner_ || !learner_->HasPendingAction())
        return resetRequested;

    try
    {
        auto const parameters = ReadParameters();
        if (!Valid(parameters) || !hasBeforeStep_)
            throw std::runtime_error("Invalid ECS feedback state");

        State after = ReadState(parameters);
        after.collided = GuardsCollided(parameters, beforeStep_, after);
        after.failed = after.collided;
        for (auto const& car : after.guards)
            after.failed |= !car.available
                || !std::isfinite(car.north) || !std::isfinite(car.east)
                || !std::isfinite(car.heading) || !std::isfinite(car.speed)
                || std::abs(car.north) >= parameters.halfNorth
                || std::abs(car.east) >= parameters.halfEast;

        float const reward = TeamReward(parameters, beforeStep_, lastAction_, after);
        bool const truncated = resetRequested && !after.failed;
        bool const updated = learner_->Observe(Observe(parameters, after), reward,
                                               after.failed, truncated);
        if (updated)
        {
            updates_ = learner_->StepCount();
            std::ostringstream serialized;
            learner_->WithActor([&](auto const& actor) {
                WritePatrolPolicy(serialized, actor);
                if (updates_ % 10U == 0U || updates_ >= maxUpdates_)
                    SavePatrolPolicy(checkpoint_, actor);
            });
            policy_->ReloadFromText(serialized.str());
            finished_ = updates_ >= maxUpdates_;
            if (finished_)
                std::cout << "[PATROL] ECS PPO completed after " << updates_
                          << " updates\n";
        }
    }
    catch (std::exception const& e)
    {
        std::cerr << "[PATROL] ECS PPO feedback failed: " << e.what() << '\n';
        finished_ = true;
    }
    return resetRequested;
}

ppo::model::sAgentStateSnaphot PatrolTrainingAgent::GetSnapshot() const noexcept
{
    auto snapshot = PatrolProdAgent::GetSnapshot();
    snapshot.m_name = "Patrol PPO (ECS training)";
    snapshot.m_common.m_step = static_cast<long long>(updates_);
    snapshot.m_common.m_maxStep = static_cast<long long>(maxUpdates_);
    snapshot.m_common.m_training = !finished_;
    snapshot.m_common.m_done = finished_;
    return snapshot;
}

} // namespace ppo::examples::twovstwo::defenders
