#include "patrol_agent.hpp"
#include "patrol_policy.hpp"
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
    auto const observation = Observe(p, ReadState(p));
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
    done_ = step_ >= p.episodeSteps;
    for (auto const& car : after.guards)
        done_ |= !car.available || !std::isfinite(car.north) || !std::isfinite(car.east)
            || std::abs(car.north) >= p.halfNorth || std::abs(car.east) >= p.halfEast;
    return done_;
}

ppo::model::sAgentStateSnaphot PatrolProdAgent::GetSnapshot() const noexcept
{
    ppo::model::sAgentStateSnaphot snapshot{};
    snapshot.m_name = role_ == 0U ? "Patrol 1 PPO (RLtools)" : "Patrol 2 PPO (RLtools)";
    snapshot.m_id = id_;
    snapshot.m_common.m_step = static_cast<long long>(step_);
    snapshot.m_common.m_maxStep = 4096;
    snapshot.m_common.m_done = done_;
    snapshot.m_common.m_training = false;
    return snapshot;
}
} // namespace ppo::examples::twovstwo::defenders
