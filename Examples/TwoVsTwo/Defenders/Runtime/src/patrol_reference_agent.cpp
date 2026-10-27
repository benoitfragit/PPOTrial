#include "patrol_reference_agent.hpp"
#include "patrol_reference_controller.hpp"
#include "initializer.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include <cmath>
#include <stdexcept>

namespace ppo::examples::twovstwo::defenders {

PatrolReferenceAgent::PatrolReferenceAgent(ppo::model::BaseEntity& own,
                                           std::size_t role)
    : own_(own.registry(), own.uniqid()), role_(role)
{
    if (own.registry() == nullptr || role_ >= 2U)
        throw std::invalid_argument("Invalid patrol reference agent");
}

void PatrolReferenceAgent::Init()
{
    step_ = 0U;
    done_ = false;
}

Parameters PatrolReferenceAgent::ReadParameters()
{
    Parameters p{};
    auto* registry = own_.registry();
    if (registry && registry->ctx().contains<ppo::model::sEnvironmentState>())
    {
        auto const& world = registry->ctx().get<ppo::model::sEnvironmentState>();
        p.halfNorth = 0.5F * world.m_maxNorth;
        p.halfEast = 0.5F * world.m_maxEast;
        p.dt = world.m_dt;
    }
    if (registry && registry->ctx().contains<InitializerContext>())
    {
        auto const& context = registry->ctx().get<InitializerContext>();
        p.cruiseSpeed = context.patrolCruiseSpeed;
        p.collisionDistance = context.patrolCollisionDistance;
        if (context.patrolGeometryReady)
            p.geometry = context.patrolGeometry;
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

void PatrolReferenceAgent::ChooseAction()
{
    if (done_ || !own_.has<ppo::model::sEntityState,
                             ppo::model::sSNAControlState>())
        return;

    auto* registry = own_.registry();
    if (!registry || !registry->ctx().contains<InitializerContext>())
        return;

    auto const p = ReadParameters();
    if (!Valid(p))
        return;

    auto const& ctx = registry->ctx().get<InitializerContext>();
    auto const [state] = own_.get<ppo::model::sEntityState>();
    Mobile const mobile{state.m_x, state.m_y, state.m_rotation,
                        state.m_speed, 0.F, !state.m_destroyed};
    auto const action = PatrolReferenceAction(
        p, mobile, ctx.side, role_, 0.F);

    auto [command] = own_.get<ppo::model::sSNAControlState>();
    command.m_rotSpeedCmd = action[0];
    command.m_accCmd = action[1];
}

bool PatrolReferenceAgent::SetNewState()
{
    if (done_)
        return true;
    ++step_;
    auto const p = ReadParameters();
    if (!Valid(p) || !own_.has<ppo::model::sEntityState>())
        return done_ = true;

    auto const [car] = own_.get<ppo::model::sEntityState>();
    done_ = step_ >= p.episodeSteps
        || car.m_destroyed
        || !std::isfinite(car.m_x) || !std::isfinite(car.m_y)
        || !std::isfinite(car.m_rotation) || !std::isfinite(car.m_speed)
        || std::abs(car.m_x) >= p.halfNorth
        || std::abs(car.m_y) >= p.halfEast;
    return done_;
}

ppo::model::sAgentStateSnaphot PatrolReferenceAgent::GetSnapshot() const noexcept
{
    ppo::model::sAgentStateSnaphot snapshot{};
    snapshot.m_name = role_ == 0U ? "Patrol 1 (reference)"
                                  : "Patrol 2 (reference)";
    snapshot.m_id = id_;
    snapshot.m_common.m_step = static_cast<long long>(step_);
    snapshot.m_common.m_maxStep = 4096;
    snapshot.m_common.m_done = done_;
    snapshot.m_common.m_training = false;
    return snapshot;
}

} // namespace ppo::examples::twovstwo::defenders
