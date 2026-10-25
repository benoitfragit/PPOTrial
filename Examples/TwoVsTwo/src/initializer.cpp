#include "initializer.hpp"
#include "spawn.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"
#include "patrol_core.hpp"
#include <random>

namespace ppo::examples::twovstwo {
void InitializeEntity(ppo::model::Entity& entity)
{
    auto* registry = entity.registry();
    if (!registry || !registry->ctx().contains<InitializerContext>()
        || !entity.has<ppo::model::sEntityState, ppo::model::sEntityMetadata>())
        return;

    auto& config = registry->ctx().get<InitializerContext>();
    if (config.count == 0)
        return;
    if (config.next == 0)
    {
        config.side = std::bernoulli_distribution{0.5}(config.rng) ? 1 : -1;
        if (config.patrolOnly)
        {
            auto const& world = registry->ctx().get<ppo::model::sEnvironmentState>();
            defenders::Parameters p{};
            p.halfNorth = 0.5F * world.m_maxNorth;
            p.halfEast = 0.5F * world.m_maxEast;
            config.arclength = std::uniform_real_distribution<float>{
                0.F, defenders::CircuitLength(p)}(config.rng);
        }
    }

    auto const& [metadata] = entity.get<ppo::model::sEntityMetadata>();
    auto const& world = registry->ctx().get<ppo::model::sEnvironmentState>();
    auto const bounds = WorldBounds{0.5F * world.m_maxNorth, 0.5F * world.m_maxEast};
    auto pose = MakeSpawn(metadata.m_name, config.side,
                                config.red2DestroyedProbability, bounds, config.rng);
    if (pose && config.patrolOnly
        && (metadata.m_name == "blue_guard_1" || metadata.m_name == "blue_guard_2"))
    {
        defenders::Parameters p{};
        p.halfNorth = bounds.halfNorth;
        p.halfEast = bounds.halfEast;
        auto const role = metadata.m_name == "blue_guard_1" ? 0U : 1U;
        auto const reference = defenders::PatrolReference(p, config.side,
                                                         role, config.arclength);
        auto const mobile = defenders::OnReference(p, reference);
        pose->north = mobile.north;
        pose->east = mobile.east;
        pose->heading = mobile.heading;
        pose->speed = mobile.speed;
    }
    if (pose)
    {
        auto [state] = entity.get<ppo::model::sEntityState>();
        state.m_x = pose->north;
        state.m_y = pose->east;
        state.m_rotation = pose->heading;
        state.m_speed = pose->speed;
        state.m_destroyed = pose->destroyed;
        state.m_prevDestroyed = pose->destroyed;
        if (entity.has<ppo::model::sFlightModelState>())
        {
            auto [flight] = entity.get<ppo::model::sFlightModelState>();
            flight.m_acc = 0.F;
            flight.m_accpsi = 0.F;
        }
        if (entity.has<ppo::model::sSNAControlState>())
        {
            auto [control] = entity.get<ppo::model::sSNAControlState>();
            control.m_accCmd = 0.F;
            control.m_rotSpeedCmd = 0.F;
        }
    }
    config.next = (config.next + 1U) % config.count;
}
} // namespace ppo::examples::twovstwo
