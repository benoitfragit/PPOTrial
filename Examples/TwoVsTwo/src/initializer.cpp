#include "initializer.hpp"
#include "spawn.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"
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
        config.side = std::bernoulli_distribution{0.5}(config.rng) ? 1 : -1;

    auto const& [metadata] = entity.get<ppo::model::sEntityMetadata>();
    auto const& world = registry->ctx().get<ppo::model::sEnvironmentState>();
    auto const bounds = WorldBounds{0.5F * world.m_maxNorth, 0.5F * world.m_maxEast};
    auto const pose = MakeSpawn(metadata.m_name, config.side,
                                config.red2DestroyedProbability, bounds, config.rng);
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
