#include "randominitializersystem.hpp"

#include "interface.hpp"
#include "utils.hpp"

namespace ppo::model {

void
RandomInitiliazerSystem::Init(Entity& entity)
{
    entt::registry* registry{ entity.registry() };
    if (registry == nullptr
    || !entity.has<sEntityState>())
        return;
        
    RandomGenerator generator{};

    auto [state] = entity.get<sEntityState>();

    if (registry->ctx().contains<sEnvironmentState>())
    {
        auto const& envState{ registry->ctx().get<sEnvironmentState>() };
        generator.registerAttr(state.m_x, -envState.m_maxNorth / 2.F, envState.m_maxNorth / 2.F);
        generator.registerAttr(state.m_y, -envState.m_maxEast / 2.F, envState.m_maxEast / 2.F);
    }

    generator.registerAttr(state.m_rotation, 0.F, static_cast<float>(2.0 * PI));

    if (entity.has<sFlightModelState>())
    {
        auto [flightmodelstate] = entity.get<sFlightModelState>();

        if (flightmodelstate.m_minSpeed < flightmodelstate.m_maxSpeed)
            generator.registerAttr(state.m_speed, flightmodelstate.m_minSpeed + 50.F, flightmodelstate.m_maxSpeed);

        generator.registerAttr(flightmodelstate.m_accpsi, -flightmodelstate.m_maxAccRot / 2.F, flightmodelstate.m_maxAccRot / 2.F);
        generator.registerAttr(flightmodelstate.m_acc, -flightmodelstate.m_maxAcc / 2.F, flightmodelstate.m_maxAcc / 2.F);
    }

    generator.generate();
}

}