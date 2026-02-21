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

        float maxNorth{ envState.m_maxNorth / 2.F };
        float maxEast { envState.m_maxEast / 2.F };
        float minNorth{ -maxNorth };
        float minEast { -maxEast };
        bool  onTheRIght{ false };

        // Choose if the entity is on the left or on the right
        RandomGenerator leftRightGenerator{};
        leftRightGenerator.registerAttr(onTheRIght, 0.5F);
        leftRightGenerator.generate();

        // left area to initialize an entity
        float minLeftEast{ minEast };
        float maxLeftEast{ minEast + (maxEast - minEast) / 4.F };

        float minRightEast{ (maxEast - minEast) / 4.F };
        float maxRightEast{ maxEast };
        
        if (state.m_type == eEntityType::SAM)
        {                          
            if (onTheRIght)
            {
                minEast = 0;
                maxEast = minRightEast;
            }
            else
            {
                minEast = maxLeftEast;
                maxEast = 0;
            }
        }
        else
        {
            if (onTheRIght)
            {
                minEast = minRightEast;
                maxEast = maxRightEast;
            }
            else
            {
                minEast = minLeftEast;
                maxEast = maxLeftEast;
            }
        }

        generator.registerAttr(state.m_x, minNorth, maxNorth);
        generator.registerAttr(state.m_y, minEast, maxEast);
    }

    if (state.m_side == eEntitySide::Hostile
    &&  state.m_type == eEntityType::SAM)
        generator.registerAttr(state.m_destroyed, 0.1F);

    // Random selection of a target
    if (state.m_type == eEntityType::Aircraft
    &&  entity.has<sSNAState>())
    {
        auto [snaState] = entity.get<sSNAState>();

        bool selectThisTarget{ false };

        RandomGenerator targetgenerator{};

        targetgenerator.registerAttr(selectThisTarget, 0.5F);

        auto v = registry->view<sEntityState>();
        for (auto target : v)
        {
            auto [targetState] = v.get(target);

            if ((targetState.m_entityId != state.m_entityId)
             && (targetState.m_side != state.m_side)
             && (   (targetState.m_type == eEntityType::Aircraft) 
                 || (targetState.m_type == eEntityType::Objective))
             && !targetState.m_destroyed)
            {
                targetgenerator.generate();

                if (selectThisTarget)
                {
                    snaState.m_target = targetState.m_entityId;
                    break;
                }
            }
        }

        if (state.m_type == eEntityType::SAM
            && entity.has<sSNAState>())
        {
            // breaking symetries
            generator.registerAttr(snaState.m_noescapezone, 1.1F * snaState.m_noescapezone, 1.2F * snaState.m_noescapezone);
        }
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