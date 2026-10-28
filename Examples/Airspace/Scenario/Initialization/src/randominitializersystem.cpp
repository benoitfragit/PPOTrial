/**
* @file Initializer/src/randominitializersystem.cpp
* @brief Implementation for randominitializersystem.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "randominitializersystem.hpp"
#include "airspacecomponents.hpp"

#include "interface.hpp"
#include "utils.hpp"

namespace ppo::model {

/**
* @brief Randomizes one entity state inside the current environment bounds.
*
* The initializer places entities in broad left or right sectors, applies
* platform-specific speed and acceleration ranges, and optionally randomizes
* targets and SAM-related parameters.
*
* @param entity Entity to initialize.
*/
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

        // Choose whether the entity spawns in the left or right sector.
        RandomGenerator leftRightGenerator{};
        leftRightGenerator.registerAttr(onTheRIght, 0.5F);
        leftRightGenerator.generate();

        // Define the broad left-hand spawn sector.
        float minLeftEast{ minEast };
        float maxLeftEast{ minEast + (maxEast - minEast) / 4.F };

        float minRightEast{ (maxEast - minEast) / 4.F };
        float maxRightEast{ maxEast };
        
        if (state.m_type == eEntityType::RestrictedZone)
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
    &&  state.m_type == eEntityType::RestrictedZone)
        generator.registerAttr(state.m_destroyed, 0.1F);

    // Random selection of a target
    if (state.m_type == eEntityType::Mobile
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
             && (   (targetState.m_type == eEntityType::Mobile) 
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

        if (state.m_type == eEntityType::RestrictedZone
            && entity.has<sSNAState>())
        {
            // Slightly vary the SAM no-escape zone to break symmetry.
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
