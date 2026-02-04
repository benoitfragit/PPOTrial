#include "targetselector.hpp"
#include "utils.hpp"

namespace ppo::mission {

void
RLTargetSelector::SetNextAction(ACTIONS const& action) noexcept
{
    for (auto j = 0; j < action.size(); ++j )
    {
        if (action[j] < 1.F)
            continue;
        
        // we found our next goal
        auto entity = p_environment->GetEntityAtIndex(j);
            
        // immediate target becomes the new one
        if (entity != nullptr && p_sna != nullptr)
            p_sna->SetTarget(entity->GetEntityId());

        break;
    }
}

void 
RLTargetSelector::SetNewState(STATES& state) noexcept
{
    memset(state.data(), 0, sizeof(float) * state.size());

    unsigned int j{ 0 };
    for (auto i = 0; i < p_environment->GetNumberOfEntities(); ++i)
    {
        auto entity = p_environment->GetEntityAtIndex(i);

        if (entity == p_aircraft
         || entity == nullptr)
            continue;

        if (MAX_TRACKS <= j)
            break;
    
        auto const& aircraftState{ p_aircraft->GetEntityState() };
        auto const& entityState  { entity->GetEntityState() };

        float const dx{ entityState.m_x - aircraftState.m_x };
        float const dy{ entityState.m_y - aircraftState.m_y };
        float const d{ sqrt(dx * dx + dy * dy) };

        float const ux{ dx / (d + 1e-6F) };
        float const uy{ dy / (d + 1e-6F) };

        float const vrx{ entityState.m_dx - aircraftState.m_dx };
        float const vry{ entityState.m_dy - aircraftState.m_dy };

        float const closingSpeed{ -(vrx * ux + vry * uy) };

        float az = atan2(uy, ux);
        az = ppo::model::wrapMinusPi2Pi(az);
        az = ppo::model::wrapMinusPi2Pi(az - aircraftState.m_rotation);

        float side{ 1.F };
        if (aircraftState.m_side == entityState.m_side)
            side= 0.F;
        else if (entityState.m_side == ppo::model::eEntitySide::Unknown)
            side=-1.F;

        float importancy{ 0.F };
        if (p_sna->GetTarget() == entity->GetEntityId())
            importancy = 1.F;
        else
            importancy = 0.5F * side;

        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::NorthRange)]    = dx / std::max(d, 1e-6F);
        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::EastRange)]     = dy / std::max(d, 1e-6F);
        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::Range)]         = 1.0F - std::exp(-d / 24000.F);
        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::ClosingSpeed)]  = std::clamp(closingSpeed / 300.F, -1.F, 1.F);
        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::CosBearing)]    = cos(az);
        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::SinBearing)]    = sin(az);
        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::Side)]          = side;
        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::IsObjective)]   = entityState.m_type==ppo::model::eEntityType::Objective;
        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::IsAircraft)]    = entityState.m_type==ppo::model::eEntityType::Missile;
        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::IsMissile)]     = entityState.m_type==ppo::model::eEntityType::Aircraft;
        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::Importancy)]    = importancy;
        state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::Available)]     = static_cast<float>(entity != nullptr);

        ++j;
    }
}

}