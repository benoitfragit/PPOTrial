#include "aircraft.hpp"

namespace ppo::model {

void
Aircraft::SetSNA(std::shared_ptr<SNA> sna) noexcept
{
    p_sna = sna;
    if (p_sna != nullptr)
        p_sna->SetAircraft(this);
};

void
Aircraft::Init(void) noexcept
{
    // Init the flight model
    m_flightModel.Init();
    
    // Init the entity
    Entity::Init();

    // Init the SNA
    if (p_sna != nullptr)
        p_sna->Init();
}


void
Aircraft::Run(void) noexcept
{
    if (p_env == nullptr
    ||  p_sna == nullptr)
        return;

    float const dt{ p_env->GetTimeStep() };

    m_flightModel.SetRotSpeed(m_flightModel.GetMaxRotSpeed() * p_sna->GetRotSpeedCmd());
    m_flightModel.SetLongitudinalAcc(m_flightModel.GetMaxLongitudinalAcc() * p_sna->GetAccCmd());

    // Run the flight model
    m_flightModel.Run(dt);

    // Move the entity
    Entity::Run();

    // Run the SNA
    Entity* target{ p_env->GetEntityById(p_sna->GetTarget()) };

    if (target != nullptr)
    {
        p_sna->SetTargetPosition(target->GetNorthPosition(), target->GetEastPosition());
        p_sna->SetTargetSpeed(target->GetNorthSpeed(), target->GetEastSpeed());
    }
    else
    {
        p_sna->SetTargetPosition(0.F, 0.F);
        p_sna->SetTargetSpeed(0.F, 0.F);
    }

    p_sna->Run();
}

void  
Aircraft::SetCheckpoint(CheckPointBuffer const& buffer)     noexcept
{
    m_flightModel.SetCheckpoint(buffer);
    Entity::SetCheckpoint(buffer);

    if (p_sna != nullptr)
        p_sna->Init();
}

CheckPointBuffer  
Aircraft::GetCheckpoint(void) noexcept
{
    CheckPointBuffer buffer = Entity::GetCheckpoint();
    m_flightModel.FillCheckpoint(buffer);
    return buffer;
}

void
Aircraft::UpdateEnv(void) const noexcept
{
    if (p_env == nullptr
     || p_sna == nullptr)
        return;
    
    // very simple effect for now
    float const pkTarget{ p_sna->GetKillProbability() };
    if (ppo::model::Entity* const target{ dynamic_cast<ppo::model::Entity*>(p_env->GetEntityById(p_sna->GetTarget())) };
        target != nullptr
        && 0.95F < pkTarget)
        target->SetDestroyed(true);
}

void 
Aircraft::SetTarget(EntityId targetId) noexcept
{
    if (p_sna == nullptr)
        return;

    p_sna->SetTarget(targetId);
}

SNA const& 
Aircraft::GetSNA(void)                             const noexcept 
{
    return *p_sna.get();
}

FlightModel const& 
Aircraft::GetFlightModel(void)                     const noexcept
{
    return m_flightModel;
}

}