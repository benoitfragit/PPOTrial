#include "snacontrolsystem.hpp"

#include "utils.hpp"

namespace ppo::model {

void 
SNAControlSystem::Create(BaseEntity& entity, ppo::serialization::json& ar)
{
    if (ar.contains("SNAControl"))
    {
        auto section = ar["SNAControl"];

        sSNAControlState& state{ entity.add<sSNAControlState>() };

        state.m_accCmd = 0.F;
        state.m_rotSpeedCmd = 0.F;

        bool available = section["Available"];
    }
}

void
SNAControlSystem::Init(BaseEntity& entity)
{
    if (!entity.has<sSNAControlState>())
        return;

    auto [snaControlState] = entity.get<sSNAControlState>();

    snaControlState.m_accCmd = 0.F;
    snaControlState.m_rotSpeedCmd = 0.F;
}

void
SNAControlSystem::Run(BaseEntity& entity)
{
    if (!entity.has<sFlightModelState, sSNAControlState>())
        return;

    auto& [flightModelState, snaControlState] = entity.get<sFlightModelState, sSNAControlState>();
    flightModelState.m_accpsi = flightModelState.m_maxAccRot * snaControlState.m_rotSpeedCmd;
    flightModelState.m_acc    = flightModelState.m_maxAcc    * snaControlState.m_accCmd;
}

}