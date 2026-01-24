#pragma once

#include "interface.hpp"

namespace ppo::model {

class FlightModel : 
    public IFlightModelState
{
public:
    explicit FlightModel(sEntityState& entityState):
        m_entityState{ entityState } {};

    void     Run                 (float dt);
    void     Init                (void);

    float    GetFuelConsumption  (void)                               const noexcept;
    float    GetMaxSpeed         (void)                               const noexcept;
    float    GetMaxRotSpeed      (void)                               const noexcept;
    float    GetMaxLongitudinalAcc(void)                              const noexcept;
    void     SetRotSpeed         (float value)                              noexcept;
    void     SetLongitudinalAcc  (float value)                              noexcept;
    
    template<typename Ar>
    void serialize(Ar& ar)
    {
        auto section = ar.begin_section("FlightModel");

        section("MaxFuel",              m_state.m_maxfuel);
        section("MaxSpeed",             m_state.m_maxSpeed);
        section("MinSpeed",             m_state.m_minSpeed);
        section("MaxRotAcceleration",   m_state.m_maxAccRot);
        section("MaxAcceleration",      m_state.m_maxAcc);

        UpdateDoubleBuffer();
    };

    sFlightModelState   GetSnapshot(void) const noexcept final;

    void                FillCheckpoint(CheckPointBuffer& buffer)                               noexcept;
    void                SetCheckpoint(CheckPointBuffer const& buffer)     noexcept;

private:
    void                UpdateDoubleBuffer(void);

    sEntityState&           m_entityState;
    sFlightModelState       m_state{};
    sFlightModelState       m_buffers[2];
    std::atomic<uint32_t>   m_index{ 0 };
};

}