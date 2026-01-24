#pragma once 

#include "entity.hpp"

namespace ppo::model {

class SNA:
    public ISNAState
{
public:
             SNA                    (void) = default;                       
                                                              
    void     Run                    (void);                   
    void     Init                   (void);                   
                         
    void     SetEntityState         (sEntityState const* pState) noexcept;
    void     SetTargetPosition      (float north, float east)       noexcept;
    void     SetTargetSpeed         (float north, float east)       noexcept;
    void     SetTarget              (EntityId targetId)             noexcept;
    EntityId GetTarget              (void)                    const noexcept;
    float    GetKillProbability     (void)                    const noexcept;
    float    GetCLosingSpeed        (void)                    const noexcept;
    float    GetTargetLOS           (void)                    const noexcept;
    float    GetTargetLOSRate       (void)                    const noexcept;
    float    GetTargetNorthPosition (void)                    const noexcept;
    float    GetTargetEastPosition  (void)                    const noexcept;
    float    GetAccCmd              (void)                    const noexcept;     
    float    GetRotSpeedCmd         (void)                    const noexcept;

    template<typename Ar>
    void serialize(Ar& ar)
    {
        auto section = ar.begin_section("SNA");

        section("MaxNoEscapeZoneRange", m_state.m_maxNoescapeZone);
        section("MaxMissile",           m_state.m_maxnbmissiles);
        section("MaxShootAngle",        m_state.m_maxShootAngle);

        UpdateDoubleBuffer();
    };

    sSNAState   GetSnapshot         (void)                   const noexcept final;

protected:
    void        UpdateDoubleBuffer  (void);

    sEntityState const*     p_entityState{ nullptr };
    sSNAState               m_state{};
    sSNAState               m_buffers[2];
    std::atomic<uint32_t>   m_index{ 0 };
};

} /*  ppo::model */