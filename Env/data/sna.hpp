#pragma once 

#include "player.hpp"

namespace ppo::model {

class SNA : public ppo::model::Player
{
public:
    template <typename Ty>
    SNA                         (std::string const& datapath, std::string const& name, 
                                    ePlayerSide side,
                                    ePlayerType type,
                                    Ty& env) :
        Player(datapath, name, side,type, env)
    {};

    void OnRun                      (void) override;
    void OnInit                     (void) override;
    void UpdateEnv                  (void) const noexcept override;

    void     SetTargetPosition      (float north, float east)                noexcept;
    void     SetTargetSpeed         (float north, float east)                noexcept;
    void     SetTarget              (PlayerId targetId)                      noexcept;
    PlayerId GetTarget              (void)                             const noexcept;
    float    GetKillProbability     (void)                             const noexcept;
    float    GetCLosingSpeed        (void)                             const noexcept;
    float    GetTargetLOS           (void)                             const noexcept;
    float    GetTargetLOSRate       (void)                             const noexcept;

protected:
    PlayerId             m_target{ ULLONG_MAX };
    float                m_pk{ 0.F };
    float                m_target_x{ 0.F };
    float                m_target_y{ 0.F };
    float                m_target_dx{ 0.F };
    float                m_target_dy{ 0.F };
    float                m_closingspeed{ 0.F };
    float                m_target_los{ 0.F };
    float                m_target_los_rate{ 0.F };
};

} /*  ppo::model */