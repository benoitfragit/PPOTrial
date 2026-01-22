#pragma once
#include "interface.hpp"

#include <memory>
#include <array>
#include <string>

namespace ppo::model {


class Player : 
    public IPlayer,
    public IPlayerState ,
    public ICheckPointBuffer
{
public:
    class IEnvironment
    {
    public:
        virtual           ~IEnvironment(void) = default;
        virtual float     GetTimeStep(void) const noexcept = 0;
        virtual Player*   GetPlayerById(PlayerId playerId) noexcept = 0;
        virtual float     GetMaxNorth(void) const noexcept = 0;
        virtual float     GetMaxEast(void) const noexcept = 0;
    };

    template<class Ty>
    class EnvironmentWrapper :
        public IEnvironment
    {
    public:
        EnvironmentWrapper(Ty& env) : m_env{ env } {};

        float   GetTimeStep(void) const noexcept { return m_env.GetTimeStep(); };
        Player* GetPlayerById(PlayerId playerId) noexcept 
        { 
            return dynamic_cast<Player*>(m_env.GetIPlayerById(playerId)); 
        };
        float   GetMaxNorth(void) const noexcept { return m_env.GetMaxNorth(); };
        float   GetMaxEast(void) const noexcept { return m_env.GetMaxEast(); };

    private:
        Ty& m_env;
    };


                      template<typename Ty>
                      Player             (std::string const& datapath, 
                                          std::string const& name, 
                                          ePlayerSide side,
                                          ePlayerType type,
                                          Ty& env)  :
                          Player(datapath, name, side, type, env.GetNewPlayerId(), std::make_shared<EnvironmentWrapper<Ty>>(env))
                      {
                      
                      };

                      
    virtual                   ~Player             (void);
                      
                      Player              (Player&)  = delete;
                      Player              (Player&&) = delete;
    Player&           operator=           (Player&)  = delete;
    Player&           operator=           (Player&&) = delete;
                      
    void              Init                (void)                                     noexcept;
    void              Run                 (void)                                     noexcept;
    void              UpdateEnv           (void)                               const noexcept override;
                      
    std::string const& GetName            (void)                               const noexcept;
    bool              GetDestroyed        (void)                               const noexcept;
    float             GetRotation         (void)                               const noexcept;
    float             GetNorthPosition    (void)                               const noexcept;
    float             GetEastPosition     (void)                               const noexcept;
    float             GetFuelConsumption  (void)                               const noexcept;
    float             GetMaxSpeed         (void)                               const noexcept;
    float             GetNorthSpeed       (void)                               const noexcept;
    float             GetEastSpeed        (void)                               const noexcept;
    float             GetMaxRotSpeed      (void)                               const noexcept;
    float             GetMaxLongitudinalAcc(void)                              const noexcept;
    void              SetRotSpeed         (float value)                              noexcept;
    void              SetLongitudinalAcc  (float value)                              noexcept;
    void              SetDestroyed          (bool destroyed)                         noexcept;
    
    PlayerId          GetPlayerId         (void)                               const noexcept;

    sPlayerState      GetSnapshot         (void)                               const noexcept final;

    CheckPointBuffer  GetCheckpoint       (void)                               noexcept final;
    void              SetCheckpoint       (CheckPointBuffer const& buffer)     noexcept final;

protected:
    Player(std::string const& datapath, 
           std::string const& name, 
           ePlayerSide side, 
           ePlayerType type, 
           PlayerId playerID, 
           std::shared_ptr<IEnvironment> pEnv);

    virtual void OnInit             (void);
    virtual void OnRun              (void);
    virtual void UpdateDoubleBuffer (void);

    template<typename Ar>
    void serialize(Ar& ar)
    {
        auto section = ar.begin_section(m_name.c_str());

        section("MaxNoEscapeZoneRange", m_maxNoescapeZone);
        section("MaxFuel", m_maxfuel);
        section("MaxMissile", m_maxnbmissiles);
        section("MaxSpeed", m_maxSpeed);
        section("MinSpeed", m_minSpeed);
        section("MaxShootAngle", m_maxShootAngle);
        section("MaxRotAcceleration", m_maxAccRot);
        section("MaxAcceleration", m_maxAcc);
    };

    float           m_maxNoescapeZone{8000.F };
    PlayerId        m_playerId{ ULLONG_MAX };
    float           m_acc{ 0.F };
    float           m_accpsi{ 0.F };
    float           m_dx{ 0.F };
    float           m_dy{ 0.F };
    float           m_x{ 0.F  };
    float           m_y{ 0.F  };
    float           m_maxAcc{ 0.F };
    float           m_maxAccRot{ 0.F };
    float           m_maxSpeed{ 0.F };
    float           m_minSpeed{ 0.F };
    float           m_speed{ 0.F };
    float           m_rotation{ 0.F };
    float           m_fuel{ 0.F };
    float           m_fuelconsumption{ 0.F };
    float           m_escapezone{ 0.F };
    float           m_noescapezone{ 0.F };
    float           m_maxfuel{ 0.F };
    float           m_maxShootAngle{ 0.F };
    unsigned short  m_nbmissiles{ 0 };
    unsigned short  m_maxnbmissiles{ 0 };
    ePlayerSide     m_side{ ePlayerSide::Unknown };
    ePlayerType     m_type{ ePlayerType::Objective };
    bool            m_destroyed{ false };

    std::shared_ptr<IEnvironment>   p_env{ nullptr };
    sPlayerState                    m_buffers[2];
    std::atomic<uint32_t>           m_index{ 0 };
    std::string                     m_name{ "unknown" };
};


}   /* ppo::model */