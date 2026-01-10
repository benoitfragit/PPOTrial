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
                                          Ty& env)  :
                          Player(datapath, name, side, env.GetNewPlayerId(), std::make_shared<EnvironmentWrapper<Ty>>(env))
                      {
                      
                      };

                      
    virtual                   ~Player             (void);
                      
                      Player              (Player&)  = delete;
                      Player              (Player&&) = delete;
    Player&           operator=           (Player&)  = delete;
    Player&           operator=           (Player&&) = delete;
                      
    void              Init                (void)                                     noexcept;
    void              Run                 (void)                                     noexcept;
                      

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
    void              SetTargetPosition     (float north, float east)                noexcept;
    void              SetTargetSpeed        (float north, float east)                noexcept;
    void              SetTarget             (PlayerId targetId)                      noexcept;
    PlayerId          GetTarget             (void)                             const noexcept;
    float             GetKillProbability    (void)                             const noexcept;
    float             GetCLosingSpeed       (void)                             const noexcept;
    float             GetTargetLOS          (void)                             const noexcept;
    float             GetTargetLOSRate      (void)                             const noexcept;
    float             GetTargetNorthPosition(void)                             const noexcept;
    float             GetTargetEastPosition (void)                             const noexcept;
    float             GetTargetNorthSpeed   (void)                             const noexcept;
    float             GetTargetEastSpeed    (void)                             const noexcept;

    PlayerId          GetPlayerId         (void)                               const noexcept;

    sPlayerState      GetSnapshot         (void)                               const noexcept final;

    CheckPointBuffer  GetCheckpoint       (void)                               noexcept final;
    void              SetCheckpoint       (CheckPointBuffer const& buffer)     noexcept final;

protected:
    Player(std::string const& datapath, std::string const& name, ePlayerSide side, PlayerId playerID, std::shared_ptr<IEnvironment> pEnv);

    struct Impl;
    std::unique_ptr<Impl> p_impl;
};


}   /* ppo::model */