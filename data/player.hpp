#pragma once
#include "interface.hpp"

#include <memory>
#include <array>
#include <string>

namespace ppo::model {


class Player : 
    public IPlayerState
{
public:
                      Player              (std::string const& name, PlayerId playerID);
                      
                      ~Player             (void) final;
                      
                      Player              (Player&)  = delete;
                      Player              (Player&&) = delete;
    Player&           operator=           (Player&)  = delete;
    Player&           operator=           (Player&&) = delete;
                      
    void              Init                (sEnvironmentState const& env)             noexcept;
    void              Run                 (float dt)                                 noexcept;
                      
    void              SetDestroyed        (bool destroyed)                           noexcept;
    void              SetRemainingFuel    (float fuel)                               noexcept;
    void              SetNbMissiles       (unsigned short nb)                        noexcept;
    void              SetMaxSpeed         (float maxSpeed)                           noexcept;
    void              SetNorthSpeed       (float speed)                              noexcept;
    void              SetEastSpeed        (float speed)                              noexcept;
    void              SetNoEscapeZoneRange(float minNEZ, float maxNEZ)               noexcept;
    void              SetEscapeZone       (float ezcapezone)                         noexcept;
    void              SetNoEscapeZone     (float noezcapezone)                       noexcept;
    void              SetMaxFuel          (float maxFuel)                            noexcept;
    void              SetMaxMissiles      (unsigned short maxMissiles)               noexcept;
    void              SetFightingArea     (float topleftnorth, 
                                           float toplefteast, 
                                           float bottomrightnorth, 
                                           float bottomrighteast)                    noexcept;
    void              SetTargetPosition   (float north, float east)                  noexcept;
    void              SetTarget           (PlayerId targetId)                        noexcept;

    std::string const& GetName            (void)                             const noexcept;
    float             GetTargetPositionNorth(void)                             const noexcept;
    float             GetTargetPositionEast (void)                             const noexcept;
    PlayerId          GetTarget           (void)                               const noexcept;
    bool              GetDestroyed        (void)                               const noexcept;
    float             GetRotation         (void)                               const noexcept;
    float             GetNorthPosition    (void)                               const noexcept;
    float             GetEastPosition     (void)                               const noexcept;
    float             GetRemainingFuel    (void)                               const noexcept;
    float             GetFuelConsumption  (void)                               const noexcept;
    unsigned short    GetRemainingMissiles(void)                               const noexcept;
    float             GetEscapeZone       (void)                               const noexcept;
    float             GetNoEscapeZone     (void)                               const noexcept;
    float             GetMaxFuel          (void)                               const noexcept; 
    float             GetMaxSpeed         (void)                               const noexcept;
    ppoArray2f const& GetNoEscapeZoneRange(void)                               const noexcept;
    unsigned short    GetMaxMissiles      (void)                               const noexcept;
    float             GetSpeed            (void)                               const noexcept;
    float             GetNorthSpeed       (void)                               const noexcept;
    float             GetEastSpeed        (void)                               const noexcept;

    PlayerId          GetPlayerId         (void)                               const noexcept;

    sPlayerState      GetSnapshot         (void)                               const noexcept final;

    void              ActionSetNorthSpeed       (float value)                              noexcept;
    void              ActionSetEastSpeed        (float value)                              noexcept;  

    float             StateNorthRange           (sEnvironmentState const& env) const noexcept;
    float             StateEastRange            (sEnvironmentState const& env) const noexcept;  
    float             StateRange                (sEnvironmentState const& env) const noexcept;

    float             StateRemainingFuel        (sEnvironmentState const& env) const noexcept;  
    float             StateRemainingMissiles    (sEnvironmentState const& env) const noexcept;

    /// <summary>
    /// This is a state to be registered in an array
    /// </summary>
    using State = float(Player::*)(sEnvironmentState const&) const noexcept;

    constexpr std::array<State, 3> const static STATES
    {
        &Player::StateNorthRange,
        &Player::StateEastRange,
        &Player::StateRange
    };

    /// <summary>
    /// this is an action to be registered in an array
    /// </summary>
    using Action = void(Player::*)(float);

    /// <summary>
    /// This is a list of action to be registered in an agent
    /// </summary>
    constexpr  std::array<Action, 2> const static ACTIONS
    {
        &Player::ActionSetNorthSpeed,
        &Player::ActionSetEastSpeed
    };

    void LoadFromFile(std::string const& filepath);
    void SaveToFile(std::string const& filepath);

private:
    struct Impl;
    std::unique_ptr<Impl> p_impl;
};


}   /* ppo::model */