#pragma once 
#include "interface.hpp"
#include "player.hpp"

#include<string>
#include <memory>
#include <array>
#include <functional>

namespace ppo::model {

class Environment {
public:

                      Environment     (void);

    virtual           ~Environment    (void);

                      Environment           (Environment&)  = delete;
                      Environment           (Environment&&) = delete;
    Environment&      operator=             (Environment&)  = delete;
    Environment&      operator=             (Environment&&) = delete;

    using UpdateEnvFunction = std::function<void(Environment&, IPlayer*)>;

    void              AddAgent(std::shared_ptr<IPlayer> player, UpdateEnvFunction updater, std::shared_ptr<IAgent> pAgent = nullptr);

    static PlayerId   GetNewPlayerId(void);

    sEnvironmentState GetSnapshot           (void) const noexcept;

    void              Start                 (void);
    void              Play                  (void);
    void              Pause                 (void);
    void              Stop                  (void);

    float             GetTimeStep           (void) const noexcept;

    IPlayer*          GetIPlayerById        (PlayerId playerId) noexcept;

    float             GetMaxNorth           (void) const noexcept;
    float             GetMaxEast            (void) const noexcept;

    void              SetMaxRange           (float north, float east);
    void              SetTimeStep           (float timestep);
    

private:

    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

}   /* ppo::model */            