#pragma once

#include "Player.hpp"
#include <string>

namespace ppo::model {

class Objective :
    public Player
{
public:
    template <typename Ty>
    Objective(std::string name, ePlayerSide side, Ty& env):
        Player("Objectives", name, side, ePlayerType::Objective, env)
    {

    };

private:
    void OnRun(void) final;
};

} /* ppo::model */