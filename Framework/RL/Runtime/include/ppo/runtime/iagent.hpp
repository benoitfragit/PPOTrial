/**
* @file Runtime/include/ppo/runtime/iagent.hpp
* @brief The agent contract of the ECS world: what the environment calls each step.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

namespace ppo::model {

/**
* @brief An agent of the ECS world, called by Environment::BeginStep and EndStep.
*/
class IAgent
{
public:
    virtual ~IAgent(void) = default;

    /**
    * @brief Starts an episode, after every entity was re-initialized.
    */
    virtual void Init(void) = 0;
    /**
    * @brief Chooses and applies the action of the step, before the entities advance.
    */
    virtual void ChooseAction(void) = 0;
    /**
    * @brief Reads the state after the step.
    * @return True when the agent ends the episode.
    */
    virtual bool SetNewState(void) = 0;
};

} /* ppo::model */
