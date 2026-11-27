/**
* @file Env/data/environment.hpp
* @brief The ECS world of the 2D examples, stepped by its caller (an environment DLL).
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once
#include "components.hpp"
#include "entity.hpp"
#include "ppo/runtime/iagent.hpp"
#include "worldrules.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>

namespace ppo::model {

/**
* @brief The ECS world: its entities, agents and rules, stepped by its caller.
*
* Each step has two halves: BeginStep (rules before the step, the re-initialization after an
* ended episode) and EndStep (agents choose, entities advance, rules resolve, agents read the
* new state). The caller observes the world between them (SteppedWorld).
*/
class Environment
{
public:
    /**
    * @brief Creates an empty environment.
    */
    Environment(void);

    /**
    * @brief Destroys the environment.
    */
    virtual ~Environment(void);

    Environment(Environment&) = delete;
    Environment(Environment&&) = delete;
    Environment& operator=(Environment&) = delete;
    Environment& operator=(Environment&&) = delete;

    /**
    * @brief Creates and registers one entity instance.
    * @tparam T Entity type.
    * @param args Constructor arguments.
    * @return Created entity or nullptr.
    */
    template<typename T, typename... Args>
    std::shared_ptr<T>
    CreateEntity(Args&&... args)
    {
        if constexpr (!std::is_base_of_v<Entity, T>)
            return nullptr;

        auto entity{ std::make_shared<T>(GetRegistry(), std::forward<Args>(args)...) };

        if (entity != nullptr)
        {
            if (auto initializer = GetEntityInitializer(); initializer != nullptr)
                entity->SetInitializerSystem(initializer);
            AddEntity(entity);
        }

        return entity;
    };

    /**
    * @brief Creates and registers one agent instance.
    * @tparam Ag Agent type.
    * @param args Constructor arguments.
    * @return Created agent or nullptr.
    */
    template<typename Ag, typename... Args>
    std::shared_ptr<Ag>
    CreateAgent(Args&&... args)
    {
        if constexpr (!std::is_base_of_v<IAgent, Ag>)
            return nullptr;

        auto agent{ std::make_shared<Ag>(std::forward<Args>(args)...) };
        if (agent != nullptr)
            AddAgent(agent);

        return agent;
    };

    /**
    * @brief Installs the scenario's outcome rules.
    * @return False when the rules are null.
    */
    bool SetWorldRules(std::unique_ptr<IWorldRules> rules);

    /**
    * @brief Sets the initializer given to the entities created next (episode resets).
    * @return Always true.
    */
    bool SetEntityInitializer(Entity::InitializerSystem initializer) noexcept;

    /**
    * @brief First half of one step: the rules before the step, and the re-initialization of
    *        the entities and agents when the last step ended an episode (or one was requested).
    *        The world is then ready to be observed.
    */
    void BeginStep(void);
    /**
    * @brief Second half of the step begun by BeginStep: agents choose, entities advance, the
    *        rules resolve, agents read the new state; then the rules after the step.
    * @return True when an agent ended the episode: the next BeginStep re-initializes.
    */
    bool EndStep(void);
    /**
    * @brief The live world, read only: the state its agents read.
    * @return The world's registry.
    */
    entt::registry const& SteppedWorld(void) const noexcept;
    /**
    * @brief The next BeginStep re-initializes the entities and agents, as after an episode
    *        an agent ended (a caller that starts a new episode in the middle of one).
    */
    void RequestReinitialization(void) noexcept;
    /**
    * @brief Seeds the engine the entity initializers draw from (sInitializationRandom): the
    *        same seed gives the same episodes. Unseeded, it draws from the system's entropy.
    * @param seed Seed of the initializations that follow.
    */
    void SeedInitialization(std::uint64_t seed);
    /**
    * @brief Applies an Environment.json: the world's limits and time step.
    * @param file The settings file.
    * @throws std::runtime_error When the file has no "environment" section.
    */
    void LoadSettings(std::filesystem::path const& file);

private:
    /**
    * @brief Returns the live registry pointer used for entity creation.
    */
    entt::registry* GetRegistry(void) noexcept;

    /**
    * @brief The initializer given to new entities.
    */
    Entity::InitializerSystem GetEntityInitializer(void) const noexcept;

    /**
    * @brief Stores one entity in the managed entity list.
    */
    void AddEntity(std::shared_ptr<Entity> entity);

    /**
    * @brief Stores one agent, once.
    */
    void AddAgent(std::shared_ptr<IAgent> pAgent);

    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::model */
