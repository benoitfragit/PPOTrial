#pragma once 
#include "interface.hpp"
#include "entity.hpp"
#include "iagent.hpp"
#include "randominitializersystem.hpp"

#include <string>
#include <memory>
#include <array>
#include <functional>
#include <filesystem>

namespace ppo::model {

/**
 * @brief Runtime container responsible for simulation orchestration.
 *
 * The environment owns entities and agents, exposes lifecycle controls,
 * and provides a thread-safe snapshot mechanism over the ECS registry.
 */
class Environment {
public:

                      Environment     (void);

    virtual           ~Environment    (void);

                      Environment           (Environment&)  = delete;
                      Environment           (Environment&&) = delete;
    Environment&      operator=             (Environment&)  = delete;
    Environment&      operator=             (Environment&&) = delete;

    template<typename T, typename... Args>
    /**
     * @brief Creates and registers a new entity instance.
     *
     * The created entity is wired with the random initializer system and then
     * stored in the internal entity list.
     *
     * @tparam T Entity type, must derive from Entity.
     * @tparam Args Constructor argument types.
     * @param args Constructor arguments forwarded to T.
     * @return Shared pointer to the created entity, or nullptr when T is not
     *         an Entity-derived type.
     */
    std::shared_ptr<T> 
    CreateEntity(Args&&... args)
    {
        if constexpr (!std::is_base_of_v<Entity, T>)
            return nullptr;

        auto entity{ std::make_shared<T>(GetRegistry(), std::forward<Args>(args)...) };
        
        if (entity != nullptr)
        {
            entity->SetInitializerSystem(&RandomInitiliazerSystem::Init);
            
            AddEntity(entity);
        }

        return entity;
    };

    template<typename Ag, typename... Args>
    /**
     * @brief Creates and registers a new agent instance.
     *
     * @tparam Ag Agent type, must derive from IAgent.
     * @tparam Args Constructor argument types.
     * @param args Constructor arguments forwarded to Ag.
     * @return Shared pointer to the created agent, or nullptr when Ag is not
     *         an IAgent-derived type.
     */
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
     * @brief Reads a consistent snapshot of the internal registry.
     * @param reader Callback executed with the snapshot registry.
     */
    void              ReadSnapshot          (std::function<void(entt::registry const&)> const& reader) const;

    /** @brief Starts the simulation loop. */
    void              Start                 (void);
    /** @brief Resumes execution when paused. */
    void              Play                  (void);
    /** @brief Pauses the simulation loop. */
    void              Pause                 (void);
    /** @brief Stops the simulation and releases runtime resources. */
    void              Stop                  (void);

    /**
     * @brief Queues a real-time command targeting one agent.
     * @param command Command payload.
     */
    void              AddAgentCommand       (sAgentRealTimeCommand const& command);

    /**
     * @brief Sets the environment spatial limits.
     * @param north Max range on north axis.
     * @param east Max range on east axis.
     */
    void              SetMaxRange           (float north, float east);
    /**
     * @brief Sets the simulation time step in seconds.
     * @param timestep Fixed delta time.
     */
    void              SetTimeStep           (float timestep);
    
private:

    entt::registry* GetRegistry(void) noexcept;

    void AddEntity(std::shared_ptr<Entity> entity);

    void AddAgent(std::shared_ptr<IAgent> pAgent);

    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

}   /* ppo::model */            
