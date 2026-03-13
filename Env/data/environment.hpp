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

class Environment {
public:

                      Environment     (void);

    virtual           ~Environment    (void);

                      Environment           (Environment&)  = delete;
                      Environment           (Environment&&) = delete;
    Environment&      operator=             (Environment&)  = delete;
    Environment&      operator=             (Environment&&) = delete;

    template<typename T, typename... Args>
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

    void              ReadSnapshot          (std::function<void(entt::registry const&)> const& reader) const;

    void              Start                 (void);
    void              Play                  (void);
    void              Pause                 (void);
    void              Stop                  (void);

    void              AddAgentCommand       (sAgentRealTimeCommand const& command);

    void              SetMaxRange           (float north, float east);
    void              SetTimeStep           (float timestep);
    
private:

    entt::registry* GetRegistry(void) noexcept;

    void AddEntity(std::shared_ptr<Entity> entity);

    void AddAgent(std::shared_ptr<IAgent> pAgent);

    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

}   /* ppo::model */            
