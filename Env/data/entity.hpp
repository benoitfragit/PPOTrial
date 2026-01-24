#pragma once
#include "interface.hpp"

#include <memory>
#include <array>
#include <string>

namespace ppo::model {


class Entity : 
    public IEntity,
    public IEntityState ,
    public ICheckPointBuffer
{
public:
    class IEnvironment
    {
    public:
        virtual           ~IEnvironment(void) = default;
        virtual float     GetTimeStep(void) const noexcept = 0;
        virtual Entity*   GetEntityById(EntityId playerId) noexcept = 0;
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
        Entity* GetEntityById(EntityId playerId) noexcept 
        { 
            return dynamic_cast<Entity*>(m_env.GetIEntityById(playerId)); 
        };
        float   GetMaxNorth(void) const noexcept { return m_env.GetMaxNorth(); };
        float   GetMaxEast(void) const noexcept { return m_env.GetMaxEast(); };

    private:
        Ty& m_env;
    };


                      template<typename Ty>
                      Entity             (std::string const& name, 
                                          eEntitySide side,
                                          eEntityType type,
                                          Ty& env)  :
                          Entity(name, side, type, env.GetNewEntityId(), std::make_shared<EnvironmentWrapper<Ty>>(env))
                      {
                      
                      };

                      
    virtual                   ~Entity             (void);
                      
                      Entity              (Entity&)  = delete;
                      Entity              (Entity&&) = delete;
    Entity&           operator=           (Entity&)  = delete;
    Entity&           operator=           (Entity&&) = delete;
                      
    void              Init                (void)                                     noexcept override;
    void              Run                 (void)                                     noexcept override;
    void              UpdateEnv           (void)                               const noexcept override;
                      
    std::string const& GetName            (void)                               const noexcept;
    bool              GetDestroyed        (void)                               const noexcept;
    float             GetNorthPosition    (void)                               const noexcept;
    float             GetEastPosition     (void)                               const noexcept;
    void              SetDestroyed        (bool destroyed)                           noexcept;
    float             GetNorthSpeed       (void)                               const noexcept;
    float             GetEastSpeed        (void)                               const noexcept;
    EntityId          GetEntityId         (void)                               const noexcept;
    float             GetRotation         (void)                               const noexcept;

    sEntityState       GetSnapshot         (void)                               const noexcept final;
    ISNAState*         GetISNA             (void)                                   noexcept override  {return nullptr;};
    IFlightModelState* GetIFlightModel     (void)                                   noexcept override  {return nullptr;};
    CheckPointBuffer   GetCheckpoint       (void)                               noexcept override;
    void               SetCheckpoint       (CheckPointBuffer const& buffer)     noexcept override;

protected:
    Entity(std::string const& name, 
           eEntitySide side, 
           eEntityType type, 
           EntityId playerID, 
           std::shared_ptr<IEnvironment> pEnv);

    void         OnInit             (void);
    
    void         UpdateDoubleBuffer (void);

    sEntityState                    m_state{};
    std::shared_ptr<IEnvironment>   p_env{ nullptr };
    sEntityState                    m_buffers[2];
    std::atomic<uint32_t>           m_index{ 0 };
    std::string                     m_name{ "unknown" };
};


}   /* ppo::model */