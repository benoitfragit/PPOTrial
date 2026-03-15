#pragma once

#include "baseentity.hpp"

#include "interface.hpp"

namespace ppo::model {

#include <memory>

/**
 * @brief Base runtime wrapper for all world entities.
 *
 * Entity instances are ECS-backed objects with optional initialization logic
 * provided by an external initializer callback.
 */
class Entity: public BaseEntity
{
public:
    /**
     * @brief Constructs an entity and attaches side/type metadata.
     * @param registry ECS registry owning the instance.
     * @param side Team or faction side.
     * @param type Functional entity type.
     */
    Entity(entt::registry* registry, eEntitySide side, eEntityType type);

    /** @brief Entity initialization callback signature. */
    using InitializerSystem = void(*)(Entity&);
    /**
     * @brief Assigns the initializer callback executed by Init().
     * @param system Function pointer to initialization logic.
     */
    void SetInitializerSystem(InitializerSystem system);

    /** @brief Executes entity initialization flow. */
    virtual void Init(void) noexcept;
    /** @brief Executes one runtime step for the entity. */
    virtual void Run (void) noexcept;
protected:
    /** @brief Entity-specific initialization hook. */
    virtual void OnInit(void) noexcept;

    /** @brief Optional initializer callback.
     */
    InitializerSystem p_initializer{ nullptr };
};

} /* ppo::model */