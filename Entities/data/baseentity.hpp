/**
* @file Entities/data/baseentity.hpp
* @brief Declarations for baseentity.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include <array>
#include <memory>

#include "entt/entt.hpp"

namespace ppo::model {

/**
 * @brief Lightweight ECS entity handle with helper utilities.
 */
class BaseEntity {
public:
    /**
     * @brief Creates a new entity in the registry when available.
     * @param registry Registry used to allocate the entity.
     */
    explicit BaseEntity(entt::registry* registry):
        p_registry{ registry }
    {
        if (p_registry != nullptr)
            m_entity = p_registry->create();
    };

    /**
     * @brief Builds a wrapper around an existing entity id.
     * @param registry Registry owning the entity id.
     * @param entity Existing ECS identifier.
     */
    BaseEntity(entt::registry* registry, entt::entity entity):
        p_registry{ registry },
        m_entity{ entity }
    {
    };

    /** @brief Copy constructor preserving registry and entity id. */
    BaseEntity(BaseEntity& other)
    {
        m_entity   = other.uniqid();
        p_registry = other.registry();
    };

    /** @brief Copy assignment preserving registry and entity id. */
    BaseEntity& operator=(BaseEntity& other)
    {
        this->m_entity   = other.uniqid();
        this->p_registry = other.registry();
        return *this;
    };

    /**
     * @brief Checks whether the entity has all requested components.
     * @tparam Components Component types to query.
     * @return True if every component is present.
     */
    template<typename... Components>
    bool 
    has(void)
    {
        if (p_registry == nullptr)
            return false;

        return p_registry->all_of<Components...>(m_entity);
    };

    /**
     * @brief Adds one component to the entity.
     * @tparam Component Component type to emplace.
     * @tparam Args Constructor argument types.
     * @param args Constructor arguments forwarded to the component.
     * @return Reference to the inserted component.
     */
    template<typename Component, typename... Args>
    Component& 
    add(Args&&... args)
    {
        assert(p_registry != nullptr);

        return p_registry->emplace<Component>(m_entity, std::forward<Args>(args)...);
    };

    /**
     * @brief Gets one or several components from the entity.
     * @tparam Components Component types to retrieve.
     * @return Component tuple or reference as provided by EnTT.
     */
    template<typename... Components>
    auto 
    get(void) {
        assert(p_registry != nullptr);
        return p_registry->view<Components...>().get(m_entity);
    };

    /**
     * @brief Gets one or several components from a const entity wrapper.
     * @tparam Components Component types to retrieve.
     * @return Const component tuple or reference as provided by EnTT.
     */
    template<typename... Components>
    auto const 
    get(void) const {
        assert(p_registry != nullptr);
        auto& view = p_registry->view<Components...>();
        return std::as_const<view>.get(m_entity);
    };

    /** @brief Returns the underlying ECS entity id. */
    entt::entity    uniqid(void) const noexcept { return m_entity; };

    /** @brief Returns the attached ECS registry pointer. */
    entt::registry* registry(void) noexcept { return p_registry; };

    /** @brief Checks whether the wrapper references a valid entity id. */
    explicit operator bool(void) const noexcept {
        return m_entity != entt::null;
    };

protected:
    entt::registry* p_registry{ nullptr };
    entt::entity    m_entity{ entt::null };
};

}   /* ppo::model */