#pragma once

#include <array>
#include <memory>

#include "entt/entt.hpp"

namespace ppo::model {

class BaseEntity {
public:
    explicit BaseEntity(entt::registry* registry):
        p_registry{ registry }
    {
        if (p_registry != nullptr)
            m_entity = p_registry->create();
    };

    BaseEntity(BaseEntity& other)
    {
        m_entity   = other.uniqid();
        p_registry = other.registry();
    };

    BaseEntity& operator=(BaseEntity& other)
    {
        this->m_entity   = other.uniqid();
        this->p_registry = other.registry();
        return *this;
    };

    template<typename... Components>
    bool 
    has(void)
    {
        if (p_registry == nullptr)
            return false;

        return p_registry->all_of<Components...>(m_entity);
    };

    template<typename Component, typename... Args>
    Component& 
    add(Args&&... args)
    {
        assert(p_registry != nullptr);

        return p_registry->emplace<Component>(m_entity, std::forward<Args>(args)...);
    };

    template<typename... Components>
    auto 
    get(void) {
        assert(p_registry != nullptr);
        return p_registry->view<Components...>().get(m_entity);
    };

    template<typename... Components>
    auto const 
    get(void) const {
        assert(p_registry != nullptr);
        auto& view = p_registry->view<Components...>();
        return std::as_const<view>.get(m_entity);
    };

    entt::entity    uniqid(void) const noexcept { return m_entity; };

    entt::registry* registry(void) noexcept { return p_registry; };

    explicit operator bool(void) const noexcept {
        return m_entity != entt::null;
    };

protected:
    entt::registry* p_registry{ nullptr };
    entt::entity    m_entity{ entt::null };
};

}   /* ppo::model */