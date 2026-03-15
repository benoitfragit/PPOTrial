#pragma once

#include "entt/meta/factory.hpp"
#include "entt/meta/resolve.hpp"

#include "environment.hpp"
#include "entitytypes.hpp"
#include "iagent.hpp"

#include <memory>
#include <string_view>

namespace ppo::reflection {

enum class eMetaCategory : unsigned char
{
    Unknown = 0,
    Enum,
    Entity,
    Component,
    Agent
};

struct sMetaDescriptor
{
    sMetaDescriptor(void) = default;
    sMetaDescriptor(eMetaCategory value, char const* name) :
        category{ value },
        scenario_name{ name }
    {
    }

    eMetaCategory category{ eMetaCategory::Unknown };
    char const* scenario_name{ "" };
};

void RegisterAllMeta(void);
bool IsMetaRegistered(void) noexcept;

entt::meta_type ResolveType(std::string_view name) noexcept;

std::shared_ptr<ppo::model::Entity>
CreateEntity(std::string_view name,
             ppo::model::Environment& environment,
             std::string_view platform,
             ppo::model::eEntitySide side);

std::shared_ptr<ppo::model::IAgent>
CreateAgent(std::string_view name,
            ppo::model::Environment& environment,
            ppo::model::BaseEntity& entity,
            std::string_view version = "prod");

} /* ppo::reflection */
