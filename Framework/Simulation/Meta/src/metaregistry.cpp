/**
* @file Meta/src/metaregistry.cpp
* @brief The factory tables of the scenarios.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "metaregistry.hpp"

#include <map>
#include <mutex>
#include <string>

namespace ppo::reflection {

namespace {

/** @brief Entity factories by scenario name. */
std::map<std::string, EntityFactory, std::less<>>&
EntityFactories(void)
{
    static std::map<std::string, EntityFactory, std::less<>> factories{};
    return factories;
}

/** @brief Agent factories by scenario name. */
std::map<std::string, AgentFactory, std::less<>>&
AgentFactories(void)
{
    static std::map<std::string, AgentFactory, std::less<>> factories{};
    return factories;
}

/** @brief Guards both tables (several environments may register and create). */
std::mutex&
FactoryMutex(void)
{
    static std::mutex mutex{};
    return mutex;
}

/** @brief The factory registered under `name`, or null. */
template<typename Factory>
Factory
Find(std::map<std::string, Factory, std::less<>> const& factories, std::string_view name)
{
    std::lock_guard<std::mutex> lock(FactoryMutex());
    auto const iter = factories.find(name);
    return iter != factories.end() ? iter->second : nullptr;
}

} // namespace

void RegisterEntityFactory(std::string_view name, EntityFactory factory)
{
    std::lock_guard<std::mutex> lock(FactoryMutex());
    EntityFactories()[std::string(name)] = factory;
}

void RegisterAgentFactory(std::string_view name, AgentFactory factory)
{
    std::lock_guard<std::mutex> lock(FactoryMutex());
    AgentFactories()[std::string(name)] = factory;
}

std::shared_ptr<ppo::model::Entity>
CreateEntity(std::string_view name,
             ppo::model::Environment& environment,
             std::string_view platform,
             ppo::model::eEntitySide side)
{
    EntityFactory const factory = Find(EntityFactories(), name);
    return factory != nullptr ? factory(environment, platform, side) : nullptr;
}

std::shared_ptr<ppo::model::IAgent>
CreateAgent(std::string_view name,
            ppo::model::Environment& environment,
            ppo::model::BaseEntity& entity,
            std::string_view version)
{
    AgentFactory const factory = Find(AgentFactories(), name);
    return factory != nullptr ? factory(environment, entity, version) : nullptr;
}

} /* ppo::reflection */
