/**
* @file Env/src/environment.cpp
* @brief Implementation of the caller-driven ECS world.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "environment.hpp"
#include "ppo/core/multi_agent.hpp"
#include "entityruntime.hpp"
#include "serializer.hpp"
#include "utils.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ppo::model {

/**
* @brief Hidden implementation owning the world, its entities, agents and rules.
*/
struct Environment::Impl
{
    /** @brief Applies the "environment" section of an Environment.json (limits, time step). */
    void ApplySettings(ppo::serialization::json const& document);

    Impl(void);

    entt::registry                      m_world{};
    std::vector<std::shared_ptr<Entity>> m_entities{};
    std::vector<std::shared_ptr<IAgent>> m_agents{};
    std::unique_ptr<IWorldRules>        m_worldRules{};
    Entity::InitializerSystem           m_entityInitializer{ nullptr };
    /** @brief An agent ended the episode: the next step re-initializes entities and agents. */
    bool                                m_reinitRequest{ false };
};

/**
* @brief Creates the world with its environment-state context.
*/
Environment::Impl::Impl(void)
{
    // The shared environment state exists before any entity is added.
    m_world.ctx().emplace<sEnvironmentState>();
}

/**
* @brief Applies the world limits and the time step of an Environment.json.
* @throws nlohmann::json::exception When a value is missing or not a number.
*/
void
Environment::Impl::ApplySettings(ppo::serialization::json const& document)
{
    auto const& section = document.at("environment");
    auto& state = m_world.ctx().get<sEnvironmentState>();
    state.m_maxNorth = section.at("maxNorth").get<float>();
    state.m_maxEast  = section.at("maxEast").get<float>();
    state.m_dt       = section.at("timeStep").get<float>();
}

Environment::Environment(void):
    p_impl{std::make_unique<Impl>()}
{
}

Environment::~Environment(void) = default;

bool Environment::SetWorldRules(std::unique_ptr<IWorldRules> rules)
{
    if (rules == nullptr)
        return false;

    p_impl->m_worldRules = std::move(rules);
    return true;
}

entt::registry*
Environment::GetRegistry(void) noexcept
{
    return &p_impl->m_world;
}

Entity::InitializerSystem
Environment::GetEntityInitializer(void) const noexcept
{
    return p_impl->m_entityInitializer;
}

bool
Environment::SetEntityInitializer(Entity::InitializerSystem initializer) noexcept
{
    p_impl->m_entityInitializer = initializer;
    return true;
}

void
Environment::AddEntity(std::shared_ptr<Entity> entity)
{
    if (entity != nullptr)
        p_impl->m_entities.push_back(std::move(entity));
}

void
Environment::AddAgent(std::shared_ptr<IAgent> pAgent)
{
    if (pAgent == nullptr
    ||  std::find(p_impl->m_agents.begin(), p_impl->m_agents.end(), pAgent) != p_impl->m_agents.end())
        return;

    p_impl->m_agents.push_back(std::move(pAgent));
}

/**
* @brief First half of a step: rules before the step, re-initialization after an ended episode.
*/
void
Environment::BeginStep(void)
{
    if (p_impl->m_worldRules != nullptr)
        p_impl->m_worldRules->BeforeStep(p_impl->m_entities);

    if (!p_impl->m_reinitRequest)
        return;

    for (auto& entity : p_impl->m_entities)
    {
        if (entity != nullptr)
            entity->Init();
    }
    // Initialize agents only after every entity has rebuilt its state.
    ppo::core::ResetAgents(p_impl->m_agents, [](IAgent& agent) { agent.Init(); });
    p_impl->m_reinitRequest = false;
}

/**
* @brief Second half of a step: agents, entities, rules, agent feedback.
*/
bool
Environment::EndStep(void)
{
    // The domain-neutral multi-agent scheduler; the ECS update and outcome rules stay here.
    p_impl->m_reinitRequest |= ppo::core::StepAgents(
        p_impl->m_agents,
        [](IAgent& agent) { agent.ChooseAction(); },
        [this]()
        {
            AdvanceLivingEntities(p_impl->m_entities, [](auto& entity) -> auto& { return entity; });
        },
        [this]()
        {
            if (p_impl->m_worldRules != nullptr)
                p_impl->m_worldRules->Resolve(p_impl->m_entities);
        },
        [](IAgent& agent) { return agent.SetNewState(); });

    if (p_impl->m_worldRules != nullptr)
        p_impl->m_worldRules->AfterStep();
    return p_impl->m_reinitRequest;
}

entt::registry const&
Environment::SteppedWorld(void) const noexcept
{
    return p_impl->m_world;
}

void
Environment::LoadSettings(std::filesystem::path const& file)
{
    ppo::serialization::json const document = ppo::serialization::Serializer::Parse(file);
    if (!document.contains("environment"))
        throw std::runtime_error(file.string() + ": no \"environment\" section (world limits and time step)");
    p_impl->ApplySettings(document);
}

void
Environment::SeedInitialization(std::uint64_t seed)
{
    InitializationEngine(p_impl->m_world).seed(seed);
}

void
Environment::RequestReinitialization(void) noexcept
{
    p_impl->m_reinitRequest = true;
}

}
