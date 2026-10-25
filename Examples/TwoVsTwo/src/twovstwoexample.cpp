#include "twovstwoexample.hpp"
#include "initializer.hpp"
#include "twovstworules.hpp"
#include "blue_agent.hpp"

#include "airspacecomponents.hpp"
#include "airspacemeta.hpp"
#include "components.hpp"
#include "environment.hpp"
#include "metaregistry.hpp"
#include "scenarioconfig.hpp"
#include "serializer.hpp"
#include "sharedworldcomponents.hpp"

#include <cmath>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ppo::examples::twovstwo {
namespace {

bool LoadScenario(std::filesystem::path const& path,
                  ppo::model::Environment& environment)
{
    if (!environment.SetEntityInitializer(&InitializeEntity)
        || !environment.SetWorldRules(std::make_unique<TwoVsTwoWorldRules>()))
        return false;

    auto const scenario = ppo::config::LoadScenarioDefinition(path);
    if (scenario.m_entities.empty())
        return false;

    auto const document = ppo::serialization::Serializer::Parse(path);
    if (!document.is_object() || !document.contains("twoVsTwo")
        || !document["twoVsTwo"].is_object())
        return false;

    auto const& settings = document["twoVsTwo"];
    if (settings.contains("patrolOnly") && !settings["patrolOnly"].is_boolean())
        return false;
    bool const patrolOnly = settings.value("patrolOnly", false);
    if (!settings.contains("red2InitiallyDestroyedProbability")
        || !settings["red2InitiallyDestroyedProbability"].is_number())
        return false;
    float const probability = settings["red2InitiallyDestroyedProbability"].get<float>();
    if (!std::isfinite(probability) || probability < 0.F || probability > 1.F
        || !document.contains("entities") || !document["entities"].is_object())
        return false;

    // Initial entity spawns precede Serializer::Load. Read its exact file
    // so the first episode and later resets use the same LiDAR/world limits.
    auto const environmentConfig =
        ppo::serialization::Serializer::Parse(path.parent_path().parent_path() / "Environment.json");
    if (!environmentConfig.is_object()
        || !environmentConfig.contains("formatVersion")
        || !environmentConfig["formatVersion"].is_number_integer()
        || environmentConfig["formatVersion"].get<int>() != 2
        || !environmentConfig.contains("environment")
        || !environmentConfig["environment"].is_object())
        return false;
    auto const& worldConfig = environmentConfig["environment"];
    if (!worldConfig.contains("maxNorth") || !worldConfig["maxNorth"].is_number()
        || !worldConfig.contains("maxEast") || !worldConfig["maxEast"].is_number())
        return false;
    float const worldNorth = worldConfig["maxNorth"].get<float>();
    float const worldEast = worldConfig["maxEast"].get<float>();
    if (!std::isfinite(worldNorth) || !std::isfinite(worldEast)
        || worldNorth <= 0.F || worldEast <= 0.F)
        return false;

    std::map<std::string, std::shared_ptr<ppo::model::Entity>> entities;
    std::vector<std::shared_ptr<ppo::model::IAgent>> agents;

    for (auto const& item : scenario.m_entities)
    {
        auto const& itemConfig = document["entities"][item.m_name];
        if (!itemConfig.is_object())
            return false;
        if (itemConfig.contains("enabled"))
        {
            if (item.m_name != "red_2" || !itemConfig["enabled"].is_boolean())
                return false;
            if (!itemConfig["enabled"].get<bool>())
                continue;
        }

        bool const mobile = item.m_type == "Mobile";
        bool const zone = item.m_type == "RestrictedZone";
        bool const objective = item.m_type == "Objective";
        if ((!mobile && !zone && !objective)
            || ((mobile || zone) && item.m_platform.empty()))
            return false;

        auto const blue = item.m_name == "blue_guard_1"
            || item.m_name == "blue_guard_2"
            || item.m_name == "blue_objective"
            || item.m_name == "blue_restricted_zone";
        auto const red = item.m_name == "red_1" || item.m_name == "red_2";

        if ((!blue && !red) || (patrolOnly && red)
            || (blue && item.m_side != ppo::model::eEntitySide::Friend)
            || (red && item.m_side != ppo::model::eEntitySide::Hostile)
            || (blue && (!item.m_agents.empty() || !item.m_target.empty()))
            || (red && (item.m_target != "blue_objective"
                || item.m_agents.size() != 1U
                || item.m_agents.front() != "TargetSelectorProdAgent"))
            || ((item.m_name == "blue_guard_1" || item.m_name == "blue_guard_2" || red) && !mobile)
            || (item.m_name == "blue_objective" && !objective)
            || (item.m_name == "blue_restricted_zone" && !zone))
            return false;

        auto entity = ppo::reflection::CreateEntity(item.m_type,
                                                     environment, item.m_platform, item.m_side);
        if (!entity)
            return false;

        entity->add<ppo::model::sEntityMetadata>(
            ppo::model::sEntityMetadata{item.m_name, item.m_platform});
        if (mobile)
        {
            auto& sensor = entity->add<ppo::model::sPseudoLidarState>();
            sensor.m_nearestCollisionRanges.fill(1.F);
        }

        for (auto const& agentName : item.m_agents)
        {
            auto agent = ppo::reflection::CreateAgent(agentName, environment, *entity);
            if (!agent)
                return false;
            agents.push_back(std::move(agent));
        }

        if (!entities.emplace(item.m_name, std::move(entity)).second)
            return false;
    }

    if (!entities.count("blue_guard_1") || !entities.count("blue_guard_2")
        || !entities.count("blue_objective") || !entities.count("blue_restricted_zone")
        || (!patrolOnly && !entities.count("red_1"))
        || entities.size() != (patrolOnly ? 4U :
            5U + static_cast<std::size_t>(entities.count("red_2") != 0)))
        return false;

    for (auto const& item : scenario.m_entities)
    {
        auto source = entities.find(item.m_name);
        if (source == entities.end() || item.m_target.empty())
            continue;
        auto target = entities.find(item.m_target);
        if (target == entities.end() || !source->second->has<ppo::model::sSNAState>())
            return false;
        auto [sna] = source->second->get<ppo::model::sSNAState>();
        sna.m_target = target->second->uniqid();
    }

    auto* registry = entities.begin()->second->registry();
    if (!registry)
        return false;
    auto& context = registry->ctx().emplace<InitializerContext>();
    context.count = entities.size();
    context.red2DestroyedProbability = probability;
    context.patrolOnly = patrolOnly;
    std::shared_ptr<defenders::BluePolicy> bluePolicy;
    if (patrolOnly)
    {
        auto const checkpoint = std::filesystem::path{RESOURCE_PATH}.parent_path()
            / "model" / "BluePatrol" / "prod" / "blue_patrol_v1.txt";
        if (std::filesystem::exists(checkpoint))
        {
            try
            {
                bluePolicy = std::make_shared<defenders::BluePolicy>(checkpoint);
            }
            catch (std::exception const&)
            {
                return false; // Do not silently fly with incompatible weights.
            }
        }
    }
    context.policyReady = bluePolicy != nullptr;
    auto& world = registry->ctx().get<ppo::model::sEnvironmentState>();
    world.m_maxNorth = worldNorth;
    world.m_maxEast = worldEast;
    // Rendering metadata is derived from these same limits on each snapshot.

    // The legacy desktop does not initialize the very first episode in Run.
    // Initialize this standalone scenario once; subsequent resets use the
    // registered initializer via Environment::Run.
    for (auto const& item : scenario.m_entities)
    {
        auto it = entities.find(item.m_name);
        if (it != entities.end())
            it->second->Init();
    }
    if (bluePolicy)
    {
        auto guard1 = environment.CreateAgent<defenders::BlueProdAgent>(
            *entities.at("blue_guard_1"), *entities.at("blue_guard_2"), 0U, bluePolicy);
        auto guard2 = environment.CreateAgent<defenders::BlueProdAgent>(
            *entities.at("blue_guard_2"), *entities.at("blue_guard_1"), 1U, bluePolicy);
        if (!guard1 || !guard2)
            return false;
        agents.push_back(std::move(guard1));
        agents.push_back(std::move(guard2));
    }
    for (auto const& agent : agents)
        agent->Init();

    return true;
}

} // namespace

ppo::app::ExampleDefinition Definition()
{
    return {
        "two-vs-two",
        "TWO VS TWO",
        std::filesystem::path("Scenarios") / "TwoVsTwo.json",
        &ppo::examples::airspace::RegisterAirspaceMeta,
        &LoadScenario,
        "Two blue defenders protect a fixed objective against RL-controlled red cars."
    };
}

ppo::app::ExampleDefinition BluePatrolDefinition()
{
    return {
        "blue-patrol",
        "BLUE PATROL PPO",
        std::filesystem::path("Scenarios") / "BluePatrol.json",
        &ppo::examples::airspace::RegisterAirspaceMeta,
        &LoadScenario,
        "Two blue defenders patrol without red attackers; RLtools policy used when trained."
    };
}

} // namespace ppo::examples::twovstwo
