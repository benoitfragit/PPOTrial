#include "twovstwoexample.hpp"
#include "initializer.hpp"
#include "spawn_config.hpp"
#include "twovstworules.hpp"
#include "patrol_agent.hpp"
#include "patrol_team_agent.hpp"
#include "patrol_reference_agent.hpp"
#include "patrol_config.hpp"

#include "airspacecomponents.hpp"
#include "airspacemeta.hpp"
#include "components.hpp"
#include "environment.hpp"
#include "metaregistry.hpp"
#include "scenarioconfig.hpp"
#include "serializer.hpp"
#include "sharedworldcomponents.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
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

    if (settings.contains("blueController")
        && !settings["blueController"].is_string())
        return false;
    std::string const blueController =
        settings.value("blueController", std::string{"mappo"});
    bool const useMappo = blueController == "mappo";
    if (!useMappo && blueController != "reference")
        return false;
    auto const spawnSettings = ParseSpawnSettings(settings);
    if (!spawnSettings || !document.contains("entities")
        || !document["entities"].is_object())
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
    std::mt19937 validationRng{0U};
    WorldBounds const worldBounds{0.5F * worldNorth, 0.5F * worldEast};
    if (!GenerateEpisodeSpawnLayout(
            1, *spawnSettings, worldBounds, validationRng))
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

    auto const settingsRoot = path.parent_path().parent_path();
    defenders::Parameters patrolParameters{};
    defenders::PatrolGeometrySettings patrolGeometrySettings{};
    defenders::PatrolTrainingSettings patrolTraining{};
    try
    {
        patrolParameters = defenders::LoadPatrolParameters(settingsRoot);
        patrolGeometrySettings =
            defenders::LoadPatrolGeometrySettings(settingsRoot);
        patrolTraining = defenders::LoadPatrolTrainingSettings(settingsRoot);
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO] Configuration failed: "
                  << error.what() << '\n';
        return false;
    }

    auto& context = registry->ctx().emplace<InitializerContext>();
    context.rng.seed(patrolTraining.seed);
    context.count = entities.size();
    context.spawnSettings = *spawnSettings;
    context.patrolOnly = patrolOnly;
    context.patrolCruiseSpeed = patrolParameters.cruiseSpeed;
    context.patrolMaximumRotationRate = patrolParameters.maxRotationRate;
    context.patrolCollisionDistance = patrolParameters.collisionDistance;
    context.patrolGeometrySettings = patrolGeometrySettings;
    std::shared_ptr<defenders::PatrolPolicy> bluePolicy;
    auto const modelRoot = std::filesystem::path{RESOURCE_PATH}.parent_path()
        / "model" / "BluePatrol";
    auto const latestCheckpoint = modelRoot / "latest"
        / "blue_patrol_mappo_v6.txt";
    auto const trainingCheckpoint = modelRoot / "latest"
        / "blue_patrol_training_v3.txt";
    auto const productionCheckpoint = modelRoot / "prod"
        / "blue_patrol_mappo_v6.txt";
    if (useMappo)
    {
        std::filesystem::path const warmStartCheckpoint =
            std::filesystem::exists(latestCheckpoint)
            ? latestCheckpoint : productionCheckpoint;
        if (std::filesystem::exists(warmStartCheckpoint))
        {
            try
            {
                bluePolicy = defenders::LoadPatrolPolicy(warmStartCheckpoint);
            }
            catch (std::exception const& error)
            {
                std::cerr << "[BLUE MAPPO] Incompatible checkpoint: "
                          << error.what() << ". Starting a new Actor.\n";
                bluePolicy.reset();
            }
        }
        if (!bluePolicy)
        {
            try
            {
                bluePolicy = defenders::CreateUntrainedPatrolPolicy(
                    patrolTraining.seed);
            }
            catch (std::exception const& error)
            {
                std::cerr << "[BLUE MAPPO] Cannot initialize Actor: "
                          << error.what() << '\n';
                return false;
            }
            std::cout << "[BLUE MAPPO] Actor initialized; Play starts embedded "
                         "RLtools v6 training from new weights.\n";
        }
    }
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
    if (context.patrolGeometrySamplingFailed || !context.patrolGeometryReady)
        return false;
    patrolParameters.geometry = context.patrolGeometry;
    try
    {
        auto& first = *entities.at("blue_guard_1");
        auto& second = *entities.at("blue_guard_2");
        if (useMappo)
        {
            std::array<ppo::model::BaseEntity*, 2> guards{&first, &second};
            std::vector<ppo::model::BaseEntity*> attackers;
            if (auto red = entities.find("red_1"); red != entities.end())
                attackers.push_back(red->second.get());
            if (auto red = entities.find("red_2"); red != entities.end())
                attackers.push_back(red->second.get());
            auto controller = environment.CreateAgent<
                defenders::PatrolMappoAgent>(
                    guards, attackers, *entities.at("blue_objective"),
                    bluePolicy, patrolParameters, patrolTraining,
                    latestCheckpoint, trainingCheckpoint,
                    productionCheckpoint);
            if (!controller)
                return false;
            agents.push_back(std::move(controller));
        }
        else
        {
            auto leader = environment.CreateAgent<
                defenders::PatrolReferenceAgent>(first, second, 0U);
            auto follower = environment.CreateAgent<
                defenders::PatrolReferenceAgent>(second, first, 1U);
            if (!leader || !follower)
                return false;
            agents.push_back(std::move(leader));
            agents.push_back(std::move(follower));
        }
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO] Configuration failed: "
                  << error.what() << '\n';
        return false;
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
        "TWO VS TWO - MAPPO vs RLMission",
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
        "BLUE PATROL ONLY - NO RED",
        std::filesystem::path("Scenarios") / "BluePatrol.json",
        &ppo::examples::airspace::RegisterAirspaceMeta,
        &LoadScenario,
        "Two blue defenders train a cooperative RLtools MAPPO patrol without attackers."
    };
}

} // namespace ppo::examples::twovstwo
