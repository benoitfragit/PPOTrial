#include "twovstwoexample.hpp"
#include "initializer.hpp"
#include "spawn_config.hpp"
#include "twovstworules.hpp"
#include "patrol_agent.hpp"
#include "patrol_team_agent.hpp"
#include "generalized_patrol_team_agent.hpp"
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

/**
 * @brief Identifies a scenario entity usable as a production red template.
 * @param definition Parsed scenario entity definition.
 * @return True for one hostile mobile driven by production RLMission.
 */
bool IsProductionRedTemplate(
    ppo::config::ScenarioEntityDefinition const& definition) noexcept
{
    return definition.m_type == "Mobile"
        && definition.m_side == ppo::model::eEntitySide::Hostile
        && definition.m_target == "blue_objective"
        && definition.m_agents.size() == 1U
        && definition.m_agents.front() == "TargetSelectorProdAgent"
        && !definition.m_platform.empty();
}

/**
 * @brief Checks that two declarative red templates have identical semantics.
 * @param first First parsed template.
 * @param second Second parsed template.
 * @return True when either template can safely instantiate the same red team.
 */
bool SameProductionRedTemplate(
    ppo::config::ScenarioEntityDefinition const& first,
    ppo::config::ScenarioEntityDefinition const& second) noexcept
{
    return first.m_type == second.m_type
        && first.m_platform == second.m_platform
        && first.m_side == second.m_side
        && first.m_target == second.m_target
        && first.m_agents == second.m_agents;
}

/**
 * @brief Identifies a declarative blue mobile template for the shared patrol controller.
 * @param definition Parsed scenario entity definition.
 * @return True for one controller-free friendly mobile platform template.
 */
bool IsGeneralizedBlueTemplate(
    ppo::config::ScenarioEntityDefinition const& definition) noexcept
{
    return definition.m_type == "Mobile"
        && definition.m_side == ppo::model::eEntitySide::Friend
        && definition.m_target.empty() && definition.m_agents.empty()
        && !definition.m_platform.empty();
}

/**
 * @brief Checks semantic equivalence of two generalized blue templates.
 * @param first First template.
 * @param second Second template.
 * @return True when both instantiate the same centrally controlled platform.
 */
bool SameGeneralizedBlueTemplate(
    ppo::config::ScenarioEntityDefinition const& first,
    ppo::config::ScenarioEntityDefinition const& second) noexcept
{
    return first.m_type == second.m_type
        && first.m_platform == second.m_platform
        && first.m_side == second.m_side
        && first.m_target == second.m_target
        && first.m_agents == second.m_agents;
}

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
    std::string spawnDiagnostic;
    auto const spawnSettings = ParseSpawnSettings(settings, &spawnDiagnostic);
    if (!spawnSettings)
    {
        std::cerr << "[TWO_VS_TWO] Invalid spawn configuration: "
                  << spawnDiagnostic << '\n';
        return false;
    }
    if (!document.contains("entities")
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

    bool const generalizedRedTeam = spawnSettings->redTeam.has_value();
    std::size_t const configuredRedCount = generalizedRedTeam
        ? spawnSettings->redTeam->activeCount : 0U;
    if (patrolOnly && configuredRedCount != 0U)
    {
        std::cerr << "[TWO_VS_TWO] patrolOnly requires redTeam.activeCount=0.\n";
        return false;
    }
    bool const generalizedBlueTeam = spawnSettings->blueTeam.has_value();
    std::size_t const configuredBlueCount = generalizedBlueTeam
        ? spawnSettings->blueTeam->activeCount : 2U;
    if (generalizedBlueTeam && !useMappo)
    {
        std::cerr << "[BLUE MAPPO] blueTeam requires the generalized mappo "
                     "controller; the archived reference controller remains "
                     "strictly two-guard.\n";
        return false;
    }

    ppo::config::ScenarioEntityDefinition const* blueTemplate = nullptr;
    if (generalizedBlueTeam)
    {
        for (auto const& item : scenario.m_entities)
        {
            bool const friendlyMobile = item.m_type == "Mobile"
                && item.m_side == ppo::model::eEntitySide::Friend;
            if (!friendlyMobile)
                continue;
            if (!IsGeneralizedBlueTemplate(item))
            {
                std::cerr << "[BLUE MAPPO] Friendly mobile template '"
                          << item.m_name
                          << "' must have a platform and no per-entity agent; "
                             "the generalized patrol controller is shared.\n";
                return false;
            }
            if (blueTemplate == nullptr)
                blueTemplate = &item;
            else if (!SameGeneralizedBlueTemplate(*blueTemplate, item))
            {
                std::cerr << "[BLUE MAPPO] Generalized blue templates must use "
                             "the same platform and controller contract.\n";
                return false;
            }
        }
        if (blueTemplate == nullptr)
        {
            std::cerr << "[BLUE MAPPO] blueTeam requires one friendly Mobile "
                         "template for the shared patrol controller.\n";
            return false;
        }
    }

    ppo::config::ScenarioEntityDefinition const* redTemplate = nullptr;
    if (generalizedRedTeam)
    {
        for (auto const& item : scenario.m_entities)
        {
            bool const hostileMobile = item.m_type == "Mobile"
                && item.m_side == ppo::model::eEntitySide::Hostile;
            if (!hostileMobile)
                continue;
            if (!IsProductionRedTemplate(item))
            {
                std::cerr << "[TWO_VS_TWO] Hostile mobile template '"
                          << item.m_name
                          << "' must target blue_objective and use exactly one "
                             "production RLMission controller.\n";
                return false;
            }
            if (redTemplate == nullptr)
                redTemplate = &item;
            else if (!SameProductionRedTemplate(*redTemplate, item))
            {
                std::cerr << "[TWO_VS_TWO] Generalized red templates must use "
                             "the same platform and controller contract.\n";
                return false;
            }
        }
        if (configuredRedCount != 0U && redTemplate == nullptr)
        {
            std::cerr << "[TWO_VS_TWO] redTeam requires one hostile Mobile "
                         "template driven by production RLMission.\n";
            return false;
        }
    }

    std::map<std::string, std::shared_ptr<ppo::model::Entity>> entities;
    std::vector<std::shared_ptr<ppo::model::IAgent>> agents;

    for (auto const& item : scenario.m_entities)
    {
        if (generalizedRedTeam && IsProductionRedTemplate(item))
            continue;
        if (generalizedBlueTeam && IsGeneralizedBlueTemplate(item))
            continue;

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

    if (generalizedBlueTeam && blueTemplate != nullptr)
    {
        for (std::size_t index = 0U; index < configuredBlueCount; ++index)
        {
            std::string const name = "blue_" + std::to_string(index + 1U);
            auto entity = ppo::reflection::CreateEntity(
                blueTemplate->m_type, environment, blueTemplate->m_platform,
                blueTemplate->m_side);
            if (!entity)
                return false;
            entity->add<ppo::model::sEntityMetadata>(
                ppo::model::sEntityMetadata{name, blueTemplate->m_platform});
            auto& sensor = entity->add<ppo::model::sPseudoLidarState>();
            sensor.m_nearestCollisionRanges.fill(1.F);
            if (!entities.emplace(name, std::move(entity)).second)
                return false;
        }
    }

    if (generalizedRedTeam && !patrolOnly && redTemplate != nullptr)
    {
        for (std::size_t index = 0U; index < configuredRedCount; ++index)
        {
            std::string const name = "red_" + std::to_string(index + 1U);
            auto entity = ppo::reflection::CreateEntity(
                redTemplate->m_type, environment, redTemplate->m_platform,
                redTemplate->m_side);
            if (!entity)
                return false;
            entity->add<ppo::model::sEntityMetadata>(
                ppo::model::sEntityMetadata{name, redTemplate->m_platform});
            auto& sensor = entity->add<ppo::model::sPseudoLidarState>();
            sensor.m_nearestCollisionRanges.fill(1.F);

            // The validated template owns exactly one production RLMission
            // declaration. Every generated red receives its own controller.
            auto agent = ppo::reflection::CreateAgent(
                redTemplate->m_agents.front(), environment, *entity);
            if (!agent)
                return false;
            agents.push_back(std::move(agent));

            if (!entities.emplace(name, std::move(entity)).second)
                return false;
        }
    }

    // Restricted zones are discovered from their dedicated typed collection;
    // no scenario identifier is hard-coded in the loader.
    for (auto const& zone : spawnSettings->restrictedZones.items)
    {
        if (entities.count(zone.id) != 0U)
            return false;
        auto entity = ppo::reflection::CreateEntity(
            "RestrictedZone", environment, zone.platform,
            ppo::model::eEntitySide::Hostile);
        if (!entity)
            return false;
        entity->add<ppo::model::sEntityMetadata>(
            ppo::model::sEntityMetadata{zone.id, zone.platform});
        if (!entities.emplace(zone.id, std::move(entity)).second)
            return false;
    }

    std::size_t const blueCoreCount = generalizedBlueTeam
        ? configuredBlueCount + 2U : 4U;
    std::size_t const redCoreCount = generalizedRedTeam
        ? (patrolOnly ? 0U : configuredRedCount)
        : (patrolOnly ? 0U
            : 1U + static_cast<std::size_t>(entities.count("red_2") != 0U));
    std::size_t const expectedCoreEntities = blueCoreCount + redCoreCount;
    bool blueCollectionComplete = true;
    if (generalizedBlueTeam)
    {
        for (std::size_t index = 0U; index < configuredBlueCount; ++index)
            blueCollectionComplete &= entities.count(
                "blue_" + std::to_string(index + 1U)) != 0U;
    }
    if ((!generalizedBlueTeam
            && (!entities.count("blue_guard_1")
                || !entities.count("blue_guard_2")))
        || !blueCollectionComplete
        || !entities.count("blue_objective")
        || !entities.count("blue_restricted_zone")
        || (!generalizedRedTeam && !patrolOnly
            && !entities.count("red_1"))
        || entities.size() != expectedCoreEntities
            + spawnSettings->restrictedZones.items.size())
        return false;

    if (generalizedRedTeam)
    {
        auto const objective = entities.find("blue_objective");
        if (objective == entities.end())
            return false;
        for (std::size_t index = 0U; index < configuredRedCount; ++index)
        {
            auto const red = entities.find(
                "red_" + std::to_string(index + 1U));
            if (red == entities.end()
                || !red->second->has<ppo::model::sSNAState>())
                return false;
            auto [sna] = red->second->get<ppo::model::sSNAState>();
            sna.m_target = objective->second->uniqid();
        }
    }
    else
    {
        for (auto const& item : scenario.m_entities)
        {
            auto source = entities.find(item.m_name);
            if (source == entities.end() || item.m_target.empty())
                continue;
            auto target = entities.find(item.m_target);
            if (target == entities.end()
                || !source->second->has<ppo::model::sSNAState>())
                return false;
            auto [sna] = source->second->get<ppo::model::sSNAState>();
            sna.m_target = target->second->uniqid();
        }
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
        / "blue_patrol_mappo_v7.txt";
    auto const trainingCheckpoint = modelRoot / "latest"
        / "blue_patrol_training_v4.txt";
    auto const productionCheckpoint = modelRoot / "prod"
        / "blue_patrol_mappo_v7.txt";
    if (useMappo && !generalizedBlueTeam)
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
    for (auto const& [name, entity] : entities)
    {
        (void)name;
        entity->Init();
    }
    if (context.patrolGeometrySamplingFailed
        || context.episodeLayoutGenerationFailed
        || !context.patrolGeometryReady)
        return false;
    patrolParameters.geometry = context.patrolGeometry;
    try
    {
        if (generalizedBlueTeam)
        {
            std::array<ppo::model::BaseEntity*,
                       defenders::TeamCapacity::MaximumBlueMobiles> guards{};
            std::array<ppo::model::BaseEntity*,
                       defenders::TeamCapacity::MaximumRedMobiles> attackers{};
            for (std::size_t index = 0U; index < configuredBlueCount; ++index)
                guards[index] = entities.at(
                    "blue_" + std::to_string(index + 1U)).get();
            std::size_t const redCount = generalizedRedTeam
                ? configuredRedCount
                : (patrolOnly ? 0U
                    : 1U + static_cast<std::size_t>(
                        entities.count("red_2") != 0U));
            for (std::size_t index = 0U; index < redCount; ++index)
                attackers[index] = entities.at(
                    "red_" + std::to_string(index + 1U)).get();

            auto const generalizedLatest = modelRoot / "latest"
                / "blue_patrol_generalized_v8.txt";
            auto const generalizedTraining = modelRoot / "latest"
                / "blue_patrol_generalized_training_v5.txt";
            auto const generalizedProduction = modelRoot / "prod"
                / "blue_patrol_generalized_v8.txt";
            auto controller = environment.CreateAgent<
                defenders::GeneralizedPatrolMappoAgent>(
                    guards, configuredBlueCount, attackers, redCount,
                    *entities.at("blue_objective"), patrolParameters,
                    patrolTraining, generalizedLatest,
                    generalizedTraining, generalizedProduction,
                    trainingCheckpoint, latestCheckpoint,
                    productionCheckpoint);
            if (!controller)
                return false;
            agents.push_back(std::move(controller));
        }
        else
        {
            auto& first = *entities.at("blue_guard_1");
            auto& second = *entities.at("blue_guard_2");
            if (useMappo)
            {
                std::array<ppo::model::BaseEntity*, 2> guards{&first, &second};
                std::vector<ppo::model::BaseEntity*> attackers;
                std::size_t const redCount = generalizedRedTeam
                    ? configuredRedCount : 2U;
                attackers.reserve(redCount);
                for (std::size_t index = 0U; index < redCount; ++index)
                {
                    auto const red = entities.find(
                        "red_" + std::to_string(index + 1U));
                    if (red != entities.end())
                        attackers.push_back(red->second.get());
                }
                if (attackers.size() != 2U)
                    return false;
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

ppo::app::ExampleDefinition GeneralizedPvsNDefinition()
{
    return {
        "p-vs-n",
        "GENERALIZED P vs N - MAPPO v8 vs RLMission",
        std::filesystem::path("Scenarios") / "PvsN.json",
        &ppo::examples::airspace::RegisterAirspaceMeta,
        &LoadScenario,
        "Bounded generalized blue MAPPO team against production RLMission reds."
    };
}

} // namespace ppo::examples::twovstwo
