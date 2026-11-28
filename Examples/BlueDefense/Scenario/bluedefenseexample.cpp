/**
 * @file Examples/BlueDefense/Scenario/bluedefenseexample.cpp
 * @brief Chargement BlueDefense et BluePatrol et composition des entités et agents.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "bluedefenseexample.hpp"
#include "initializer.hpp"
#include "spawn_config.hpp"
#include "bluedefenserules.hpp"
#include "generalized_patrol_team_agent.hpp"
#include "patrol_config.hpp"

#include "airspacecomponents.hpp"
#include "airspacemeta.hpp"
#include "components.hpp"
#include "environment.hpp"
#include "metaregistry.hpp"
#include "scenarioconfig.hpp"
#include "serializer.hpp"
#include "sharedworldcomponents.hpp"

#include "ppokit/schema/project.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ppo::examples::bluedefense {
namespace {

/** @brief MAPPO training file of a BluePatrol PPOKit project (the scenario is its "settings"). */
constexpr char ProjectTrainingFile[] = "blue_patrol_training.json";

/** @brief Where a BlueDefense scenario reads its settings. */
struct BlueDefenseLocations final
{
    defenders::PatrolSettingsFiles settings;
};

/**
 * @brief Settings files and model folder of a scenario.
 * @details A BlueDefense scenario lives in a PPOKit project
 * (Examples/BlueDefense/projects/<name>): it reads the project's training file and keeps
 * its models in the project's checkpoints/. The world files (Environment.json, platforms)
 * are shared in Examples/Shared/settings. A scenario outside a project uses the
 * blue_patrol_training.json and checkpoints/ beside it.
 */
BlueDefenseLocations LocationsFor(std::filesystem::path const& scenarioPath)
{
    std::filesystem::path const absolute = std::filesystem::absolute(scenarioPath);
    std::optional<std::filesystem::path> const project = ppokit::schema::ProjectFolderOf(absolute);
    std::filesystem::path const folder = project.value_or(absolute.parent_path());
    return {defenders::PatrolSettingsIn(std::filesystem::path{PPO_SHARED_SETTINGS},
                folder / ProjectTrainingFile)};
}

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

/**
 * @brief Creates the entities and agents of a BlueDefense scenario in `environment`.
 * @param path BlueDefense scenario document (its project holds the training settings).
 * @param environment Destination ECS environment.
 * @return The blue team's controller, or null when the scenario is invalid.
 */
std::shared_ptr<defenders::GeneralizedPatrolMappoAgent> LoadScenario(
    std::filesystem::path const& path,
    ppo::model::Environment& environment)
{
    if (!environment.SetEntityInitializer(&InitializeEntity)
        || !environment.SetWorldRules(std::make_unique<BlueDefenseWorldRules>()))
        return nullptr;

    auto const scenario = ppo::config::LoadScenarioDefinition(path);
    if (scenario.m_entities.empty())
        return nullptr;

    auto const document = ppo::serialization::Serializer::Parse(path);
    if (!document.is_object() || !document.contains("blueDefense")
        || !document["blueDefense"].is_object())
        return nullptr;

    auto const& settings = document["blueDefense"];
    if (settings.contains("patrolOnly") && !settings["patrolOnly"].is_boolean())
        return nullptr;
    bool const patrolOnly = settings.value("patrolOnly", false);

    if (settings.contains("blueController")
        && (!settings["blueController"].is_string()
            || settings["blueController"].get<std::string>() != "mappo"))
        return nullptr;
    std::string spawnDiagnostic;
    auto const spawnSettings = ParseSpawnSettings(settings, &spawnDiagnostic);
    if (!spawnSettings)
    {
        std::cerr << "[BLUE_DEFENSE] Invalid spawn configuration: "
                  << spawnDiagnostic << '\n';
        return nullptr;
    }
    if (!document.contains("entities")
        || !document["entities"].is_object())
        return nullptr;

    // The initial entity spawns precede LoadSettings: read Environment.json now
    // so the first episode and later resets use the same LiDAR/world limits.
    BlueDefenseLocations const locations = LocationsFor(path);
    auto const environmentConfig =
        ppo::serialization::Serializer::Parse(locations.settings.environment);
    if (!environmentConfig.is_object()
        || !environmentConfig.contains("formatVersion")
        || !environmentConfig["formatVersion"].is_number_integer()
        || environmentConfig["formatVersion"].get<int>() != 2
        || !environmentConfig.contains("environment")
        || !environmentConfig["environment"].is_object())
        return nullptr;
    auto const& worldConfig = environmentConfig["environment"];
    if (!worldConfig.contains("maxNorth") || !worldConfig["maxNorth"].is_number()
        || !worldConfig.contains("maxEast") || !worldConfig["maxEast"].is_number())
        return nullptr;
    float const worldNorth = worldConfig["maxNorth"].get<float>();
    float const worldEast = worldConfig["maxEast"].get<float>();
    if (!std::isfinite(worldNorth) || !std::isfinite(worldEast)
        || worldNorth <= 0.F || worldEast <= 0.F)
        return nullptr;
    std::mt19937 validationRng{0U};
    WorldBounds const worldBounds{0.5F * worldNorth, 0.5F * worldEast};
    if (spawnSettings->objective.mode == ObjectiveSpawnMode::Fixed
        && !GenerateEpisodeSpawnLayout(
            1, *spawnSettings, worldBounds, validationRng))
        return nullptr;

    if (!spawnSettings->blueTeam || !spawnSettings->redTeam)
    {
        std::cerr << "[BLUE MAPPO] Missing P/N team configuration.\n";
        return nullptr;
    }
    std::size_t const configuredBlueCount = spawnSettings->blueTeam->activeCount;
    std::size_t const configuredRedCount = spawnSettings->redTeam->activeCount;
    if (patrolOnly && configuredRedCount != 0U)
        return nullptr;

    ppo::config::ScenarioEntityDefinition const* blueTemplate = nullptr;
    ppo::config::ScenarioEntityDefinition const* redTemplate = nullptr;
    for (auto const& item : scenario.m_entities)
    {
        if (item.m_type != "Mobile")
            continue;
        if (item.m_side == ppo::model::eEntitySide::Friend)
        {
            if (!IsGeneralizedBlueTemplate(item)
                || (blueTemplate != nullptr
                    && !SameGeneralizedBlueTemplate(*blueTemplate, item)))
                return nullptr;
            blueTemplate = &item;
        }
        else if (item.m_side == ppo::model::eEntitySide::Hostile)
        {
            if (!IsProductionRedTemplate(item)
                || (redTemplate != nullptr
                    && !SameProductionRedTemplate(*redTemplate, item)))
                return nullptr;
            redTemplate = &item;
        }
    }
    if (!blueTemplate || (configuredRedCount > 0U && !redTemplate))
        return nullptr;

    std::map<std::string, std::shared_ptr<ppo::model::Entity>> entities;
    std::vector<std::shared_ptr<ppo::model::IAgent>> agents;

    for (auto const& item : scenario.m_entities)
    {
        if (item.m_type == "Mobile")
            continue;
        bool const objective = item.m_name == "blue_objective"
            && item.m_type == "Objective"
            && item.m_side == ppo::model::eEntitySide::Friend;
        bool const zone = item.m_name == "blue_restricted_zone"
            && item.m_type == "RestrictedZone"
            && item.m_side == ppo::model::eEntitySide::Friend;
        if ((!objective && !zone)
            || !document["entities"].contains(item.m_name)
            || !item.m_agents.empty() || !item.m_target.empty())
            return nullptr;
        auto entity = ppo::reflection::CreateEntity(
            item.m_type, environment, item.m_platform, item.m_side);
        if (!entity)
            return nullptr;
        entity->add<ppo::model::sEntityMetadata>(
            ppo::model::sEntityMetadata{item.m_name, item.m_platform});
        if (!entities.emplace(item.m_name, std::move(entity)).second)
            return nullptr;
    }

    if (blueTemplate != nullptr)
    {
        for (std::size_t index = 0U; index < configuredBlueCount; ++index)
        {
            std::string const name = "blue_" + std::to_string(index + 1U);
            auto entity = ppo::reflection::CreateEntity(
                blueTemplate->m_type, environment, blueTemplate->m_platform,
                blueTemplate->m_side);
            if (!entity)
                return nullptr;
            entity->add<ppo::model::sEntityMetadata>(
                ppo::model::sEntityMetadata{name, blueTemplate->m_platform});
            auto& sensor = entity->add<ppo::model::sPseudoLidarState>();
            sensor.m_nearestCollisionRanges.fill(1.F);
            if (!entities.emplace(name, std::move(entity)).second)
                return nullptr;
        }
    }

    if (configuredRedCount > 0U && redTemplate != nullptr)
    {
        for (std::size_t index = 0U; index < configuredRedCount; ++index)
        {
            std::string const name = "red_" + std::to_string(index + 1U);
            auto entity = ppo::reflection::CreateEntity(
                redTemplate->m_type, environment, redTemplate->m_platform,
                redTemplate->m_side);
            if (!entity)
                return nullptr;
            entity->add<ppo::model::sEntityMetadata>(
                ppo::model::sEntityMetadata{name, redTemplate->m_platform});
            auto& sensor = entity->add<ppo::model::sPseudoLidarState>();
            sensor.m_nearestCollisionRanges.fill(1.F);

            // The validated template owns exactly one production RLMission
            // declaration. Every generated red receives its own controller.
            auto agent = ppo::reflection::CreateAgent(
                redTemplate->m_agents.front(), environment, *entity);
            if (!agent)
                return nullptr;
            agents.push_back(std::move(agent));

            if (!entities.emplace(name, std::move(entity)).second)
                return nullptr;
        }
    }

    // Restricted zones are discovered from their dedicated typed collection;
    // no scenario identifier is hard-coded in the loader.
    for (auto const& zone : spawnSettings->restrictedZones.items)
    {
        if (entities.count(zone.id) != 0U)
            return nullptr;
        auto entity = ppo::reflection::CreateEntity(
            "RestrictedZone", environment, zone.platform,
            ppo::model::eEntitySide::Hostile);
        if (!entity)
            return nullptr;
        entity->add<ppo::model::sEntityMetadata>(
            ppo::model::sEntityMetadata{zone.id, zone.platform});
        if (!entities.emplace(zone.id, std::move(entity)).second)
            return nullptr;
    }

    if (!entities.count("blue_objective")
        || !entities.count("blue_restricted_zone")
        || entities.size() != configuredBlueCount + configuredRedCount + 2U
            + spawnSettings->restrictedZones.items.size())
        return nullptr;
    auto const objectiveId = entities.at("blue_objective")->uniqid();
    for (std::size_t index = 0U; index < configuredRedCount; ++index)
    {
        auto& red = *entities.at("red_" + std::to_string(index + 1U));
        if (!red.has<ppo::model::sSNAState>())
            return nullptr;
        auto [sna] = red.get<ppo::model::sSNAState>();
        sna.m_target = objectiveId;
    }

    auto* registry = entities.begin()->second->registry();
    if (!registry)
        return nullptr;

    defenders::Parameters patrolParameters{};
    defenders::PatrolGeometrySettings patrolGeometrySettings{};
    defenders::PatrolTrainingSettings patrolTraining{};
    try
    {
        patrolParameters = defenders::LoadPatrolParameters(locations.settings);
        patrolGeometrySettings =
            defenders::LoadPatrolGeometrySettings(locations.settings);
        patrolTraining = defenders::LoadPatrolTrainingSettings(locations.settings);
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO] Configuration failed: "
                  << error.what() << '\n';
        return nullptr;
    }

    auto& context = registry->ctx().emplace<InitializerContext>();
    context.rng.seed(patrolTraining.seed);
    context.count = entities.size();
    context.spawnSettings = *spawnSettings;
    if (context.spawnSettings.objective.mode == ObjectiveSpawnMode::Randomized)
    {
        for (auto& zone : context.spawnSettings.restrictedZones.items)
        {
            auto const& entity = entities.at(zone.id);
            if (!entity->has<ppo::model::sSNAState>())
                return nullptr;
            auto const [sna] = entity->get<ppo::model::sSNAState>();
            zone.lethalRadiusMeters = sna.m_maxNoescapeZone;
            if (context.spawnSettings.restrictedZones.enabled
                && zone.enabled && zone.enabledProbability > 0.F
                && (!std::isfinite(zone.lethalRadiusMeters)
                    || zone.lethalRadiusMeters <= 0.F))
            {
                std::cerr << "[BLUE_DEFENSE] Invalid physical radius for "
                          << zone.id << '\n';
                return nullptr;
            }
        }
    }
    context.patrolOnly = patrolOnly;
    context.patrolCruiseSpeed = patrolParameters.cruiseSpeed;
    context.patrolMaximumRotationRate = patrolParameters.maxRotationRate;
    context.patrolCollisionDistance = patrolParameters.collisionDistance;
    context.patrolMinimumSeparationMeters = patrolTraining.noThreatPatrol.minimumSeparationMeters;
    context.patrolMaximumSpeed = patrolParameters.maxSpeed;
    context.patrolGeometrySettings = patrolGeometrySettings;
    auto& world = registry->ctx().get<ppo::model::sEnvironmentState>();
    world.m_maxNorth = worldNorth;
    world.m_maxEast = worldEast;

    // The world starts in its INIT state, which does not re-initialize: initialize the
    // first episode here; later resets use the registered initializer.
    for (auto const& [name, entity] : entities)
    {
        (void)name;
        entity->Init();
    }
    if (context.patrolGeometrySamplingFailed
        || context.episodeLayoutGenerationFailed
        || !context.patrolGeometryReady)
        return nullptr;
    patrolParameters.geometry = context.patrolGeometry;
    std::shared_ptr<defenders::GeneralizedPatrolMappoAgent> blueController;
    try
    {
            std::array<ppo::model::BaseEntity*,
                       defenders::TeamCapacity::MaximumBlueMobiles> guards{};
            std::array<ppo::model::BaseEntity*,
                       defenders::TeamCapacity::MaximumRedMobiles> attackers{};
            for (std::size_t index = 0U; index < configuredBlueCount; ++index)
                guards[index] = entities.at(
                    "blue_" + std::to_string(index + 1U)).get();
            std::size_t const redCount = configuredRedCount;
            for (std::size_t index = 0U; index < redCount; ++index)
                attackers[index] = entities.at(
                    "red_" + std::to_string(index + 1U)).get();

            blueController = environment.CreateAgent<
                defenders::GeneralizedPatrolMappoAgent>(
                    guards, configuredBlueCount, attackers, redCount,
                    *entities.at("blue_objective"), patrolParameters,
                    patrolTraining);
            if (!blueController)
                return nullptr;
            agents.push_back(blueController);
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO] Configuration failed: "
                  << error.what() << '\n';
        return nullptr;
    }
    for (auto const& agent : agents)
        agent->Init();

    return blueController;
}

} // namespace

std::shared_ptr<defenders::GeneralizedPatrolMappoAgent> LoadExternalScenario(
    std::filesystem::path const& path, ppo::model::Environment& environment)
{
    auto controller = LoadScenario(path, environment);
    if (controller)
    {
        // The time step and limits of the world: the shared Environment.json.
        environment.LoadSettings(LocationsFor(path).settings.environment);
    }
    return controller;
}

} // namespace ppo::examples::bluedefense
