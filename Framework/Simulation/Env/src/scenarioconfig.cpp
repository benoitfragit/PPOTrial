/**
* @file Env/src/scenarioconfig.cpp
* @brief Implementation for scenarioconfig.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "scenarioconfig.hpp"

#include "serializer.hpp"

#include <iostream>

namespace ppo::config {

namespace {

/**
* @brief Converts one scenario-side string into the runtime enumeration.
*/
ppo::model::eEntitySide
ParseEntitySide(std::string const& name) noexcept
{
    if (name == "Hostile")
        return ppo::model::eEntitySide::Hostile;

    if (name == "Friend")
        return ppo::model::eEntitySide::Friend;

    return ppo::model::eEntitySide::Unknown;
}

/**
* @brief Resolves a relative path from either the source file or the resource root.
*/
std::filesystem::path
ResolveRelativePath(std::filesystem::path const& sourcePath, std::filesystem::path const& candidate)
{
    if (candidate.is_absolute())
        return candidate;

    // All V2 references are resolved relative to their declaring file.
    return (sourcePath.parent_path() / candidate).lexically_normal();
}

/**
* @brief Emits one scenario parsing error message.
*/
void
LogScenarioError(std::string const& message, std::filesystem::path const& path)
{
    std::cerr << "[SCENARIO_CONFIG] file='" << path.string() << "' message='" << message << "'" << std::endl;
}

} // namespace

/**
* @brief Resolves one resource-relative path against the project resource root.
*/
std::filesystem::path
ResolveResourcePath(std::filesystem::path const& path)
{
    if (path.is_absolute())
        return path;

    return (std::filesystem::path{ RESOURCE_PATH } / path).lexically_normal();
}

/**
* @brief Loads one scenario JSON file into a typed runtime definition.
*/
ScenarioDefinition
LoadScenarioDefinition(std::filesystem::path const& scenarioPath)
{
    ScenarioDefinition definition{};
    definition.m_sourcePath = scenarioPath;

    auto const resolvedPath = ResolveResourcePath(scenarioPath);
    auto content = ppo::serialization::Serializer::Parse(resolvedPath);
    if (!content.is_object())
    {
        LogScenarioError("scenario must contain an object", resolvedPath);
        return definition;
    }

    if (!content.contains("formatVersion")
        || !content["formatVersion"].is_number_integer()
        || content["formatVersion"].get<int>() != 2)
    {
        LogScenarioError("scenario requires formatVersion = 2", resolvedPath);
        return definition;
    }

    auto const& scenarioSection = content;
    if (!scenarioSection.contains("entities") || !scenarioSection["entities"].is_object())
    {
        LogScenarioError("missing entities object", resolvedPath);
        return definition;
    }

    // V2 scenario files can use one logical agent profile in different modes.
    // The catalog is owned by the application, not by GenericRLCore, and
    // resolves to the same registered factories used by the runtime.
    ppo::serialization::json agentProfiles{};
    if (content.contains("agentProfiles") && content["agentProfiles"].is_string())
    {
        auto const catalogFile = ResolveRelativePath(
            resolvedPath, content["agentProfiles"].get<std::string>());
        auto const catalog = ppo::serialization::Serializer::Parse(catalogFile);
        if (!catalog.is_object()
        || !catalog.contains("formatVersion")
        || !catalog["formatVersion"].is_number_integer()
        || catalog["formatVersion"].get<int>() != 2
        || !catalog.contains("profiles")
        || !catalog["profiles"].is_object())
        {
            LogScenarioError("cannot load agent profile catalog", catalogFile);
            return definition;
        }
        agentProfiles = catalog["profiles"];
    }

    for (auto const& [name, entitySection] : scenarioSection["entities"].items())
    {
        if (!entitySection.is_object())
            continue;

        ScenarioEntityDefinition entity{};
        entity.m_name = name;
        entity.m_type = entitySection.value("type", "");
        entity.m_platform = entitySection.value("platform", "");
        entity.m_side = ParseEntitySide(entitySection.value("side", ""));
        entity.m_target = entitySection.value("target", "");

        if (entitySection.contains("agents") && entitySection["agents"].is_array())
        {
            for (auto const& agent : entitySection["agents"])
            {
                if (!agent.is_object()
                || !agent.contains("profile") || !agent["profile"].is_string()
                || !agent.contains("mode") || !agent["mode"].is_string())
                {
                    LogScenarioError("invalid agent declaration for entity '" + name + "'", resolvedPath);
                    return ScenarioDefinition{};
                }

                auto const profileName = agent["profile"].get<std::string>();
                auto const modeName = agent["mode"].get<std::string>();
                if (!agentProfiles.is_object()
                || !agentProfiles.contains(profileName)
                || !agentProfiles[profileName].is_object()
                || !agentProfiles[profileName].contains(modeName)
                || !agentProfiles[profileName][modeName].is_string())
                {
                    LogScenarioError("unresolved agent profile '" + profileName
                                     + "' mode '" + modeName + "' for entity '" + name + "'", resolvedPath);
                    return ScenarioDefinition{};
                }

                entity.m_agents.push_back(
                    agentProfiles[profileName][modeName].get<std::string>());
            }
        }

        if (!entity.m_name.empty() && !entity.m_type.empty())
            definition.m_entities.push_back(std::move(entity));
    }

    if (content.contains("evaluation") && content["evaluation"].is_object())
    {
        auto const& evaluationSection = content["evaluation"];
        definition.m_evaluation.m_enabled = true;
        definition.m_evaluation.m_evaluatedEntity = evaluationSection.value("evaluatedEntity", "");
        definition.m_evaluation.m_nbTests = evaluationSection.value("nbTests", 0);
        definition.m_evaluation.m_nbSteps = evaluationSection.value("nbSteps", 0);
        definition.m_evaluation.m_timeStep = evaluationSection.value("timeStep", definition.m_evaluation.m_timeStep);
        definition.m_evaluation.m_maxNorth = evaluationSection.value("maxNorth", definition.m_evaluation.m_maxNorth);
        definition.m_evaluation.m_maxEast = evaluationSection.value("maxEast", definition.m_evaluation.m_maxEast);

        if (evaluationSection.contains("casesFile") && evaluationSection["casesFile"].is_string())
        {
            definition.m_evaluation.m_casesFile = ResolveRelativePath(
                resolvedPath,
                std::filesystem::path{ evaluationSection["casesFile"].get<std::string>() });
        }
    }

    definition.m_sourcePath = resolvedPath;
    return definition;
}

/**
* @brief Loads one evaluation buffer JSON file into typed test cases.
*/
EvaluationBufferDefinition
LoadEvaluationBufferDefinition(std::filesystem::path const& bufferPath)
{
    EvaluationBufferDefinition definition{};
    definition.m_sourcePath = bufferPath;

    auto const resolvedPath = ResolveResourcePath(bufferPath);
    auto content = ppo::serialization::Serializer::Parse(resolvedPath);
    if (!content.is_object()
    || !content.contains("formatVersion")
    || !content["formatVersion"].is_number_integer()
    || content["formatVersion"].get<int>() != 2)
    {
        LogScenarioError("evaluation cases require formatVersion = 2", resolvedPath);
        return definition;
    }

    if (!content.contains("tests") || !content["tests"].is_array())
    {
        LogScenarioError("missing tests array", resolvedPath);
        return definition;
    }

    for (auto const& testSection : content["tests"])
    {
        if (!testSection.is_object() || !testSection.contains("entities") || !testSection["entities"].is_object())
            continue;

        EvaluationTestDefinition test{};
        test.m_name = testSection.value("name", "");

        for (auto const& [entityName, stateSection] : testSection["entities"].items())
        {
            if (!stateSection.is_object())
                continue;

            EvaluationEntityState state{};
            if (stateSection.contains("north"))   state.m_north = stateSection["north"].get<float>();
            if (stateSection.contains("east"))    state.m_east = stateSection["east"].get<float>();
            if (stateSection.contains("heading")) state.m_heading = stateSection["heading"].get<float>();
            if (stateSection.contains("speed"))   state.m_speed = stateSection["speed"].get<float>();
            if (stateSection.contains("accpsi"))  state.m_accpsi = stateSection["accpsi"].get<float>();
            if (stateSection.contains("acc"))     state.m_acc = stateSection["acc"].get<float>();

            test.m_entities.emplace(entityName, std::move(state));
        }

        definition.m_tests.push_back(std::move(test));
    }

    definition.m_sourcePath = resolvedPath;
    return definition;
}

} /* ppo::config */
