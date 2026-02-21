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

    std::filesystem::path const fromScenario = sourcePath.parent_path() / candidate;
    if (std::filesystem::exists(fromScenario))
        return fromScenario.lexically_normal();

    std::filesystem::path const fromResources = std::filesystem::path{ RESOURCE_PATH } / candidate;
    return fromResources.lexically_normal();
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
    if (!content.is_object() || !content.contains("Scenario"))
    {
        LogScenarioError("missing Scenario section", resolvedPath);
        return definition;
    }

    auto const& scenarioSection = content["Scenario"];
    if (!scenarioSection.contains("entities") || !scenarioSection["entities"].is_object())
    {
        LogScenarioError("missing entities object", resolvedPath);
        return definition;
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
                if (agent.is_string())
                    entity.m_agents.push_back(agent.get<std::string>());
            }
        }

        if (!entity.m_name.empty() && !entity.m_type.empty())
            definition.m_entities.push_back(std::move(entity));
    }

    if (content.contains("Evaluation") && content["Evaluation"].is_object())
    {
        auto const& evaluationSection = content["Evaluation"];
        definition.m_evaluation.m_enabled = true;
        definition.m_evaluation.m_evaluatedEntity = evaluationSection.value("evaluatedEntity", "");
        definition.m_evaluation.m_nbTests = evaluationSection.value("nbTests", 0);
        definition.m_evaluation.m_nbSteps = evaluationSection.value("nbSteps", 0);
        definition.m_evaluation.m_timeStep = evaluationSection.value("timeStep", definition.m_evaluation.m_timeStep);
        definition.m_evaluation.m_maxNorth = evaluationSection.value("maxNorth", definition.m_evaluation.m_maxNorth);
        definition.m_evaluation.m_maxEast = evaluationSection.value("maxEast", definition.m_evaluation.m_maxEast);

        if (evaluationSection.contains("checkpointFile") && evaluationSection["checkpointFile"].is_string())
        {
            definition.m_evaluation.m_checkpointFile = ResolveRelativePath(
                resolvedPath,
                std::filesystem::path{ evaluationSection["checkpointFile"].get<std::string>() });
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
    || !content.contains("EvaluationBuffer")
    || !content["EvaluationBuffer"].is_object())
    {
        LogScenarioError("missing EvaluationBuffer section", resolvedPath);
        return definition;
    }

    auto const& bufferSection = content["EvaluationBuffer"];
    if (!bufferSection.contains("tests") || !bufferSection["tests"].is_array())
    {
        LogScenarioError("missing tests array", resolvedPath);
        return definition;
    }

    for (auto const& testSection : bufferSection["tests"])
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
