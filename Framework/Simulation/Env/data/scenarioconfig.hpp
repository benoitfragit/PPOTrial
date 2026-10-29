/**
* @file Env/data/scenarioconfig.hpp
* @brief Declarations for scenarioconfig.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "entitytypes.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace ppo::config {

/**
* @brief Declarative description of one entity in a scenario file.
*/
struct ScenarioEntityDefinition
{
    /**
    * @brief Unique entity name used inside the scenario.
    */
    std::string m_name{};
    /**
    * @brief Logical entity type name.
    */
    std::string m_type{};
    /**
    * @brief Platform file name used for instantiation.
    */
    std::string m_platform{};
    /**
    * @brief Team side assigned to the entity.
    */
    ppo::model::eEntitySide m_side{ ppo::model::eEntitySide::Unknown };
    /**
    * @brief Name of the primary target entity.
    */
    std::string m_target{};
    /**
    * @brief Agent type names attached to the entity.
    */
    std::vector<std::string> m_agents{};
};

/**
* @brief Optional evaluation settings embedded in a scenario file.
*/
struct EvaluationDefinition
{
    /**
    * @brief True when the scenario contains evaluation settings.
    */
    bool m_enabled{ false };
    /**
    * @brief Name of the entity whose policy must be evaluated.
    */
    std::string m_evaluatedEntity{};
    /**
    * @brief Path to the evaluation cases file.
    */
    std::filesystem::path m_casesFile{};
    /**
    * @brief Number of evaluation tests to execute.
    */
    int m_nbTests{ 0 };
    /**
    * @brief Number of steps executed per evaluation test.
    */
    int m_nbSteps{ 0 };
    /**
    * @brief Evaluation time step in seconds.
    */
    float m_timeStep{ 0.2F };
    /**
    * @brief North-axis evaluation boundary.
    */
    float m_maxNorth{ 30000.F };
    /**
    * @brief East-axis evaluation boundary.
    */
    float m_maxEast{ 40000.F };
};

/**
* @brief Fully parsed scenario definition.
*/
struct ScenarioDefinition
{
    /**
    * @brief Source file path used to build the definition.
    */
    std::filesystem::path m_sourcePath{};
    /**
    * @brief Declared entity list.
    */
    std::vector<ScenarioEntityDefinition> m_entities{};
    /**
    * @brief Optional embedded evaluation configuration.
    */
    EvaluationDefinition m_evaluation{};
};

/**
* @brief Initial state override for one entity inside one evaluation test.
*/
struct EvaluationEntityState
{
    /**
    * @brief Optional north position in meters.
    */
    std::optional<float> m_north{};
    /**
    * @brief Optional east position in meters.
    */
    std::optional<float> m_east{};
    /**
    * @brief Optional heading in radians.
    */
    std::optional<float> m_heading{};
    /**
    * @brief Optional scalar speed in meters per second.
    */
    std::optional<float> m_speed{};
    /**
    * @brief Optional turn acceleration.
    */
    std::optional<float> m_accpsi{};
    /**
    * @brief Optional longitudinal acceleration.
    */
    std::optional<float> m_acc{};
};

/**
* @brief One named evaluation test containing states for multiple entities.
*/
struct EvaluationTestDefinition
{
    /**
    * @brief Human-readable test name.
    */
    std::string m_name{};
    /**
    * @brief Per-entity initial states keyed by entity name.
    */
    std::unordered_map<std::string, EvaluationEntityState> m_entities{};
};

/**
* @brief Parsed evaluation buffer file.
*/
struct EvaluationBufferDefinition
{
    /**
    * @brief Source file path used to build the definition.
    */
    std::filesystem::path m_sourcePath{};
    /**
    * @brief Ordered evaluation test list.
    */
    std::vector<EvaluationTestDefinition> m_tests{};
};

/**
* @brief Resolves one resource-relative path against the project resource root.
* @param path Resource-relative or absolute path.
* @return Resolved filesystem path.
*/
std::filesystem::path ResolveResourcePath(std::filesystem::path const& path);

/**
* @brief Loads and validates one scenario definition file.
* @param scenarioPath Scenario JSON path.
* @return Parsed scenario definition.
*/
ScenarioDefinition LoadScenarioDefinition(std::filesystem::path const& scenarioPath);

/**
* @brief Loads and validates one evaluation buffer definition file.
* @param bufferPath Evaluation buffer JSON path.
* @return Parsed evaluation buffer definition.
*/
EvaluationBufferDefinition LoadEvaluationBufferDefinition(std::filesystem::path const& bufferPath);

} /* ppo::config */
