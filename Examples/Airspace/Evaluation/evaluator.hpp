/**
* @file Evaluation/data/evaluator.hpp
* @brief Declarations for evaluator.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include "entity.hpp"
#include "iagent.hpp"

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace ppo::evaluation {

/**
* @brief Factory signature used to create evaluation agents from scenario data.
*/
using AgentFactory = std::function<std::shared_ptr<ppo::model::IAgent>(
    ppo::model::BaseEntity&,
    std::string_view)>;

/**
* @brief Registers one evaluation-time agent factory by name.
* @param name Agent type name.
* @param factory Factory callback creating the agent.
*/
void RegisterAgentFactory(std::string_view name, AgentFactory factory);

/**
* @brief Shared evaluator implementation used by training agents.
*/
class Evaluator :
    public ppo::model::IEvaluator
{
public:
    /**
    * @brief Creates an idle evaluator instance.
    */
    Evaluator(void);

    /**
    * @brief Stops and destroys the evaluator worker thread.
    */
    ~Evaluator(void);

    Evaluator(Evaluator&) = delete;
    Evaluator(Evaluator&&) = delete;

    Evaluator& operator=(Evaluator&) = delete;
    Evaluator& operator=(Evaluator&&) = delete;

    /**
    * @brief Starts one asynchronous evaluation campaign.
    * @param name Agent family name.
    * @param modelpath Root model folder containing eval/prod weights.
    */
    void Start(std::string const& name, std::filesystem::path const& modelpath) final;
    /**
    * @brief Stops the current evaluation campaign if one is running.
    */
    void Stop(void) final;

    /**
    * @brief Returns the latest evaluator snapshot.
    * @return Immutable evaluator state snapshot.
    */
    ppo::model::sEvaluatorState GetSnapshot(void) const noexcept final;

private:
    /**
    * @brief Hidden implementation storing evaluator runtime state.
    */
    struct Impl;
    /**
    * @brief Pointer to the evaluator implementation.
    */
    std::unique_ptr<Impl> p_impl{ nullptr };
};

} /* ppo::evaluation */
