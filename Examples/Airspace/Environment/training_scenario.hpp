/**
 * @file Examples/Airspace/Environment/training_scenario.hpp
 * @brief The training scenario of an Airspace agent, stepped by an environment DLL: the world
 *        of the project's scenario, its agent in training mode, its reward channels and its
 *        step limit. Shared by the RLMission and RLSNA environments.
 * @details reset / begin_step: the world begins its step (rules before the step, the
 * re-initialization after an ended episode, which re-initializes the agent and gives its
 * first observation). step: the action is handed to the agent; the world ends its step
 * (agents choose, entities advance, rules resolve, the agent's reward function runs and it
 * reads the next observation).
 *
 * The reward is the agent's: its weighted channels summed and clamped (the rewardClamp of
 * its Training/<agent>.json); the channels are the weighted ones. An episode ends when the
 * reward function ends it (terminal when it set done), or truncated after the agent's
 * rollout size of steps. The reward function never pays its last-step mission failure
 * (IsLastStep stays false, as under ppokit-run before the in-world modes were removed).
 *
 * The world's random initializer draws the episodes from the seed given to the environment.
 */
#pragma once

#include "airspaceexample.hpp"
#include "environment.hpp"
#include "tactical_world.hpp"

#include "ppokit/runtime/rollout.hpp"
#include "ppokit/schema/project.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace ppo::examples::airspace {

/** @brief The scenario of a project (project.json "settings"), or nothing when it names none. */
inline std::optional<std::filesystem::path> TrainingScenarioOf(std::filesystem::path const& project)
{
    std::optional<std::filesystem::path> const settings = ppokit::schema::LoadProject(project).SettingsFile();
    if (!settings)
    {
        return std::nullopt;
    }
    return std::filesystem::absolute(*settings);
}

/** @brief A scenario whose agent in training mode is driven by the environment's caller. */
class TrainingScenario final
{
public:
    /**
     * @param scenario The scenario file (one agent in training mode).
     * @param seed Seed of the episodes: the draws of the world's random initializer.
     * @param observationSize, actionSize, channels The task's sizes, which the agent must have.
     * @throws std::runtime_error for a scenario without a fitting agent in training mode.
     */
    TrainingScenario(std::filesystem::path const& scenario, std::uint64_t seed, std::size_t observationSize,
        std::size_t actionSize, std::size_t channels)
    {
        world_.SeedInitialization(seed);
        ExternalScenario loaded = LoadExternalScenario(scenario, world_);
        agent_ = std::move(loaded.agent);
        controlled_ = loaded.entity;
        ppo::model::ExternalStep const& exchange = agent_->External();
        if (exchange.observation.size() != observationSize || exchange.action.size() != actionSize
            || exchange.channels.size() != channels)
        {
            throw std::runtime_error(scenario.string() + ": its training agent does not have the task's sizes");
        }
    }

    /**
     * @brief Starts an episode. The first one, and one cut by the step limit (still running
     *        in the world), start with the world's re-initialization: entities drawn by the
     *        random initializer, agents initialized. Without it the first step would run on
     *        the scenario's raw state (the world begins in its INIT state).
     */
    void Reset(float* observations, std::uint8_t* available, float* world)
    {
        if (episodeOpen_ || !started_)
        {
            world_.RequestReinitialization();
            episodeOpen_ = false;
            started_ = true;
        }
        steps_ = 0;
        BeginStep(observations, available, world);
    }

    /** @brief The world begins its step; the agent's observation and the world row. */
    void BeginStep(float* observations, std::uint8_t* available, float* world)
    {
        world_.BeginStep();
        Publish(observations, available, world);
    }

    /** @brief One step of the agent: its reward and how the episode goes on. */
    ppokit::runtime::StepEnd Step(float const* actions, float* observations, std::uint8_t* available, float* world,
        double* rewards, double* rewardChannels)
    {
        ppo::model::ExternalStep& exchange = agent_->External();
        std::copy(actions, actions + exchange.action.size(), exchange.action.begin());
        bool const worldEnded = world_.EndStep();
        ++steps_;
        episodeOpen_ = true;
        Publish(observations, available, world);
        rewards[0] = exchange.reward;
        std::copy(exchange.channels.begin(), exchange.channels.end(), rewardChannels);
        if (exchange.terminated)
        {
            episodeOpen_ = false; // The world re-initializes at the next begin_step.
            return ppokit::runtime::StepEnd::Terminated;
        }
        if (exchange.ended || worldEnded)
        {
            episodeOpen_ = false;
            return ppokit::runtime::StepEnd::Truncated;
        }
        if (steps_ >= agent_->RolloutSteps())
        {
            return ppokit::runtime::StepEnd::Truncated; // Reset re-initializes the world.
        }
        return ppokit::runtime::StepEnd::Running;
    }

    void SetWeights(double const* weights) { agent_->SetRewardWeights(weights); }

    void Weights(double* weights) const { agent_->RewardWeights(weights); }

private:
    /** @brief The agent's current observation (always present) and the world row. */
    void Publish(float* observations, std::uint8_t* available, float* world) const
    {
        ppo::model::ExternalStep const& exchange = agent_->External();
        std::copy(exchange.observation.begin(), exchange.observation.end(), observations);
        available[0] = 1;
        tactical::WriteWorld(world_.SteppedWorld(), controlled_, world);
    }

    ppo::model::Environment world_;
    std::shared_ptr<ppo::model::IExternalTrainingAgent> agent_;
    /** @brief The training agent's entity (its role in the world row). */
    entt::entity controlled_ = entt::null;
    /** @brief Steps of the current episode (step limit: the agent's rollout size). */
    std::size_t steps_ = 0;
    /** @brief A step was taken since the episode started and did not end it in the world. */
    bool episodeOpen_ = false;
    /** @brief A training episode was started (the first one re-initializes the world). */
    bool started_ = false;
};

} // namespace ppo::examples::airspace
