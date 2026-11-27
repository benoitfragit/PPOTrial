/**
 * @file Examples/Airspace/Environment/evaluation_episodes.hpp
 * @brief The evaluator's cases as evaluation episodes of an environment DLL (ppokit-run's
 *        suites): the evaluated agent plays the caller's actions.
 * @details The campaign (evaluation_campaign.hpp) loads Scenarios/Evaluation/<agent>.json;
 * its evaluated entity's agent is the production agent of the family (Agent) playing an
 * ExternalPolicy, so ppokit-run's policy acts exactly where the evaluator's "eval" policy
 * acted. Episode `n` of a suite is case `n` of the cases file (a suite's seeds are case
 * numbers). The episode ends as the evaluator's case: the evaluated entity destroyed or its
 * primary target destroyed (terminated), or the step limit (truncated, a timeout). There is
 * no reward in an evaluation episode; its metrics are the case's results and its score,
 * whose mean over the cases is the evaluator's campaign score.
 */
#pragma once

#include "evaluation_campaign.hpp"
#include "external_training.hpp"
#include "tactical_world.hpp"

#include "ppokit/runtime/rollout.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace ppo::examples::airspace {

/** @brief task.json "metrics" of the Airspace agents, in this order. */
constexpr std::size_t EvaluationMetricCount = 6;

/**
 * @brief Evaluation episodes of one agent family.
 * @tparam Agent Production agent with a (BaseEntity&, unique_ptr<IPolicy>) constructor and State().
 */
template<typename Agent>
class EvaluationEpisodes final
{
public:
    /**
     * @param family Agent family (RLMission, RLSNA): Scenarios/Evaluation/<family>.json.
     * @param actions Size of an action.
     */
    EvaluationEpisodes(std::string family, std::size_t actions) : family_(std::move(family)), actions_(actions) {}

    /** @brief Cases of the campaign (loads it on first use). */
    std::size_t CaseCount()
    {
        Load();
        return static_cast<std::size_t>(campaign_.CaseCount());
    }

    /** @brief Starts case `index`; writes the evaluated agent's first observation. */
    void Start(std::size_t index, float* observation)
    {
        Load();
        if (index >= static_cast<std::size_t>(campaign_.CaseCount()) || !campaign_.ResetCase(index))
        {
            throw std::runtime_error(family_ + ": no evaluation case " + std::to_string(index) + " (cases 0 to "
                + std::to_string(campaign_.CaseCount() - 1) + ")");
        }
        result_ = ppo::evaluation::CaseResult{};
        steps_ = 0;
        WriteObservation(observation);
    }

    /** @brief One step of the case with the caller's action. */
    ppokit::runtime::StepEnd Step(float const* action, float* observation)
    {
        std::copy(action, action + actions_, policy_->Action().begin());
        ppo::evaluation::CaseStep const step = campaign_.Step();
        result_.Add(step);
        ++steps_;
        WriteObservation(observation);
        if (step.Ends())
        {
            result_.Finish(steps_ - 1, campaign_.StepLimit());
            return ppokit::runtime::StepEnd::Terminated;
        }
        if (steps_ >= campaign_.StepLimit())
        {
            result_.Finish(campaign_.StepLimit(), campaign_.StepLimit());
            return ppokit::runtime::StepEnd::Truncated;
        }
        return ppokit::runtime::StepEnd::Running;
    }

    /** @brief The tactical world row of the current case, seen from the evaluated entity. */
    void WriteWorld(float* world) const noexcept
    {
        tactical::WriteWorld(campaign_.World(), campaign_.Evaluated()->uniqid(), world);
    }

    /** @brief The last case's metrics, in task.json order. */
    void Metrics(double* metrics) const noexcept
    {
        metrics[0] = result_.Score();
        metrics[1] = result_.targetDestroyed;
        metrics[2] = result_.playerDestroyed;
        metrics[3] = result_.primaryPk;
        metrics[4] = result_.secondaryDestroyed;
        metrics[5] = result_.timeout ? 1.0 : 0.0;
    }

private:
    /** @brief Loads the campaign once, with the external evaluated agent. */
    void Load()
    {
        if (agent_ != nullptr)
        {
            return;
        }
        bool const loaded = campaign_.Load(
            family_, [this](ppo::model::BaseEntity& entity) { return CreateEvaluatedAgent(entity); });
        if (!loaded || agent_ == nullptr)
        {
            agent_ = nullptr;
            throw std::runtime_error(family_ + ": the evaluation scenario does not load (see the messages above)");
        }
    }

    /** @brief The family's production agent, playing the caller's actions. */
    std::shared_ptr<ppo::model::IAgent> CreateEvaluatedAgent(ppo::model::BaseEntity& entity)
    {
        auto policy = std::make_unique<ppo::model::ExternalPolicy>(actions_);
        policy_ = policy.get();
        auto agent = std::make_shared<Agent>(entity, std::move(policy));
        agent_ = agent.get();
        return agent;
    }

    void WriteObservation(float* observation) const
    {
        auto const& state = agent_->State();
        std::copy(state.begin(), state.end(), observation);
    }

    std::string family_;
    std::size_t actions_;
    ppo::evaluation::EvaluationCampaign campaign_{};
    Agent* agent_ = nullptr;
    ppo::model::ExternalPolicy* policy_ = nullptr;
    ppo::evaluation::CaseResult result_{};
    int steps_ = 0;
};

} // namespace ppo::examples::airspace
