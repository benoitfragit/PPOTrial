/**
 * @file PPOKit/include/ppokit/runtime/hosted_trainer.hpp
 * @brief Hosted MAPPO: the client's simulation drives the loop and calls PPOKit each step.
 * @details In the Owned mode PPOKit calls the task (Reset, Step). Here the client keeps its
 * own loop (an ECS, a renderer, synchronized agents) and calls the trainer:
 *
 * @code
 * trainer.ChooseActions(observations, critics, presence, actions); // whole team
 * // ... the simulation advances one step ...
 * bool const updated = trainer.Observe(nextCritics, rewards, ends);  // update when full
 * @endcode
 *
 * The trainer owns the rollout (TeamRollout) and the PPO algorithm: critic baselines on
 * the full rollout, per-agent GAE, advantage normalization over the available agents,
 * then epochs of shuffled minibatches; each trains the actor on its available rows (mean
 * over them) and the critic on all its member rows, with one optimizer step per network.
 * Rows of agents absent from the episode (AgentPresence::Absent) are skipped. The
 * centralized critic input is the client's: it may see the whole team.
 *
 * The networks stay the client's too, behind the Models interface (rl_tools, PPOKit
 * graphs...). Models must provide, for float rows of the sizes they declare:
 *
 * @code
 * std::size_t ObservationSize() const, CriticSize() const, ActionSize() const;
 * float LogStd(std::size_t action) const;                         // Gaussian heads
 * void ActorMeans(float const* const* observations, std::size_t rows, float* means);
 * void CriticValues(float const* const* inputs, std::size_t rows, float* values);
 * // Training: forward rows (null rows are padding), then backward and one optimizer step.
 * void ActorForward(float const* const* observations, std::size_t rows, float* means);
 * void ActorBackwardAndStep(float const* meanGradients, float const* logStdGradients);
 * void CriticForward(float const* const* inputs, std::size_t rows, float* values);
 * void CriticBackwardAndStep(float const* valueGradients);
 * @endcode
 *
 * Random must provide `void Shuffle(std::vector<std::size_t>&)` and
 * `float Normal(float mean, float deviation)`, so that a client keeps its own generator
 * (and its saved state). Rows are passed as one pointer per agent; actions are written
 * agent after agent. Every buffer is allocated at construction: a team step never
 * allocates. Actions are Gaussian; the log-probability sums the action components.
 */
#pragma once

#include "ppokit/core/ppo_surrogate.hpp"
#include "ppokit/core/kl.hpp"
#include "ppokit/heads/gaussian.hpp"
#include "ppokit/runtime/team_rollout.hpp"
#include "ppokit/runtime/ppo_update_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppokit::runtime {

/** @brief What a row of the team holds at a step. */
enum class AgentPresence : std::uint8_t
{
    /** @brief No agent of this episode: the row takes no part in sampling or in the update. */
    Absent,
    /** @brief An agent of the episode that was destroyed: it shares the return (critic only). */
    Destroyed,
    /** @brief An agent that acts: it trains the actor and the critic. */
    Active,
};

/** @brief PPO hyper-parameters of a hosted MAPPO update, in the scalar of the rollout. */
struct TeamPpoSettings final
{
    /** @brief Discount factor: weight of future rewards. */
    float gamma = 0.99F;
    /** @brief GAE bias/variance trade-off. */
    float lambda = 0.95F;
    /** @brief PPO clipping range epsilon. */
    float clip = 0.2F;
    /** @brief Weight of the entropy bonus on the log standard deviation. */
    float entropyCoefficient = 0.F;
    /** @brief Sampled KL target; zero disables the actor-step guard. */
    float targetKl = 0.F;
    /** @brief Passes over each rollout. */
    std::size_t epochs = 4;
    /** @brief Rows per minibatch (the models' batch size). */
    std::size_t minibatch = 64;
};

/** @brief Diagnostics of one hosted update; means over the optimized samples. */
struct TeamUpdateStatistics final
{
    std::uint64_t update = 0;
    float policyLoss = 0.F;
    float valueLoss = 0.F;
    float entropy = 0.F;
    float approximateKl = 0.F;
    float clipFraction = 0.F;
    /** @brief Critic quality before the update (1 is perfect). */
    float explainedVariance = 0.F;
    /** @brief Mean |advantage| of the available rows, before normalization. */
    float advantageAbsMean = 0.F;
    /** @brief Sums of |gradient| sent to the networks (scale of the updates). */
    float actorDerivativeL1 = 0.F;
    float criticDerivativeL1 = 0.F;
    float logStdGradientL1 = 0.F;
    /** @brief Mean standard deviation of the actions after the update. */
    float actionStdMean = 0.F;
    std::size_t learnableRows = 0;
    std::size_t criticRows = 0;
    /** @brief Minibatches whose actor step was skipped by the KL guard. */
    std::size_t actorKlSkips = 0;
};

/**
 * @brief Optional notifications of the phases of an update (progress logs, timings).
 * @details Called synchronously from Observe; an empty function is skipped.
 */
struct HostedUpdateHooks final
{
    /** @brief Before the critic baselines; `update` counts from 1, `rows` = member rows of the rollout. */
    std::function<void(std::uint64_t update, std::size_t rows)> began;
    /** @brief After V and V' of every row. */
    std::function<void()> baselinesDone;
    /** @brief After the GAE and the normalization, before the first epoch. */
    std::function<void()> advantagesDone;
    /** @brief After each epoch; `epoch` counts from 1. */
    std::function<void(std::size_t epoch)> epochDone;
    /** @brief After the last epoch, with the update statistics. */
    std::function<void(TeamUpdateStatistics const& statistics)> finished;
};

/**
 * @brief Team-synchronized PPO driven by the client's loop.
 * @tparam Models Actor and critic of the client (see the file comment).
 * @tparam Random Random generator of the client (see the file comment).
 */
template<typename Models, typename Random>
class HostedTrainer final
{
public:
    using Rollout = TeamRollout<float>;

    /**
     * @param models Actor and critic; must outlive the trainer.
     * @param random Generator for sampling and shuffling; must outlive the trainer.
     * @param agents Team size, fixed for the trainer's life.
     * @param teamSteps Team steps collected per update.
     * @throws std::invalid_argument for zero sizes or inconsistent settings.
     */
    HostedTrainer(Models& models, Random& random, std::size_t agents, std::size_t teamSteps,
        TeamPpoSettings const& settings)
        : models_(models), random_(random), settings_(Checked(settings)), observationSize_(models.ObservationSize()),
          actionSize_(models.ActionSize()),
          rollout_(agents, teamSteps, observationSize_, models.CriticSize(), actionSize_),
          updateEngine_(rollout_.Size()), batchInputs_(settings.minibatch, nullptr),
          batchCritics_(settings.minibatch, nullptr), batchRows_(settings.minibatch, nullptr),
          batchMeans_(settings.minibatch * actionSize_), meanGradients_(settings.minibatch * actionSize_),
          logStdGradients_(actionSize_), batchValues_(settings.minibatch), valueGradients_(settings.minibatch),
          teamMeans_(agents * actionSize_)
    {
    }

    std::size_t ObservationSize() const noexcept { return observationSize_; }
    std::size_t ActionSize() const noexcept { return actionSize_; }

    std::size_t Agents() const noexcept { return rollout_.Agents(); }
    std::size_t TeamSteps() const noexcept { return rollout_.TeamSteps(); }
    std::size_t CollectedTeamSteps() const noexcept { return rollout_.Collected(); }
    bool HasPendingAction() const noexcept { return pending_; }
    /** @brief True between updates: no team step collected, none pending (safe to save or evaluate). */
    bool AtUpdateBoundary() const noexcept { return !pending_ && rollout_.Collected() == 0; }
    std::uint64_t CompletedUpdates() const noexcept { return updates_; }
    TeamUpdateStatistics const& LastUpdate() const noexcept { return statistics_; }
    TeamPpoSettings const& Settings() const noexcept { return settings_; }
    /** @brief Phase notifications of the following updates. */
    void SetHooks(HostedUpdateHooks hooks) { hooks_ = std::move(hooks); }

    /**
     * @brief Samples one action per agent from means the client already computed.
     * @param observations One row per agent (stored for training); may be null for an absent row.
     * @param means Actor means of those observations, `Agents() * ActionSize()` values.
     * @param critics Centralized critic input of each agent at this step; may be null for an absent row.
     * @param presence Of each row. A destroyed agent still acts (its action is ignored by the
     *        simulation) and shares the return, but does not train the actor. An absent row
     *        draws no action (its actions are zero) and is left out of the update.
     * @param actions Receives one sampled action per agent, `Agents() * ActionSize()` values.
     * @throws std::logic_error when the previous step has no feedback yet.
     * @throws std::runtime_error for a non-finite mean, deviation or sample (agent named).
     */
    void ChooseActionsFromMeans(float const* const* observations, float const* means, float const* const* critics,
        AgentPresence const* presence, float* actions)
    {
        if (pending_)
        {
            throw std::logic_error("PPOKit hosted trainer: ChooseActions needs the previous Observe");
        }
        for (std::size_t agent = 0; agent < Agents(); ++agent)
        {
            auto& row = rollout_.Current(agent);
            row.member = presence[agent] != AgentPresence::Absent;
            row.learnable = presence[agent] == AgentPresence::Active;
            row.logProbability = 0.F;
            if (!row.member)
            {
                std::fill_n(actions + agent * actionSize_, actionSize_, 0.F);
                continue;
            }
            std::copy_n(observations[agent], observationSize_, row.observation.begin());
            std::copy_n(critics[agent], row.critic.size(), row.critic.begin());
            for (std::size_t component = 0; component < actionSize_; ++component)
            {
                float const mean = means[agent * actionSize_ + component];
                float const logStd = models_.LogStd(component);
                float const deviation = std::exp(logStd);
                if (!std::isfinite(mean) || !std::isfinite(deviation) || deviation <= 0.F)
                {
                    throw std::runtime_error(
                        "PPOKit hosted trainer: invalid Gaussian distribution for agent " + std::to_string(agent));
                }
                float const sampled = random_.Normal(mean, deviation);
                if (!std::isfinite(sampled))
                {
                    throw std::runtime_error(
                        "PPOKit hosted trainer: non-finite action for agent " + std::to_string(agent));
                }
                row.logProbability += heads::GaussianAxisLogDensity(sampled, mean, logStd);
                actions[agent * actionSize_ + component] = sampled;
                row.action[component] = sampled;
            }
        }
        pending_ = true;
    }

    /** @brief Same as ChooseActionsFromMeans, the means computed by the models. */
    void ChooseActions(float const* const* observations, float const* const* critics,
        AgentPresence const* presence, float* actions)
    {
        models_.ActorMeans(observations, Agents(), teamMeans_.data());
        ChooseActionsFromMeans(observations, teamMeans_.data(), critics, presence, actions);
    }

    /**
     * @brief Feedback of the step: next critic inputs, rewards and episode ends. Absent rows'
     *        critic inputs (may be null) and rewards are ignored.
     * @return True when this step filled the rollout and an update ran (see LastUpdate).
     * @throws std::logic_error without a pending ChooseActions.
     */
    bool Observe(float const* const* nextCritics, float const* rewards, StepEnd const* ends)
    {
        if (!pending_)
        {
            throw std::logic_error("PPOKit hosted trainer: Observe needs ChooseActions first");
        }
        for (std::size_t agent = 0; agent < Agents(); ++agent)
        {
            auto& row = rollout_.Current(agent);
            row.end = ends[agent];
            if (!row.member)
            {
                row.reward = 0.F;
                continue;
            }
            std::copy_n(nextCritics[agent], row.nextCritic.size(), row.nextCritic.begin());
            row.reward = rewards[agent];
        }
        rollout_.EndTeamStep();
        pending_ = false;
        if (!rollout_.Full())
        {
            return false;
        }
        Update();
        rollout_.Clear();
        return true;
    }

    /** @brief Forgets an action without feedback (the simulation was reset mid-step). */
    void AbortPendingStep() noexcept { pending_ = false; }

    /** @brief Rows of the rollout, for a client's diagnostics or tests. */
    Rollout const& RolloutRows() const noexcept { return rollout_; }

private:
    static TeamPpoSettings Checked(TeamPpoSettings const& settings)
    {
        bool const valid = settings.gamma >= 0.F && settings.gamma <= 1.F && settings.lambda >= 0.F
            && settings.lambda <= 1.F && settings.clip > 0.F && std::isfinite(settings.targetKl)
            && settings.targetKl >= 0.F && settings.epochs > 0 && settings.minibatch > 0;
        if (!valid)
        {
            throw std::invalid_argument("PPOKit hosted trainer: invalid PPO settings");
        }
        return settings;
    }

    /** @brief Critic baselines, advantages, then the PPO epochs. */
    void Update()
    {
        statistics_ = {};
        std::size_t const members = rollout_.CollectMembers();
        if (members == 0)
        {
            throw std::logic_error("PPOKit hosted trainer: a rollout without any agent of an episode");
        }
        Notify(hooks_.began, updates_ + 1, members);
        EvaluateBaselines();
        Notify(hooks_.baselinesDone);
        statistics_.explainedVariance = PpoUpdateEngine::Prepare(rollout_, settings_.gamma, settings_.lambda,
            [this](std::size_t learnable) {
                statistics_.advantageAbsMean = rollout_.MeanAbsoluteLearnableAdvantage();
                statistics_.learnableRows = learnable;
                statistics_.criticRows = rollout_.MemberCount();
            });
        Notify(hooks_.advantagesDone);
        PpoSampleCounts const counts = updateEngine_.RunEpochs(settings_.epochs, settings_.minibatch,
            [this](auto& order) { random_.Shuffle(order); },
            [](std::size_t) {},
            [this](std::size_t start, std::size_t end) { return TrainMinibatch(start, end); },
            [this](std::size_t epoch) { Notify(hooks_.epochDone, epoch); });
        PpoUpdateEngine::Finalize(statistics_, counts);
        MeasureActionStd();
        statistics_.update = ++updates_;
        Notify(hooks_.finished, statistics_);
    }

    template<typename Hook, typename... Arguments>
    static void Notify(Hook const& hook, Arguments const&... arguments)
    {
        if (hook)
        {
            hook(arguments...);
        }
    }

    /** @brief Train available actor rows and all critic rows, preserving the fixed padded model batch. */
    std::size_t TrainMinibatch(std::size_t start, std::size_t end)
    {
        SelectBatch(start);
        std::size_t const learnable = CountLearnable();
        if (learnable != 0)
        {
            TrainActor(learnable);
        }
        TrainCritic(end - start);
        return learnable;
    }

    /** @brief V(critic) and V(nextCritic) of every member row, with this update's critic. */
    void EvaluateBaselines()
    {
        updateEngine_.ResetOrder(rollout_.MemberCount());
        for (std::size_t offset = 0; offset < updateEngine_.Size(); offset += settings_.minibatch)
        {
            SelectBatch(offset);
            for (bool const next : {false, true})
            {
                for (std::size_t row = 0; row < settings_.minibatch; ++row)
                {
                    auto const* const transition = batchRows_[row];
                    batchCritics_[row] = transition == nullptr
                        ? nullptr
                        : (next ? transition->nextCritic.data() : transition->critic.data());
                }
                models_.CriticValues(batchCritics_.data(), settings_.minibatch, batchValues_.data());
                for (std::size_t row = 0; row < settings_.minibatch; ++row)
                {
                    auto* const transition = batchRows_[row];
                    if (transition != nullptr)
                    {
                        (next ? transition->nextValue : transition->value) = batchValues_[row];
                    }
                }
            }
        }
    }

    /** @brief Member rows `offset .. offset + minibatch` of the current order; null past the end. */
    void SelectBatch(std::size_t offset)
    {
        for (std::size_t row = 0; row < settings_.minibatch; ++row)
        {
            std::size_t const position = offset + row;
            auto* const transition =
                position < updateEngine_.Size() ? &rollout_[rollout_.Member(updateEngine_.IndexAt(position))] : nullptr;
            batchRows_[row] = transition;
            batchInputs_[row] = transition == nullptr ? nullptr : transition->observation.data();
            batchCritics_[row] = transition == nullptr ? nullptr : transition->critic.data();
        }
    }

    std::size_t CountLearnable() const noexcept
    {
        std::size_t count = 0;
        for (auto const* const transition : batchRows_)
        {
            count += transition != nullptr && transition->learnable ? 1 : 0;
        }
        return count;
    }

    /** @brief Clipped surrogate on the available rows, mean over them, then one actor step. */
    void TrainActor(std::size_t batchLearnable)
    {
        models_.ActorForward(batchInputs_.data(), settings_.minibatch, batchMeans_.data());
        std::fill(meanGradients_.begin(), meanGradients_.end(), 0.F);
        std::fill(logStdGradients_.begin(), logStdGradients_.end(), 0.F);
        float sampledKl = 0.F;
        for (std::size_t row = 0; row < settings_.minibatch; ++row)
        {
            auto const* const transition = batchRows_[row];
            if (transition == nullptr || !transition->learnable)
            {
                continue;
            }
            float const* const means = &batchMeans_[row * actionSize_];
            float logProbability = 0.F;
            for (std::size_t component = 0; component < actionSize_; ++component)
            {
                logProbability += heads::GaussianAxisLogDensity(
                    transition->action[component], means[component], models_.LogStd(component));
            }
            auto const surrogate = core::EvaluateClippedSurrogate(
                logProbability, transition->logProbability, transition->advantage, settings_.clip);
            statistics_.policyLoss -= surrogate.objective;
            statistics_.approximateKl += std::max(0.F, surrogate.ratio - 1.F - surrogate.logRatio);
            if (settings_.targetKl > 0.F)
            {
                sampledKl += core::SampledKl(surrogate.logRatio) / static_cast<float>(batchLearnable);
            }
            statistics_.clipFraction += surrogate.outsideClipRange ? 1.F : 0.F;
            // Mean over the learnable samples of the minibatch.
            float const logProbabilityGradient =
                surrogate.logProbabilityGradient / static_cast<float>(batchLearnable);
            for (std::size_t component = 0; component < actionSize_; ++component)
            {
                float const logStd = models_.LogStd(component);
                statistics_.entropy += heads::GaussianAxisEntropy(logStd);
                auto const gradient =
                    heads::GaussianAxisLogDensityGradient(transition->action[component], means[component], logStd);
                float const meanGradient = logProbabilityGradient * gradient.mean;
                meanGradients_[row * actionSize_ + component] = meanGradient;
                statistics_.actorDerivativeL1 += std::abs(meanGradient);
                // The entropy bonus adds d entropy / d log-std = 1 per component.
                float const logStdGradient = logProbabilityGradient * gradient.logStd
                    - settings_.entropyCoefficient / static_cast<float>(batchLearnable);
                logStdGradients_[component] += logStdGradient;
                statistics_.logStdGradientL1 += std::abs(logStdGradient);
            }
        }
        // The critic still learns when the policy has moved beyond the target KL.
        if (!core::ActorStepAllowed(sampledKl, settings_.targetKl))
        {
            ++statistics_.actorKlSkips;
            return;
        }
        models_.ActorBackwardAndStep(meanGradients_.data(), logStdGradients_.data());
    }

    /** @brief Squared error to the value target on every row, mean over them, one critic step. */
    void TrainCritic(std::size_t criticRows)
    {
        models_.CriticForward(batchCritics_.data(), settings_.minibatch, batchValues_.data());
        std::fill(valueGradients_.begin(), valueGradients_.end(), 0.F);
        for (std::size_t row = 0; row < settings_.minibatch; ++row)
        {
            auto const* const transition = batchRows_[row];
            if (transition == nullptr)
            {
                continue;
            }
            float const difference = batchValues_[row] - transition->valueTarget;
            statistics_.valueLoss += difference * difference;
            float const gradient = 2.F * difference / static_cast<float>(criticRows);
            valueGradients_[row] = gradient;
            statistics_.criticDerivativeL1 += std::abs(gradient);
        }
        models_.CriticBackwardAndStep(valueGradients_.data());
    }

    /** @brief Mean action deviation after the common update has finished. */
    void MeasureActionStd() noexcept
    {
        float deviation = 0.F;
        for (std::size_t component = 0; component < actionSize_; ++component)
        {
            deviation += std::exp(models_.LogStd(component));
        }
        statistics_.actionStdMean = deviation / static_cast<float>(actionSize_);
    }

    Models& models_;
    Random& random_;
    TeamPpoSettings settings_;
    std::size_t observationSize_;
    std::size_t actionSize_;
    Rollout rollout_;
    PpoUpdateEngine updateEngine_;
    std::vector<float const*> batchInputs_;
    std::vector<float const*> batchCritics_;
    std::vector<typename Rollout::Row*> batchRows_;
    std::vector<float> batchMeans_;
    std::vector<float> meanGradients_;
    std::vector<float> logStdGradients_;
    std::vector<float> batchValues_;
    std::vector<float> valueGradients_;
    std::vector<float> teamMeans_;
    TeamUpdateStatistics statistics_;
    HostedUpdateHooks hooks_;
    std::uint64_t updates_ = 0;
    bool pending_ = false;
};

} // namespace ppokit::runtime
