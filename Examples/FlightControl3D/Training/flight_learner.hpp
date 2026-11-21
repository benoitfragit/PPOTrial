/**
 * @file flight_learner.hpp
 * @brief PPO learner: actor and critic MLPs (RLtools), rollout buffer, GAE, updates and checkpoints.
 *
 * Role in the chain: observation -> **PPO** -> action, and transitions -> **PPO update**.
 * The session asks Choose() for an action, Record()s the transition and calls Optimize()
 * once the rollout buffer holds TrainingSettings::rolloutSteps transitions.
 *
 * Networks: actor and critic are independent MLPs with two hidden tanh layers of
 * TrainingSettings::hiddenWidth units (32, 64, 128 or 256) on the 33 observations; the
 * actor has 5 identity outputs (Gaussian means) and 5 global log-std parameters, the critic
 * one identity output. Each has its own Adam optimizer.
 *
 * Checkpoint format (text, "PPO_FLIGHT_POLICY_V3"): header line (magic, observations,
 * actions, outputs, width, updates), quoted contract signature, Adam ages, RNG state, then
 * actor layers, actor log-std, critic layers, each parameter with its two Adam moments.
 * This format is a compatibility contract: never reorder or rename its fields.
 *
 * Not here: JSBSim, the mission and the session logic. This library has no ECS dependency.
 *
 * Read next: flight_learner.cpp, flight_policy.hpp, flight_model_store.hpp.
 */
#pragma once

#include "flight_policy.hpp"
#include <memory>

namespace ppo::examples::flight {

/** @brief One collected transition and its GAE outputs. */
struct Transition final
{
    Observation observation{};
    PolicyAction action{};
    /** @brief Sum of the channel rewards of the action. */
    double reward{};
    /** @brief Critic value V(s) at collection time. */
    double value{};
    /** @brief Critic value V(s') at collection time, zero for a terminated transition. */
    double nextValue{};
    /** @brief GAE advantage, then its normalized value during Optimize. */
    double advantage{};
    /** @brief Critic regression target: raw advantage + value. */
    double target{};
    bool terminated{};
    bool truncated{};
};

/**
 * @brief Computes GAE(lambda) advantages and value targets in place, from the last transition backward.
 *
 * A_t = delta_t + gamma * lambda * A_{t+1}, with the recursion cut at every episode
 * boundary (terminated or truncated) so no advantage leaks through a reset; a truncated
 * transition still bootstraps its own delta from V(s').
 * @param transitions Rollout in collection order.
 * @param count Number of transitions.
 * @param settings Discount gamma and GAE lambda.
 */
void ComputeAdvantages(Transition* transitions, std::size_t count, TrainingSettings const& settings) noexcept;

/** @brief Diagnostics from the last complete on-policy update. */
struct LearningStats final
{
    /** @brief Completed PPO updates, including those restored from a checkpoint. */
    std::uint64_t updates{};
    /** @brief Mean clipped surrogate (without the entropy bonus) over all samples and epochs. */
    double policyLoss{};
    /** @brief Mean squared critic error over all samples and epochs. */
    double valueLoss{};
    /** @brief Mean policy entropy over all samples and epochs. */
    double entropy{};
    /** @brief Mean action reward of the rollout. */
    double meanReward{};
    /** @brief Largest minibatch mean of exp(log r) - 1 - log r (always non-negative KL estimate). */
    double maximumApproximateKl{};
    /** @brief Largest sampled mean old-log-probability minus new-log-probability per minibatch. */
    double maximumSampledKl{};
    /** @brief Applied actor steps. */
    std::size_t actorSteps{};
    /** @brief Applied critic steps. */
    std::size_t criticSteps{};
    /** @brief Actor minibatches skipped by the KL guard. */
    std::size_t actorKlSkips{};
    /** @brief Mean weighted reward contribution per actor action and channel. */
    std::array<double, RewardCount> meanRewards{};
    /** @brief False before the first PPO update in this process, including after resume. */
    bool hasRewardMeans{};
};

/** @brief RLtools MLPs with a diagonal Gaussian PPO actor; no JSBSim/ECS dependency. */
class FlightLearner final
{
public:
    /**
     * @brief Initializes the networks from TrainingSettings::seed.
     * @param config Validated configuration (hidden width, PPO and Adam settings, signature).
     */
    explicit FlightLearner(Config const& config);
    ~FlightLearner();
    FlightLearner(FlightLearner const&) = delete;
    FlightLearner& operator=(FlightLearner const&) = delete;
    /** @brief Actor means and log-std for one observation (no sampling, no RNG use). */
    PolicyDistribution Evaluate(Observation const& observation);
    /**
     * @brief Samples controls; the brake is active only during the ground roll.
     * @param observation Normalized observation; its Rollout slot enables the brake axis.
     * @param deterministic True to return the means (validation, inference).
     * @return Raw action; a stochastic action consumes the learner RNG.
     */
    PolicyAction Choose(Observation const& observation, bool deterministic = false);
    /**
     * @brief Stores one transition and evaluates the critic on both states.
     * @throws std::logic_error when the rollout is full; std::invalid_argument for non-finite data.
     */
    void Record(Observation const& observation, PolicyAction const& action,
        StepResult const& step, Observation const& nextObservation);
    /** @brief True when the rollout buffer is full and Optimize must run. */
    bool Ready() const noexcept;
    /** @brief Runs one PPO update on the full rollout and empties it. */
    void Optimize();
    LearningStats const& Stats() const noexcept;
    /** @brief Transitions collected since the last update. */
    std::size_t Collected() const noexcept;
    /** @brief Update checkpoint metadata at an empty PPO rollout boundary. */
    void SetRewardContract(RewardSettings const& settings, std::string const& signature);
    /**
     * @brief Atomically writes the checkpoint at an update boundary.
     *
     * The previous file is kept as path + ".previous".
     */
    void Save(std::filesystem::path const& path) const;
    /** @brief Loads a compatible checkpoint at an update boundary; nothing changes on failure. */
    void Load(std::filesystem::path const& path);
    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

} // namespace ppo::examples::flight
