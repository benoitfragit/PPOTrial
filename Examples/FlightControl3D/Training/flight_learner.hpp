#pragma once
#include "flight_policy.hpp"
#include <memory>

namespace ppo::examples::flight {

struct Transition final
{
    Observation observation{};
    PolicyAction action{};
    double reward{}, value{}, nextValue{}, advantage{}, target{};
    bool terminated{}, truncated{};
};
void ComputeAdvantages(Transition* transitions, std::size_t count, TrainingSettings const& settings) noexcept;

/** @brief Diagnostics from the last complete on-policy update. */
struct LearningStats final
{
    std::uint64_t updates{};
    double policyLoss{}, valueLoss{}, entropy{}, meanReward{};
    double maximumApproximateKl{};
    /** @brief Largest sampled mean old-log-probability minus new-log-probability per minibatch. */
    double maximumSampledKl{};
    /** @brief Counts applied actor/critic steps and actor minibatches skipped by the KL guard. */
    std::size_t actorSteps{},criticSteps{},actorKlSkips{};
    /** @brief Mean weighted reward contribution per actor action and channel. */
    std::array<double,RewardCount> meanRewards{};
    /** @brief False before the first PPO update in this process, including after resume. */
    bool hasRewardMeans{};
};

/** RLtools MLPs with a diagonal Gaussian PPO actor; no JSBSim/ECS dependency. */
class FlightLearner final
{
public:
    explicit FlightLearner(Config const& config);
    ~FlightLearner();
    FlightLearner(FlightLearner const&) = delete;
    FlightLearner& operator=(FlightLearner const&) = delete;
    PolicyDistribution Evaluate(Observation const& observation);
    /** @brief Samples controls; the brake is active only during the ground roll. */
    PolicyAction Choose(Observation const& observation, bool deterministic = false);
    void Record(Observation const& observation, PolicyAction const& action,
        StepResult const& step, Observation const& nextObservation);
    bool Ready() const noexcept;
    void Optimize();
    LearningStats const& Stats() const noexcept;
    std::size_t Collected() const noexcept;
    /** @brief Update checkpoint metadata at an empty PPO rollout boundary. */
    void SetRewardContract(RewardSettings const& settings,std::string const& signature);
    void Save(std::filesystem::path const& path) const;
    void Load(std::filesystem::path const& path);
    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};
} // namespace ppo::examples::flight
