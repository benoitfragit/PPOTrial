#pragma once
#include "hybrid_policy.hpp"
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

struct LearningStats final { std::uint64_t updates{}; double policyLoss{}, valueLoss{}, entropy{}; };

/** RLtools MLPs with Gaussian + Bernoulli PPO; no JSBSim/ECS dependency. */
class FlightLearner final
{
public:
    explicit FlightLearner(Config const& config);
    ~FlightLearner();
    FlightLearner(FlightLearner const&) = delete;
    FlightLearner& operator=(FlightLearner const&) = delete;
    PolicyVector Evaluate(Observation const& observation);
    PolicyAction Choose(Observation const& observation, bool deterministic = false);
    void Record(Observation const& observation, PolicyAction const& action,
        StepResult const& step, Observation const& nextObservation);
    bool Ready() const noexcept;
    void Optimize();
    LearningStats const& Stats() const noexcept;
    std::size_t Collected() const noexcept;
    void Save(std::filesystem::path const& path) const;
    void Load(std::filesystem::path const& path);
    struct Impl;
private:
    std::unique_ptr<Impl> impl_;
};
} // namespace ppo::examples::flight
