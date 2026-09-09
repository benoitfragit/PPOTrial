#pragma once
#include "ppo/rl/environment.hpp"
#include <filesystem>

namespace ppo::rl {
/// Evaluation implementations must own a frozen policy and avoid exploration.
class IPolicy
{
public:
    virtual ~IPolicy() = default;
    virtual Value act(const Value& observation) = 0;
};
class ILearner : public IPolicy
{
public:
    /// Called with the pre-reset final observation, including on truncation.
    virtual void observe(const StepResult& transition) = 0;
};
struct Episode
{
    double reward = 0;
    std::size_t length = 0;
    Info final_info;
    bool terminated = false;
    bool truncated = false;
};
/// One implementation of the episode loop for training and evaluation.
/// max_steps is an external time limit; zero is rejected. Does not autoreset at the end.
Episode run_episode(IEnvironment& environment, IPolicy& policy,
                    const ResetOptions& reset, std::size_t max_steps);
Episode train_episode(IEnvironment& environment, ILearner& learner,
                      const ResetOptions& reset, std::size_t max_steps);
/// Converts a truncated transition to the existing PPO done/masked-GAE representation.
/// The additional value is bootstrapped once; the recurrence then stops at the boundary.
float bootstrap_reward(double reward, bool terminated, bool truncated,
                       float final_value, float gamma);

enum class Reduction { Mean, Sum, Min, Max, Std, Rate };
struct Metric
{
    std::string key; // episode_reward, episode_length, or a numeric final-info key
    Reduction reduction = Reduction::Mean;
    double weight = 1.0;
};
/// Population standard deviation; rate is the fraction of nonzero numeric values.
/// Missing, nonnumeric and nonfinite metrics fail explicitly, never silently become zero.
/// Strict text rows: metric-key reduction weight. Lines starting with # are comments.
std::vector<Metric> load_metrics(const std::filesystem::path& path);
double aggregate(const std::vector<Episode>& episodes, const Metric& metric);
double score(const std::vector<Episode>& episodes, const std::vector<Metric>& metrics);
/// Immutable candidate directory + single atomic best-reference file.
/// Caller supplies a fully saved candidate and its independently evaluated score.
/// Returns false for ties/worse candidates. Existing candidate directories are never overwritten.
/// Publication failure throws, preserving the previous best reference.
bool promote_checkpoint(const std::filesystem::path& candidate,
                        const std::filesystem::path& store, const std::string& version,
                        double candidate_score);
} // namespace ppo::rl
