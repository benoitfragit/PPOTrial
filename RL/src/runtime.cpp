#include "ppo/rl/runtime.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <stdexcept>
#include <sstream>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ppo::rl {
namespace {
Episode execute(IEnvironment& environment, IPolicy& policy, ILearner* learner,
                const ResetOptions& reset, std::size_t max_steps)
{
    EpisodeSession session(environment, policy, max_steps, learner);
    session.reset(reset);
    while (!session.finished()) session.advance();
    return session.episode();
}

double metric_value(const Episode& episode, const std::string& key)
{
    if (key == "episode_reward") return episode.reward;
    if (key == "episode_length") return static_cast<double>(episode.length);
    const auto found = episode.final_info.find(key);
    if (found == episode.final_info.end())
        throw std::invalid_argument("missing metric: " + key);
    const auto& value = found->second;
    if (const auto* number = std::get_if<double>(&value)) return *number;
    if (const auto* integer = std::get_if<std::int64_t>(&value)) return static_cast<double>(*integer);
    if (const auto* flag = std::get_if<bool>(&value)) return *flag ? 1.0 : 0.0;
    throw std::invalid_argument("nonnumeric metric: " + key);
}
}
EpisodeSession::EpisodeSession(IEnvironment& environment, IPolicy& policy,
                               std::size_t max_steps, ILearner* learner)
    : environment_(environment), policy_(policy), learner_(learner), max_steps_(max_steps)
{
    if (max_steps == 0) throw std::invalid_argument("max_steps must be positive");
    if (learner != nullptr && static_cast<IPolicy*>(learner) != &policy)
        throw std::invalid_argument("learner must be the acting policy");
}
void EpisodeSession::reset(const ResetOptions& options)
{
    if (faulted_) throw std::logic_error("recreate session and policy after a failed transition");
    if (!finished_) throw std::logic_error("finish the current episode before reset");
    auto initial = environment_.reset(options);
    if (!contains(environment_.observation_space(), initial.observation))
        throw std::runtime_error("reset observation outside space");
    transition_ = {std::move(initial.observation), 0, false, false, std::move(initial.info)};
    episode_ = {};
    stop_requested_ = false;
    finished_ = false;
}
const StepResult& EpisodeSession::advance()
{
    if (finished_) throw std::logic_error("reset required before advance");
    // Fail closed on an exception: never reuse a possibly pending learner transition.
    finished_ = true;
    faulted_ = true;
    auto action = policy_.act(transition_.observation);
    if (!contains(environment_.action_space(), action))
        throw std::runtime_error("policy action outside space");
    auto result = environment_.step(action);
    if (!contains(environment_.observation_space(), result.observation) || !std::isfinite(result.reward))
        throw std::runtime_error("invalid environment transition");
    if (episode_.length == max_steps_ - 1 || stop_requested_) result.truncated = true;
    if (learner_ != nullptr) learner_->observe(result);
    episode_.reward += result.reward;
    if (!std::isfinite(episode_.reward)) throw std::overflow_error("episode reward overflow");
    ++episode_.length;
    episode_.final_info = result.info;
    episode_.terminated = result.terminated;
    episode_.truncated = result.truncated;
    transition_ = std::move(result);
    finished_ = transition_.terminated || transition_.truncated;
    faulted_ = false;
    return transition_;
}
void EpisodeSession::request_stop() noexcept { stop_requested_ = true; }
bool EpisodeSession::finished() const noexcept { return finished_; }
const Episode& EpisodeSession::episode() const noexcept { return episode_; }
const Value& EpisodeSession::observation() const noexcept { return transition_.observation; }

Episode run_episode(IEnvironment& env, IPolicy& policy, const ResetOptions& reset, std::size_t limit)
{ return execute(env, policy, nullptr, reset, limit); }
Episode train_episode(IEnvironment& env, ILearner& learner, const ResetOptions& reset, std::size_t limit)
{ return execute(env, learner, &learner, reset, limit); }

float bootstrap_reward(double reward, bool terminated, bool truncated, float final_value, float gamma)
{
    if (!std::isfinite(reward) || !std::isfinite(final_value)
        || !std::isfinite(gamma) || gamma < 0 || gamma > 1)
        throw std::invalid_argument("invalid bootstrap input");
    const double adjusted = reward + ((!terminated && truncated) ? gamma * static_cast<double>(final_value) : 0.0);
    if (!std::isfinite(adjusted) || std::abs(adjusted) > std::numeric_limits<float>::max())
        throw std::overflow_error("bootstrap reward overflows float");
    return static_cast<float>(adjusted);
}

std::vector<Metric> load_metrics(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("cannot open metrics configuration");
    std::vector<Metric> result;
    std::string line;
    while (std::getline(input, line))
    {
        if (line.empty() || line.front() == '#') continue;
        std::istringstream row(line);
        row.imbue(std::locale::classic());
        Metric metric;
        std::string reduction, extra;
        if (!(row >> metric.key >> reduction >> metric.weight) || (row >> extra)
            || !std::isfinite(metric.weight)) throw std::invalid_argument("invalid metric configuration");
        if (reduction == "mean") metric.reduction = Reduction::Mean;
        else if (reduction == "sum") metric.reduction = Reduction::Sum;
        else if (reduction == "min") metric.reduction = Reduction::Min;
        else if (reduction == "max") metric.reduction = Reduction::Max;
        else if (reduction == "std") metric.reduction = Reduction::Std;
        else if (reduction == "rate") metric.reduction = Reduction::Rate;
        else throw std::invalid_argument("unknown reduction");
        result.push_back(metric);
        if (result.size() > 1024) throw std::invalid_argument("too many metrics");
    }
    if (input.bad() || result.empty()) throw std::invalid_argument("empty or unreadable metrics configuration");
    return result;
}

std::string metrics_text(const std::vector<Metric>& metrics)
{
    if (metrics.empty() || metrics.size() > 1024) throw std::invalid_argument("invalid scoring criteria count");
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& metric : metrics)
    {
        if (metric.key.empty() || metric.key.front() == '#'
            || metric.key.find_first_of(" \t\r\n") != std::string::npos || !std::isfinite(metric.weight))
            throw std::invalid_argument("invalid metric key or weight");
        const char* reduction = nullptr;
        switch (metric.reduction)
        {
        case Reduction::Mean: reduction = "mean"; break;
        case Reduction::Sum: reduction = "sum"; break;
        case Reduction::Min: reduction = "min"; break;
        case Reduction::Max: reduction = "max"; break;
        case Reduction::Std: reduction = "std"; break;
        case Reduction::Rate: reduction = "rate"; break;
        default: throw std::invalid_argument("unknown metric reduction");
        }
        output << metric.key << ' ' << reduction << ' ' << metric.weight << '\n';
    }
    return output.str();
}

double aggregate(const std::vector<Episode>& episodes, const Metric& metric)
{
    if (episodes.empty()) throw std::invalid_argument("empty evaluation");
    double mean = 0, squares = 0, sum = 0, minimum = std::numeric_limits<double>::infinity();
    double maximum = -minimum;
    std::size_t count = 0, nonzero = 0;
    for (const auto& episode : episodes)
    {
        const double value = metric_value(episode, metric.key);
        if (!std::isfinite(value)) throw std::invalid_argument("nonfinite metric");
        ++count;
        if (value != 0) ++nonzero;
        const double delta = value - mean;
        mean += delta / static_cast<double>(count);
        squares += delta * (value - mean);
        sum += value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
    }
    double result = 0;
    switch (metric.reduction)
    {
    case Reduction::Mean: result = mean; break;
    case Reduction::Sum: result = sum; break;
    case Reduction::Min: result = minimum; break;
    case Reduction::Max: result = maximum; break;
    case Reduction::Std:
        if (!std::isfinite(squares)) throw std::overflow_error("variance accumulation overflow");
        result = std::sqrt(std::max(0.0, squares / static_cast<double>(count)));
        break;
    case Reduction::Rate: result = static_cast<double>(nonzero) / static_cast<double>(count); break;
    default: throw std::invalid_argument("unknown metric reduction");
    }
    if (!std::isfinite(result)) throw std::overflow_error("metric aggregation overflow");
    return result;
}
double score(const std::vector<Episode>& episodes, const std::vector<Metric>& metrics)
{
    if (metrics.empty()) throw std::invalid_argument("empty scoring criteria");
    double total = 0;
    for (const auto& metric : metrics)
    {
        if (!std::isfinite(metric.weight)) throw std::invalid_argument("invalid metric weight");
        total += metric.weight * aggregate(episodes, metric);
    }
    if (!std::isfinite(total)) throw std::overflow_error("score overflow");
    return total;
}

bool promote_checkpoint(const std::filesystem::path& candidate, const std::filesystem::path& store,
                        const std::string& version, double candidate_score, const std::string& evaluation_context)
{
    if (evaluation_context.size() > 1024 * 1024 || !std::isfinite(candidate_score) || version.empty()
        || version.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos)
        throw std::invalid_argument("invalid checkpoint version or score");
    if (!std::filesystem::is_directory(candidate) || std::filesystem::is_empty(candidate))
        throw std::invalid_argument("candidate must be a saved, nonempty directory");
    std::filesystem::create_directories(store);
    // Exclusive directory lock avoids competing promotion of an older policy.
    const auto lock = store / ".promotion-lock";
    if (!std::filesystem::create_directory(lock)) throw std::runtime_error("checkpoint store locked");
    try
    {
        const auto best = store / "best.txt";
        if (std::filesystem::exists(best))
        {
            std::ifstream input(best, std::ios::binary);
            input.imbue(std::locale::classic());
            double best_score;
            std::string best_version, best_context;
            if (!(input >> best_score >> best_version >> std::quoted(best_context)) || !std::isfinite(best_score)
                || best_version.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_") != std::string::npos)
                throw std::runtime_error("invalid best checkpoint reference");
            input >> std::ws;
            if (!input.eof()) throw std::runtime_error("trailing best checkpoint reference data");
            if (best_context != evaluation_context)
                throw std::invalid_argument("evaluation context differs; use a separate checkpoint store");
            if (candidate_score <= best_score)
            {
                std::filesystem::remove(lock);
                return false;
            }
        }
        const auto destination = store / version;
        if (std::filesystem::exists(destination)) throw std::invalid_argument("checkpoint version already exists");
        // The caller must save candidates outside this store; avoid recursive self-copy.
        const auto source = std::filesystem::canonical(candidate);
        const auto target = std::filesystem::weakly_canonical(store);
        const auto relative = target.lexically_relative(source);
        if (relative.empty() || (*relative.begin() != ".."))
            throw std::invalid_argument("checkpoint store cannot be inside candidate");
        std::filesystem::copy(candidate, destination, std::filesystem::copy_options::recursive);
        const auto temporary = lock / "best.txt";
        {
            std::ofstream output(temporary, std::ios::binary);
            output.imbue(std::locale::classic());
            output << std::setprecision(17) << candidate_score << '\n' << version << '\n' << std::quoted(evaluation_context) << '\n';
            output.close();
            if (!output) throw std::runtime_error("cannot write checkpoint reference");
        }
        // On platforms that cannot replace an existing file atomically, fail safely.
        #ifdef _WIN32
        if (!MoveFileExW(temporary.c_str(), best.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), "checkpoint publication");
#else
        std::filesystem::rename(temporary, best);
#endif
        std::filesystem::remove(lock);
        return true;
    }
    catch (...)
    {
        std::error_code cleanup_error;
        std::filesystem::remove_all(lock, cleanup_error);
        throw;
    }
}
} // namespace ppo::rl
