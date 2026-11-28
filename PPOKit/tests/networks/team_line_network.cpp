/**
 * @file PPOKit/tests/networks/team_line_network.cpp
 * @brief A team network DLL for the ppokit-run tests (team_line environment): a one-gain
 *        policy, action = gain x offset, behind the team API.
 * @details Its "update" is a fixed gain step every RolloutSteps team steps: enough to test
 * the run loop (acting for present agents only, observing, updating, saving between
 * rollouts, resuming, BEST) without a learning engine. Like a MAPPO learner, it refuses to
 * save in the middle of a rollout and to observe without a pending step; it reports its
 * settings and probes its two "networks" every ProbeEveryUpdates updates.
 */
#include "ppokit/network/team_sdk.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>

namespace {

constexpr std::size_t Capacity = 3;
constexpr std::size_t ObservationSize = 2;
constexpr std::size_t ActionSize = 1;
constexpr std::size_t RolloutSteps = 16;
constexpr double InitialGain = 0.5;
constexpr double GainStep = 0.25;
constexpr double ExplorationDeviation = 0.1;
constexpr std::uint64_t ProbeEveryUpdates = 2;
constexpr char const* PolicyTag = "team_line_policy";
constexpr char const* TrainingTag = "team_line_training";

/** @brief Deterministic actions of the present agents; absent rows are zero. */
void Means(double gain, float const* observations, std::uint8_t const* available, float* actions) noexcept
{
    for (std::size_t agent = 0; agent < Capacity; ++agent)
    {
        double const offset = observations[agent * ObservationSize];
        actions[agent * ActionSize] = available[agent] != 0 ? static_cast<float>(gain * offset) : 0.F;
    }
}

/** @brief Reads "<tag> <gain> <updates>" from `file`. @throws std::runtime_error otherwise. */
void ReadState(std::filesystem::path const& file, char const* tag, double& gain, std::uint64_t& updates)
{
    std::ifstream in(file);
    std::string read;
    if (!(in >> read >> gain >> updates) || read != tag)
    {
        throw std::runtime_error(file.string() + ": not a " + tag + " file");
    }
}

void WriteState(std::filesystem::path const& file, char const* tag, double gain, std::uint64_t updates)
{
    std::filesystem::create_directories(file.parent_path());
    std::ofstream out(file);
    out << tag << ' ' << gain << ' ' << updates << '\n';
    if (!out)
    {
        throw std::runtime_error("cannot write " + file.string());
    }
}

class TeamLineLearner final
{
public:
    TeamLineLearner(std::filesystem::path const& /*project*/, std::uint64_t seed)
        : random_(static_cast<std::uint32_t>(seed == 0 ? 7 : seed))
    {
    }

    void Act(float const* observations, std::uint8_t const* available, float* actions, bool deterministic)
    {
        if (pending_)
        {
            throw std::logic_error("the previous step was not observed");
        }
        if (expectNextDecision_ && (!std::equal(nextAvailable_.begin(), nextAvailable_.end(), available)
            || !std::equal(nextObservation_.begin(), nextObservation_.end(), observations)))
        {
            throw std::logic_error("the team bootstrap differs from the next decision");
        }
        Means(gain_, observations, available, actions);
        if (deterministic)
        {
            return;
        }
        std::normal_distribution<double> noise(0.0, ExplorationDeviation);
        for (std::size_t agent = 0; agent < Capacity; ++agent)
        {
            if (available[agent] != 0)
            {
                actions[agent * ActionSize] += static_cast<float>(noise(random_));
            }
        }
        pending_ = true;
    }

    bool Observe(float const* observations, std::uint8_t const* available, double const* rewards,
        ppokit::runtime::StepEnd end)
    {
        if (!pending_)
        {
            throw std::logic_error("observe without sampled actions");
        }
        expectNextDecision_ = end == ppokit::runtime::StepEnd::Running;
        if (expectNextDecision_)
        {
            std::copy_n(observations, nextObservation_.size(), nextObservation_.begin());
            std::copy_n(available, nextAvailable_.size(), nextAvailable_.begin());
        }
        pending_ = false;
        for (std::size_t agent = 0; agent < Capacity; ++agent)
        {
            if (!std::isfinite(rewards[agent]))
            {
                throw std::invalid_argument("nonfinite reward");
            }
        }
        if (++collected_ < RolloutSteps)
        {
            return false;
        }
        collected_ = 0;
        gain_ += GainStep;
        ++updates_;
        return true;
    }

    PpokitUpdateStatistics Statistics() const
    {
        PpokitUpdateStatistics statistics{};
        statistics.updates = updates_;
        statistics.learnableRows = RolloutSteps;
        return statistics;
    }

    PpokitTeamSettings Settings() const noexcept
    {
        PpokitTeamSettings settings{};
        settings.gamma = 0.5;
        settings.epochs = 1;
        settings.minibatch = RolloutSteps;
        return settings;
    }

    /** @brief The gain as one "actor" block and one "critic" block, on probed updates. */
    std::size_t ProbeCount() const noexcept { return updates_ % ProbeEveryUpdates == 0 ? 2 : 0; }

    PpokitBlockProbe Probe(std::size_t index) const
    {
        if (index >= ProbeCount())
        {
            throw std::out_of_range("no such probe");
        }
        PpokitBlockProbe probe{};
        probe.network = index == 0 ? "actor" : "critic";
        probe.block = "gain";
        probe.type = "dense";
        probe.samples = 1;
        probe.mean = gain_;
        probe.minimum = gain_;
        probe.maximum = gain_;
        return probe;
    }

    void Save(std::filesystem::path const& trainingFile, std::filesystem::path const& policyFile) const
    {
        if (pending_ || collected_ != 0)
        {
            throw std::logic_error("a team learner saves between rollouts only");
        }
        if (!trainingFile.empty())
        {
            WriteState(trainingFile, TrainingTag, gain_, updates_);
        }
        if (!policyFile.empty())
        {
            WriteState(policyFile, PolicyTag, gain_, updates_);
        }
    }

    void Load(std::filesystem::path const& trainingFile) { ReadState(trainingFile, TrainingTag, gain_, updates_); }

    std::uint64_t Updates() const noexcept { return updates_; }

private:
    std::mt19937 random_;
    double gain_ = InitialGain;
    std::uint64_t updates_ = 0;
    std::size_t collected_ = 0;
    bool pending_ = false;
    bool expectNextDecision_ = false;
    std::array<float, Capacity * ObservationSize> nextObservation_{};
    std::array<std::uint8_t, Capacity> nextAvailable_{};
};

class TeamLinePolicy final
{
public:
    TeamLinePolicy(std::filesystem::path const& /*project*/, std::filesystem::path const& policyFile)
    {
        std::uint64_t updates = 0;
        ReadState(policyFile, PolicyTag, gain_, updates);
    }

    void Act(float const* observations, std::uint8_t const* available, float* actions) const noexcept
    {
        Means(gain_, observations, available, actions);
    }

private:
    double gain_ = InitialGain;
};

} // namespace

PPOKIT_TEAM_NETWORK(TeamLineLearner, TeamLinePolicy, "team_line", Capacity, ObservationSize, ActionSize, "test")
