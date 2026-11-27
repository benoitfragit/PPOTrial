/**
 * @file Examples/Airspace/tests/training_reward_tests.cpp
 * @brief The reward RLMission and RLSNA train on under ppokit-run: the last-step mission failure is
 *        never paid, the reward is the clamped sum of the weighted channels, the step limit
 *        truncates, and the seed draws the episodes.
 * @details The environment DLL is driven as ppokit-run drives it while training (learning
 * on, reset, then begin_step and step) with fixed actions. The checks hold for every
 * episode instead of comparing recorded values (the distributions of the random initializer
 * differ between standard libraries):
 * - every reward and channel is finite, and the reward is the sum of the channels clamped to
 *   the settings' clamp;
 * - the "missionfailure" channel stays 0: the controllers pay it on IsLastStep, which was
 *   false under ppokit-run before the in-world modes were removed, so the policies were
 *   trained without it (an episode cut by the step limit was paid -20 when it was true);
 * - an episode lasts at most the settings' rolloutSteps, and a truncated one exactly that;
 * - at least one episode reaches the step limit (a full-throttle turn lasts to it);
 * - the seed given to the environment draws the episodes: the same seed gives the same
 *   rewards, another seed others.
 *
 * `training_reward_tests <environment DLL> <project> <training settings>`.
 */
#include "ppokit/environment/environment_library.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Json = nlohmann::ordered_json;
using ppokit::environment::Environment;
using ppokit::environment::EnvironmentLibrary;
using ppokit::runtime::StepEnd;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

Json ReadJson(std::string const& path)
{
    std::ifstream stream(path);
    if (!stream)
        throw std::runtime_error("cannot read " + path);
    return Json::parse(stream);
}

/**
 * @brief The fixed actions of an episode: turn and acceleration are `turn` and
 *        `acceleration` plus a slow sweep of amplitude `sweep`; fire (RLMission's value after
 *        them, RLSNA has none) is pressed while the sweep is positive when `fire`.
 */
struct ActionPlan final
{
    float turn = 0.F;
    float acceleration = 0.F;
    float sweep = 0.F;
    bool fire = false;
};

/** @brief A full-throttle turn: it lasts to the step limit without destroying the target. */
constexpr ActionPlan kToTheLimit{1.F, 1.F, 0.F, false};
/** @brief Attempts of kToTheLimit to reach the step limit (40 runs from the system's entropy all did). */
constexpr std::size_t kLimitAttempts = 5;
/** @brief Then a sweep that ends early, and the same sweep firing. */
constexpr std::array<ActionPlan, 2> kSweeps{{
    {0.F, 0.F, 0.6F, false},
    {0.F, 0.F, 0.6F, true},
}};

/** @brief The actions of step `step` of an episode played with `plan`. */
void FillActions(std::vector<float>& actions, std::size_t step, ActionPlan const& plan) noexcept
{
    constexpr std::size_t kTurn = 0;
    constexpr std::size_t kAcceleration = 1;
    for (std::size_t index = 0; index < actions.size(); ++index)
    {
        double const phase = 0.013 * static_cast<double>(step) + 1.7 * static_cast<double>(index);
        float const sweep = plan.sweep * static_cast<float>(std::sin(phase));
        if (index == kTurn)
            actions[index] = plan.turn + sweep;
        else if (index == kAcceleration)
            actions[index] = plan.acceleration + sweep;
        else if (plan.fire && sweep > 0.F)
            actions[index] = 1.F;
        else
            actions[index] = 0.F;
    }
}

/** @brief What the checks need of the task and of the training settings. */
struct Expectations final
{
    std::size_t missionFailure = 0;
    std::size_t rolloutSteps = 0;
    double clamp = 0.0;
};

/** @brief The "missionfailure" channel's index in task.json, the step limit and the clamp. */
Expectations LoadExpectations(std::string const& project, std::string const& settings)
{
    Expectations expectations;
    Json const task = ReadJson(project + "/task.json");
    std::size_t index = 0;
    bool found = false;
    for (auto const& reward : task.at("rewards").items())
    {
        if (reward.key() == "missionfailure")
        {
            expectations.missionFailure = index;
            found = true;
        }
        ++index;
    }
    if (!found)
        throw std::runtime_error("task.json has no missionfailure reward");
    Json const training = ReadJson(settings);
    expectations.rolloutSteps = training.at("training").at("rolloutSteps").get<std::size_t>();
    expectations.clamp = training.at("rewards").at("clamp").get<double>();
    return expectations;
}

/** @brief Buffers of the environment, sized by its info. */
struct Buffers final
{
    explicit Buffers(PpokitEnvironmentInfo const& info)
        : observations(info.agentCapacity * info.observationSize)
        , available(info.agentCapacity)
        , world(info.worldSize)
        , actions(info.agentCapacity * info.actionSize)
        , rewards(info.agentCapacity)
        , channels(info.agentCapacity * info.rewardChannels)
    {
    }

    float* World() noexcept { return world.empty() ? nullptr : world.data(); }

    std::vector<float> observations;
    std::vector<std::uint8_t> available;
    std::vector<float> world;
    std::vector<float> actions;
    std::vector<double> rewards;
    std::vector<double> channels;
};

/** @brief Checks one step's reward: finite, the clamped sum of the channels, no mission failure. */
void CheckStepReward(Buffers const& buffers, Expectations const& expectations, std::string const& where)
{
    double sum = 0.0;
    bool finite = std::isfinite(buffers.rewards[0]);
    for (double const channel : buffers.channels)
    {
        finite = finite && std::isfinite(channel);
        sum += channel;
    }
    Check(finite, where + ": finite reward and channels");
    double const clamped = std::clamp(sum, -expectations.clamp, expectations.clamp);
    // The agent sums its channels in float: equal to float precision.
    Check(std::fabs(buffers.rewards[0] - clamped) <= 1e-4 * std::max(1.0, std::fabs(clamped)),
        where + ": reward " + std::to_string(buffers.rewards[0]) + ", clamped sum of the channels "
            + std::to_string(clamped));
    Check(buffers.channels[expectations.missionFailure] == 0.0,
        where + ": the last-step mission failure is paid ("
            + std::to_string(buffers.channels[expectations.missionFailure]) + ")");
}

/** @brief Plays one episode with `plan`, checking each step. @return How it ended. */
StepEnd PlayEpisode(Environment& environment, Buffers& buffers, Expectations const& expectations,
    ActionPlan const& plan, std::size_t episode)
{
    environment.Reset(buffers.observations.data(), buffers.available.data(), buffers.World());
    std::string const name = "episode " + std::to_string(episode);
    StepEnd end = StepEnd::Running;
    std::size_t steps = 0;
    // The environment truncates at the step limit: the bound only stops a broken one.
    while (end == StepEnd::Running && steps <= expectations.rolloutSteps)
    {
        if (steps > 0)
            environment.BeginStep(buffers.observations.data(), buffers.available.data(), buffers.World());
        FillActions(buffers.actions, steps, plan);
        end = environment.Step(buffers.actions.data(), buffers.observations.data(), buffers.available.data(),
            buffers.World(), buffers.rewards.data(), buffers.channels.data());
        ++steps;
        CheckStepReward(buffers, expectations, name + " step " + std::to_string(steps));
    }
    Check(steps <= expectations.rolloutSteps, name + ": " + std::to_string(steps) + " steps, beyond the step limit");
    if (end == StepEnd::Truncated)
        Check(steps == expectations.rolloutSteps, name + ": truncated at " + std::to_string(steps) + " steps");
    return end;
}

/** @brief Every reward and channel of the kSweeps episodes of a new environment seeded `seed`. */
std::vector<double> RewardTrace(std::shared_ptr<EnvironmentLibrary const> const& library, char const* project,
    std::uint64_t seed)
{
    Environment environment(library, project, seed);
    environment.SetLearning(true);
    Buffers buffers(environment.Info());
    std::vector<double> trace;
    for (ActionPlan const& plan : kSweeps)
    {
        environment.Reset(buffers.observations.data(), buffers.available.data(), buffers.World());
        StepEnd end = StepEnd::Running;
        for (std::size_t step = 0; end == StepEnd::Running; ++step)
        {
            if (step > 0)
                environment.BeginStep(buffers.observations.data(), buffers.available.data(), buffers.World());
            FillActions(buffers.actions, step, plan);
            end = environment.Step(buffers.actions.data(), buffers.observations.data(), buffers.available.data(),
                buffers.World(), buffers.rewards.data(), buffers.channels.data());
            trace.push_back(buffers.rewards[0]);
            trace.insert(trace.end(), buffers.channels.begin(), buffers.channels.end());
        }
    }
    return trace;
}

/** @brief The same seed gives the same episodes, another seed other ones. */
void CheckSeeding(std::shared_ptr<EnvironmentLibrary const> const& library, char const* project)
{
    std::vector<double> const first = RewardTrace(library, project, 42);
    Check(RewardTrace(library, project, 42) == first, "the same seed gives the same episodes");
    Check(RewardTrace(library, project, 43) != first, "another seed gives other episodes");
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cerr << "usage: training_reward_tests <environment DLL> <project> <training settings>\n";
        return 2;
    }
    try
    {
        Expectations const expectations = LoadExpectations(argv[2], argv[3]);
        auto const library = EnvironmentLibrary::Open(argv[1]);
        Environment environment(library, argv[2], 0);
        Check(environment.Info().agentCapacity == 1, "an Airspace agent is alone");
        environment.SetLearning(true);
        Buffers buffers(environment.Info());
        std::size_t episode = 0;
        bool reachedLimit = false;
        for (std::size_t attempt = 0; attempt < kLimitAttempts && !reachedLimit; ++attempt)
        {
            ++episode;
            reachedLimit = PlayEpisode(environment, buffers, expectations, kToTheLimit, episode) == StepEnd::Truncated;
        }
        Check(reachedLimit, "an episode reaches the step limit, where the mission failure would be paid");
        for (ActionPlan const& plan : kSweeps)
        {
            ++episode;
            PlayEpisode(environment, buffers, expectations, plan, episode);
        }
        CheckSeeding(library, argv[2]);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    if (failures != 0)
    {
        std::cerr << failures << " training reward check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "training reward checks passed\n";
    return EXIT_SUCCESS;
}
