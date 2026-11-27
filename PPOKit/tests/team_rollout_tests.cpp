/**
 * @file PPOKit/tests/team_rollout_tests.cpp
 * @brief TeamRollout: per-agent GAE along team steps, destroyed agents, episode ends, and
 *        bit-for-bit identity with the MAPPO formulas of BlueDefense (copied as an oracle).
 */
#include "ppokit/runtime/team_rollout.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using ppokit::runtime::StepEnd;
using Rollout = ppokit::runtime::TeamRollout<float>;

/** @brief Observation, critic input and action sizes of the rollouts of this test. */
constexpr std::size_t ObservationSize = 2;
constexpr std::size_t CriticSize = 3;
constexpr std::size_t ActionSize = 1;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

bool SameBits(float a, float b)
{
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

/** @brief Compares bit for bit and reports the first difference of each quantity. */
bool SameBitsReported(float actual, float expected, char const* quantity, std::size_t index, bool& reported)
{
    if (SameBits(actual, expected))
    {
        return true;
    }
    if (!reported)
    {
        std::uint32_t actualBits = 0;
        std::uint32_t expectedBits = 0;
        std::memcpy(&actualBits, &actual, sizeof(float));
        std::memcpy(&expectedBits, &expected, sizeof(float));
        std::cerr << "  first difference: " << quantity << " at row " << index << ": " << std::hexfloat << actual
                  << " (0x" << std::hex << actualBits << ") instead of " << std::hexfloat << expected << " (0x"
                  << std::hex << expectedBits << std::dec << std::defaultfloat << ")\n";
        reported = true;
    }
    return false;
}

/** @brief Fills a full rollout with random rewards and values, deaths and episode ends. */
void FillRandom(Rollout& rollout, std::mt19937& rng)
{
    std::uniform_real_distribution<float> uniform(-1.F, 1.F);
    std::uniform_int_distribution<int> event(0, 19);
    rollout.Clear();
    while (!rollout.Full())
    {
        for (std::size_t agent = 0; agent < rollout.Agents(); ++agent)
        {
            auto& row = rollout.Current(agent);
            row.reward = uniform(rng);
            row.logProbability = uniform(rng);
            int const draw = event(rng);
            row.end = draw == 0 ? StepEnd::Terminated : (draw == 1 ? StepEnd::Truncated : StepEnd::Running);
            row.learnable = event(rng) > 3;
        }
        rollout.EndTeamStep();
    }
    for (std::size_t index = 0; index < rollout.Size(); ++index)
    {
        rollout[index].value = uniform(rng);
        rollout[index].nextValue = uniform(rng);
    }
}

/** @brief Oracle: the BlueDefense GeneralizedPatrolLearner formulas, verbatim in float. */
struct BlueOracle final
{
    struct Transition final
    {
        float reward, value, nextValue, returnValue, advantage;
        bool available, terminated, truncated;
    };
    std::vector<Transition> transitions;
    std::size_t agents;
    std::size_t steps;

    static float GaeStep(float reward, float value, float nextValue, float following, float gamma, float lambda,
        bool terminated, bool truncated)
    {
        float const trace = terminated || truncated ? 0.F : following;
        return reward + (terminated ? 0.F : gamma * nextValue) - value + gamma * lambda * trace;
    }

    std::size_t Estimate(float gamma, float lambda, float& absoluteMean)
    {
        std::size_t learnable = 0;
        float absolute = 0.F;
        for (std::size_t agent = 0; agent < agents; ++agent)
        {
            float following = 0.F;
            for (std::size_t reverse = steps; reverse > 0; --reverse)
            {
                auto& t = transitions[(reverse - 1) * agents + agent];
                t.advantage = GaeStep(t.reward, t.value, t.nextValue, following, gamma, lambda, t.terminated,
                    t.truncated);
                t.returnValue = t.value + t.advantage;
                following = t.advantage;
                if (t.available)
                {
                    absolute += std::abs(t.advantage);
                    ++learnable;
                }
            }
        }
        absoluteMean = learnable == 0 ? 0.F : absolute / static_cast<float>(learnable);
        return learnable;
    }

    float Explained() const
    {
        float meanReturn = 0.F;
        float meanResidual = 0.F;
        for (auto const& t : transitions)
        {
            meanReturn += t.returnValue;
            meanResidual += t.returnValue - t.value;
        }
        float const inverse = 1.F / static_cast<float>(transitions.size());
        meanReturn *= inverse;
        meanResidual *= inverse;
        float returnVariance = 0.F;
        float residualVariance = 0.F;
        for (auto const& t : transitions)
        {
            float const centeredReturn = t.returnValue - meanReturn;
            float const centeredResidual = t.returnValue - t.value - meanResidual;
            returnVariance += centeredReturn * centeredReturn;
            residualVariance += centeredResidual * centeredResidual;
        }
        returnVariance *= inverse;
        residualVariance *= inverse;
        if (returnVariance <= 1e-8F)
        {
            return 0.F;
        }
        float const result = 1.F - residualVariance / returnVariance;
        return std::isfinite(result) ? result : 0.F;
    }

    void Normalize(std::size_t learnable)
    {
        float mean = 0.F;
        for (auto const& t : transitions)
            if (t.available)
                mean += t.advantage;
        mean /= static_cast<float>(learnable);
        float variance = 0.F;
        for (auto const& t : transitions)
        {
            if (!t.available)
                continue;
            float const centered = t.advantage - mean;
            variance += centered * centered;
        }
        float const scale = 1.F / std::sqrt(variance / static_cast<float>(learnable) + 1e-8F);
        for (auto& t : transitions)
            if (t.available)
                t.advantage = (t.advantage - mean) * scale;
    }
};

/** @brief Same rewards, values, deaths and ends: identical bits for every result. */
void BitIdenticalToBlue()
{
    std::mt19937 rng(123);
    constexpr float Gamma = 0.99F;
    constexpr float Lambda = 0.95F;
    for (std::size_t agents : {std::size_t{1}, std::size_t{3}, std::size_t{9}})
    {
        Rollout rollout(agents, 257, ObservationSize, CriticSize, ActionSize);
        FillRandom(rollout, rng);
        BlueOracle oracle{{}, agents, 257};
        for (std::size_t index = 0; index < rollout.Size(); ++index)
        {
            auto const& row = rollout[index];
            oracle.transitions.push_back({row.reward, row.value, row.nextValue, 0.F, 0.F, row.learnable,
                row.end == StepEnd::Terminated, row.end == StepEnd::Truncated});
        }
        float oracleAbsolute = 0.F;
        std::size_t const oracleLearnable = oracle.Estimate(Gamma, Lambda, oracleAbsolute);
        std::size_t const learnable = rollout.ComputeAdvantages(Gamma, Lambda);
        // One report per quantity: the first row where it differs, if any.
        bool same = learnable == oracleLearnable;
        bool reportedAbsolute = false;
        bool reportedExplained = false;
        bool reportedAdvantage = false;
        bool reportedTarget = false;
        bool reportedNormalized = false;
        same = SameBitsReported(rollout.MeanAbsoluteLearnableAdvantage(), oracleAbsolute, "mean |advantage|", 0,
                   reportedAbsolute) && same;
        same = SameBitsReported(rollout.ExplainedVariance(), oracle.Explained(), "explained variance", 0,
                   reportedExplained) && same;
        for (std::size_t index = 0; index < rollout.Size(); ++index)
        {
            same = SameBitsReported(rollout[index].advantage, oracle.transitions[index].advantage, "advantage", index,
                       reportedAdvantage) && same;
            same = SameBitsReported(rollout[index].valueTarget, oracle.transitions[index].returnValue, "value target",
                       index, reportedTarget) && same;
        }
        oracle.Normalize(oracleLearnable);
        same = rollout.NormalizeAdvantages() == oracleLearnable && same;
        for (std::size_t index = 0; index < rollout.Size(); ++index)
        {
            same = SameBitsReported(rollout[index].advantage, oracle.transitions[index].advantage,
                       "normalized advantage", index, reportedNormalized) && same;
        }
        Check(same, "bit-identical to the BlueDefense MAPPO formulas with " + std::to_string(agents) + " agents");
    }
}

/** @brief Agents are independent trajectories; a destroyed agent shares the return but is not learnable. */
void PerAgentSemantics()
{
    Rollout rollout(2, 3, ObservationSize, CriticSize, ActionSize);
    float const rewards[3][2] = {{1.F, 0.F}, {0.F, 0.F}, {2.F, 5.F}};
    for (int step = 0; step < 3; ++step)
    {
        for (std::size_t agent = 0; agent < 2; ++agent)
        {
            auto& row = rollout.Current(agent);
            row.reward = rewards[step][agent];
            row.learnable = !(agent == 1 && step > 0); // agent 1 destroyed after the first step
        }
        rollout.EndTeamStep();
    }
    Check(rollout.Full() && rollout.Collected() == 3, "three team steps fill the rollout");
    std::size_t const learnable = rollout.ComputeAdvantages(1.F, 1.F);
    Check(learnable == 4, "destroyed rows are not learnable");
    // With values at zero and gamma = lambda = 1, the advantage is the reward-to-go of each agent.
    Check(rollout.At(0, 0).advantage == 3.F && rollout.At(1, 0).advantage == 2.F, "agent 0 sums its own rewards");
    Check(rollout.At(0, 1).advantage == 5.F && rollout.At(2, 1).advantage == 5.F,
        "a destroyed agent still receives the mission return");

    Rollout ends(1, 3, ObservationSize, CriticSize, ActionSize);
    float const values[3] = {0.5F, 0.5F, 0.5F};
    StepEnd const kinds[3] = {StepEnd::Truncated, StepEnd::Running, StepEnd::Terminated};
    for (int step = 0; step < 3; ++step)
    {
        auto& row = ends.Current(0);
        row.reward = 1.F;
        row.end = kinds[step];
        ends.EndTeamStep();
    }
    for (int step = 0; step < 3; ++step)
    {
        ends.At(static_cast<std::size_t>(step), 0).value = values[step];
        ends.At(static_cast<std::size_t>(step), 0).nextValue = 2.F;
    }
    ends.ComputeAdvantages(1.F, 1.F);
    Check(ends.At(0, 0).advantage == 1.F + 2.F - 0.5F, "a truncated step bootstraps on its next value and stops");
    Check(ends.At(2, 0).advantage == 1.F - 0.5F, "a terminated step does not bootstrap");
    Check(ends.At(1, 0).advantage == (1.F + 2.F - 0.5F) + ends.At(2, 0).advantage,
        "the trace continues inside an episode");
}

void Errors()
{
    bool refused = false;
    try
    {
        Rollout rollout(0, 4, ObservationSize, CriticSize, ActionSize);
    }
    catch (std::invalid_argument const&)
    {
        refused = true;
    }
    Check(refused, "zero agents are refused");
    refused = false;
    try
    {
        Rollout rollout(1, 4, ObservationSize, 0, ActionSize);
    }
    catch (std::invalid_argument const&)
    {
        refused = true;
    }
    Check(refused, "a zero row dimension is refused");
    Rollout rollout(1, 1, ObservationSize, CriticSize, ActionSize);
    rollout.Current(0).reward = std::nanf("");
    refused = false;
    try
    {
        rollout.EndTeamStep();
    }
    catch (std::invalid_argument const& error)
    {
        refused = std::string(error.what()).find("agent 0") != std::string::npos;
    }
    Check(refused, "a NaN reward is refused with the agent named");
    rollout.Current(0).reward = 0.F;
    rollout.EndTeamStep();
    refused = false;
    try
    {
        rollout.Current(0);
    }
    catch (std::logic_error const&)
    {
        refused = true;
    }
    Check(refused, "a full rollout refuses another team step");
}

} // namespace

int main()
{
    BitIdenticalToBlue();
    PerAgentSemantics();
    Errors();
    if (failures != 0)
    {
        std::cerr << failures << " team rollout check(s) failed\n";
        return 1;
    }
    std::cout << "PPOKit team rollout tests passed\n";
    return 0;
}
