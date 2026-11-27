/**
 * @file PPOKit/tests/ppo_update_engine_tests.cpp
 * @brief Shared update scheduling, rollout preparation and reduction regressions.
 */
#include "ppokit/runtime/ppo_update_engine.hpp"
#include "ppokit/runtime/rollout.hpp"
#include "ppokit/runtime/team_rollout.hpp"
#include "ppokit/core/ppo_settings.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using namespace ppokit::runtime;
int failures = 0;

void Check(bool condition, char const* message)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << message << '\n';
        ++failures;
    }
}

/** @brief Frozen schedule oracle: same seed, order and partial minibatches as the old loops. */
void ScheduleMatchesReference()
{
    constexpr std::size_t rows = 11;
    constexpr std::size_t minibatch = 4;
    constexpr std::size_t epochs = 3;
    std::vector<std::size_t> expectedOrder(rows);
    std::iota(expectedOrder.begin(), expectedOrder.end(), std::size_t{0});
    std::mt19937 expectedRandom(27);
    std::vector<std::vector<std::size_t>> expected;
    for (std::size_t epoch = 0; epoch < epochs; ++epoch)
    {
        std::shuffle(expectedOrder.begin(), expectedOrder.end(), expectedRandom);
        for (std::size_t start = 0; start < rows; start += minibatch)
        {
            auto const end = std::min(start + minibatch, rows);
            expected.emplace_back(expectedOrder.begin() + start, expectedOrder.begin() + end);
        }
    }
    PpoUpdateEngine engine(rows);
    std::mt19937 actualRandom(27);
    std::vector<std::vector<std::size_t>> actual;
    std::vector<std::size_t> began;
    std::vector<std::size_t> ended;
    auto const counts = engine.RunEpochs(epochs, minibatch,
        [&](auto& order) { std::shuffle(order.begin(), order.end(), actualRandom); },
        [&](std::size_t epoch) { began.push_back(epoch); },
        [&](std::size_t start, std::size_t end) {
            std::vector<std::size_t> batch;
            std::size_t learnable = 0;
            for (std::size_t position = start; position < end; ++position)
            {
                auto const index = engine.IndexAt(position);
                batch.push_back(index);
                learnable += index % 2 == 0 ? 1 : 0;
            }
            actual.push_back(std::move(batch));
            return learnable;
        },
        [&](std::size_t epoch) { ended.push_back(epoch); });
    Check(actual == expected && actualRandom == expectedRandom, "schedule and RNG match reference exactly");
    Check(began == std::vector<std::size_t>{0, 1, 2} && ended == std::vector<std::size_t>{1, 2, 3},
        "epoch notifications retain their order");
    Check(counts.actor == 18 && counts.critic == 33, "actor and critic use different populations");
    ppokit::core::UpdateStatistics statistics;
    statistics.policyLoss = 36.0;
    statistics.valueLoss = 99.0;
    statistics.entropy = 18.0;
    PpoUpdateEngine::Finalize(statistics, counts);
    Check(statistics.policyLoss == 2.0 && statistics.valueLoss == 3.0 && statistics.entropy == 1.0,
        "loss reduction uses respective sample populations");
    engine.ResetOrder(rows);
    Check(engine.IndexAt(0) == 0 && engine.IndexAt(rows - 1) == rows - 1, "next update starts in collection order");
    auto const none = engine.RunEpochs(1, 4, [](auto&) {}, [](std::size_t) {},
        [](std::size_t, std::size_t) { return std::size_t{0}; }, [](std::size_t) {});
    Check(none.actor == 0 && none.critic == rows, "all-masked actor still trains critic");
    bool refused = false;
    try
    {
        engine.RunEpochs(1, 0, [](auto&) {}, [](std::size_t) {},
            [](std::size_t, std::size_t) { return std::size_t{0}; }, [](std::size_t) {});
    }
    catch (std::invalid_argument const&) { refused = true; }
    Check(refused, "zero minibatch cannot hang the optimizer");
}

/** @brief One-agent synchronized and ordinary buffers preserve the same independent episode boundaries. */
void SingleAgentMatchesTeam()
{
    RolloutBuffer<float, 1, 1> single(5);
    TeamRollout<float> team(1, 5, 1, 1, 1);
    for (std::size_t step = 0; step < 5; ++step)
    {
        RolloutBuffer<float, 1, 1>::Row row;
        row.reward = static_cast<float>(step) * 0.25F;
        row.value = 0.2F;
        row.nextValue = 0.4F;
        row.learnable = step != 3;
        row.end = step == 1 ? StepEnd::Truncated : (step == 4 ? StepEnd::Terminated : StepEnd::Running);
        single.Append(row);
        auto& other = team.Current(0);
        other.reward = row.reward;
        other.value = row.value;
        other.nextValue = row.nextValue;
        other.learnable = row.learnable;
        other.end = row.end;
        team.EndTeamStep();
    }
    std::size_t singleLearnable = 0;
    std::size_t teamLearnable = 0;
    float const singleVariance = PpoUpdateEngine::Prepare(single, 0.99F, 0.95F,
        [&](std::size_t count) { singleLearnable = count; });
    float const teamVariance = PpoUpdateEngine::Prepare(team, 0.99F, 0.95F,
        [&](std::size_t count) { teamLearnable = count; });
    bool identical = singleLearnable == 4 && teamLearnable == 4 && singleVariance == teamVariance;
    for (std::size_t step = 0; step < 5; ++step)
    {
        identical = identical && single[step].advantage == team[step].advantage
            && single[step].valueTarget == team[step].valueTarget;
    }
    Check(identical, "PPO and one-agent MAPPO produce identical targets and normalized advantages");
}

/** @brief Compare interleaved streams with a frozen, independent per-stream recursion. */
template<typename Scalar>
void InterleavedTrajectoryParity()
{
    constexpr std::size_t streams = 3;
    constexpr std::size_t steps = 7;
    constexpr Scalar gamma = Scalar(0.99);
    constexpr Scalar lambda = Scalar(0.95);
    using Single = RolloutBuffer<Scalar, 1, 1>;
    TeamRollout<Scalar> team(streams, steps, 1, 1, 1);
    std::array<Single, streams> independent{Single(steps), Single(steps), Single(steps)};
    std::array<Scalar, streams * steps> expectedAdvantage{};
    std::array<Scalar, streams * steps> expectedTarget{};
    std::size_t expectedLearnable = 0;
    for (std::size_t step = 0; step < steps; ++step)
    {
        for (std::size_t stream = 0; stream < streams; ++stream)
        {
            typename Single::Row row;
            row.reward = Scalar(step + 1) * Scalar(0.17) - Scalar(stream) * Scalar(0.43);
            row.value = Scalar(stream + 1) * Scalar(0.21);
            row.nextValue = Scalar(step + stream) * Scalar(0.31);
            row.end = step == stream ? StepEnd::Truncated
                : (step == stream + 3 ? StepEnd::Terminated : StepEnd::Running);
            row.learnable = stream != 2 && (step + stream) % 3 != 0;
            independent[stream].Append(row);
            auto& other = team.Current(stream);
            other.reward = row.reward;
            other.value = row.value;
            other.nextValue = row.nextValue;
            other.end = row.end;
            other.learnable = row.learnable;
            expectedLearnable += row.learnable ? 1 : 0;
        }
        team.EndTeamStep();
    }
    // Deliberately keep the pre-extraction recursion here, independently of the new helper.
    for (std::size_t stream = 0; stream < streams; ++stream)
    {
        Scalar following{};
        for (std::size_t step = steps; step > 0; --step)
        {
            auto const& row = independent[stream][step - 1];
            auto const index = (step - 1) * streams + stream;
            Scalar const trace = row.end == StepEnd::Running ? following : Scalar(0);
            expectedAdvantage[index] = row.reward
                + (row.end == StepEnd::Terminated ? Scalar(0) : gamma * row.nextValue)
                - row.value + gamma * lambda * trace;
            expectedTarget[index] = row.value + expectedAdvantage[index];
            following = expectedAdvantage[index];
        }
    }
    bool identical = team.ComputeAdvantages(gamma, lambda) == expectedLearnable;
    std::size_t actualLearnable = 0;
    for (std::size_t stream = 0; stream < streams; ++stream)
    {
        actualLearnable += independent[stream].ComputeAdvantages(gamma, lambda);
        for (std::size_t step = 0; step < steps; ++step)
        {
            auto const index = step * streams + stream;
            auto const& row = independent[stream][step];
            auto const& other = team[index];
            identical = identical
                && std::memcmp(&row.advantage, &expectedAdvantage[index], sizeof(Scalar)) == 0
                && std::memcmp(&other.advantage, &expectedAdvantage[index], sizeof(Scalar)) == 0
                && std::memcmp(&row.valueTarget, &expectedTarget[index], sizeof(Scalar)) == 0
                && std::memcmp(&other.valueTarget, &expectedTarget[index], sizeof(Scalar)) == 0;
        }
    }
    Check(identical && actualLearnable == expectedLearnable,
        "interleaved and independent streams match the frozen recursion bit for bit");
    Single empty(1);
    Check(empty.ComputeAdvantages(gamma, lambda) == 0, "empty single stream remains a no-op");
}

/** @brief Both adapters retain their public size semantics, cursor and fixed row addresses. */
void SharedStorageLifecycle()
{
    RolloutBuffer<float, 1, 1> single(2);
    DynamicRolloutBuffer dynamic(2, 2, 1);
    TeamRollout<float> team(2, 2, 1, 1, 1);
    Check(single.Size() == 0 && dynamic.Size() == 0 && team.Size() == 4 && team.Collected() == 0,
        "single buffers report collected rows and team buffers report total rows");
    auto const* singleAddress = &single[0];
    auto const* teamAddress = &team.Current(0);
    auto const* dynamicAddress = dynamic[0].observation.data();
    bool rejected = false;
    try { (void)team.Current(2); }
    catch (std::out_of_range const&) { rejected = true; }
    Check(rejected, "invalid stream cannot alias the next step");
    rejected = false;
    try { (void)team.At(0, 2); }
    catch (std::out_of_range const&) { rejected = true; }
    Check(rejected, "invalid stream cannot alias a stored row");
    rejected = false;
    try { (void)team.At(2, 0); }
    catch (std::out_of_range const&) { rejected = true; }
    Check(rejected, "invalid step is rejected");
    for (int step = 0; step < 2; ++step)
    {
        single.Append({});
        dynamic.Append(dynamic[0]);
        team.EndTeamStep();
    }
    Check(single.Full() && dynamic.Full() && team.Full(), "the shared cursor fills at the exact capacity");
    single.Clear();
    dynamic.Clear();
    team.Clear();
    Check(single.Size() == 0 && dynamic.Size() == 0 && team.Collected() == 0,
        "clear resets collection without reallocating");
    Check(&single[0] == singleAddress && &team.Current(0) == teamAddress
            && dynamic[0].observation.data() == dynamicAddress,
        "row and observation addresses remain stable across reuse");
    rejected = false;
    try { TeamRollout<float> overflow(2, std::numeric_limits<std::size_t>::max(), 1, 1, 1); }
    catch (std::length_error const&) { rejected = true; }
    Check(rejected, "overflow is rejected before allocating a rollout");
}

} // namespace

int main()
{
    ScheduleMatchesReference();
    SingleAgentMatchesTeam();
    InterleavedTrajectoryParity<float>();
    InterleavedTrajectoryParity<double>();
    SharedStorageLifecycle();
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
