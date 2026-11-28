/**
 * @file PPOKit/tests/graph_learner_tests.cpp
 * @brief GraphLearner end to end: task.json + graph.json (ppokit.graph/2) -> built graphs
 *        -> Owned collection -> PPO updates, with Gaussian and Bernoulli heads.
 *
 * These are smoke tests of the training engine, not convergence claims: both toy tasks
 * are solved by any working PPO within a few updates. The frozen corridor policy is
 * evaluated on independent seeds by the corridor_smoke suite.
 */
#include "ppokit/eval/evaluator.hpp"
#include "ppokit/graph/graph_learner.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "support/corridor_task.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace {

using ppokit::core::PpoSettings;
using ppokit::runtime::RolloutCollector;
using ppokit::runtime::StepEnd;
using ppokit::runtime::StepOutcome;
using ppokit::tests::CorridorTask;

using CorridorLearner = ppokit::graph::GraphLearner<2, 1>;
using SwitchLearner = ppokit::graph::GraphLearner<2, 2>;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/** @brief Learner built from a task file and a ppokit.graph/2 file. */
template<typename Learner>
Learner LoadLearner(char const* taskPath, char const* graphPath, PpoSettings const& settings)
{
    auto const task = ppokit::schema::LoadTaskSchema(taskPath);
    auto const graphs = ppokit::schema::LoadBlockGraph(graphPath, task);
    return Learner(ppokit::schema::BuildGraph(graphs.actor), ppokit::schema::BuildGraph(graphs.critic),
        ppokit::schema::MakePolicyLayout(graphs, task), settings);
}

PpoSettings CorridorSettings()
{
    PpoSettings settings;
    settings.actorLearningRate = 3e-3;
    settings.criticLearningRate = 3e-3;
    settings.minibatch = 64;
    settings.epochs = 4;
    settings.seed = 7;
    return settings;
}

/** @brief Same corridor, same settings as the rl_tools learner test. */
void LearnsTheCorridor()
{
    CorridorTask task;
    auto learner = LoadLearner<CorridorLearner>(PPOKIT_CORRIDOR_TASK, PPOKIT_CORRIDOR_BLOCKS, CorridorSettings());
    CorridorLearner::Buffer buffer(512);
    RolloutCollector<CorridorTask, CorridorLearner, CorridorLearner::Buffer> collector(task, learner);
    collector.Collect(buffer);
    double const firstReturn = collector.Statistics().MeanReturn();
    double lastReturn = firstReturn;
    for (int update = 0; update < 30; ++update)
    {
        auto const statistics = learner.Update(buffer);
        Check(std::isfinite(statistics.policyLoss) && std::isfinite(statistics.valueLoss), "corridor: finite losses");
        collector.ResetStatistics();
        collector.Collect(buffer);
        lastReturn = collector.Statistics().MeanReturn();
    }
    int const steps = ppokit::tests::DeterministicStepsToGoal(learner);
    std::cout << "graph corridor: mean return " << firstReturn << " -> " << lastReturn
              << ", deterministic policy reaches the goal in " << steps << " steps (optimum 6), log-std "
              << learner.LogStd(0) << '\n';
    Check(lastReturn > firstReturn + 0.3, "corridor: training raises the mean episode return");
    Check(steps > 0 && steps <= 8, "corridor: the deterministic policy walks to the goal");

    auto const suite = ppokit::eval::LoadEvaluationSuite(PPOKIT_CORRIDOR_SMOKE_SUITE);
    ppokit::eval::RequireDisjointSeeds(suite, {CorridorSettings().seed});
    auto const report = ppokit::eval::Evaluate(suite, ppokit::tests::MakeCorridor, learner, CorridorTask::StepLimit);
    Check(report.Passed(), "corridor: the trained policy passes the corridor_smoke gates");
    if (!report.Passed())
    {
        std::cerr << report.ToJson() << '\n';
    }
}

/**
 * @brief One-step task needing both heads: move to `target` (Gaussian) and press the
 *        switch exactly when `signal` is positive (Bernoulli).
 * @details Reward: 1 - |position - target| for the move, +1 or -1 for the switch.
 */
class SwitchTask final
{
public:
    explicit SwitchTask(std::uint32_t seed) : rng_(seed) {}

    std::array<float, 2> Reset()
    {
        std::uniform_real_distribution<float> uniform(-1.F, 1.F);
        signal_ = uniform(rng_);
        target_ = 0.8F * uniform(rng_);
        return {signal_, target_};
    }

    StepOutcome<double> Step(std::array<float, 2> const& action, std::array<float, 2>& nextObservation)
    {
        float const position = std::clamp(action[0], -1.F, 1.F);
        bool const pressed = action[1] > 0.5F;
        bool const shouldPress = signal_ > 0.F;
        double const tracking = 1.0 - std::abs(static_cast<double>(position - target_));
        double const decision = pressed == shouldPress ? 1.0 : -1.0;
        nextObservation = {signal_, target_};
        return {tracking + decision, StepEnd::Terminated};
    }

private:
    std::mt19937 rng_;
    float signal_ = 0.F;
    float target_ = 0.F;
};

PpoSettings SwitchSettings()
{
    PpoSettings settings;
    settings.gamma = 0.0;
    // 1e-2 makes the actor unstable on this task (KL spikes); 3e-3 converges smoothly.
    settings.actorLearningRate = 3e-3;
    settings.criticLearningRate = 1e-2;
    settings.minibatch = 64;
    settings.epochs = 4;
    settings.seed = 3;
    return settings;
}

struct SwitchScore final
{
    double correctSwitch = 0;
    double meanDistance = 0;
};

/** @brief Deterministic policy on fresh draws, independent of the training seed. */
SwitchScore ScoreSwitch(SwitchLearner& learner, std::uint32_t seed)
{
    constexpr int Episodes = 400;
    SwitchTask task(seed);
    SwitchScore score;
    for (int episode = 0; episode < Episodes; ++episode)
    {
        auto const observation = task.Reset();
        auto const action = learner.Mean(observation);
        bool const pressed = action[1] > 0.5F;
        score.correctSwitch += pressed == (observation[0] > 0.F) ? 1.0 : 0.0;
        score.meanDistance += std::abs(std::clamp(action[0], -1.F, 1.F) - observation[1]);
    }
    score.correctSwitch /= Episodes;
    score.meanDistance /= Episodes;
    return score;
}

void LearnsBothHeads()
{
    SwitchTask task(SwitchSettings().seed);
    auto learner = LoadLearner<SwitchLearner>(PPOKIT_SWITCH_TASK, PPOKIT_SWITCH_GRAPH, SwitchSettings());
    SwitchScore const before = ScoreSwitch(learner, 9001);
    SwitchLearner::Buffer buffer(512);
    RolloutCollector<SwitchTask, SwitchLearner, SwitchLearner::Buffer> collector(task, learner);
    for (int update = 0; update < 40; ++update)
    {
        collector.Collect(buffer);
        auto const statistics = learner.Update(buffer);
        Check(std::isfinite(statistics.policyLoss) && std::isfinite(statistics.entropy), "switch: finite losses");
    }
    SwitchScore const after = ScoreSwitch(learner, 9001);
    std::cout << "graph switch: correct switch " << before.correctSwitch << " -> " << after.correctSwitch
              << ", mean distance " << before.meanDistance << " -> " << after.meanDistance << '\n';
    Check(after.correctSwitch >= 0.9, "switch: the Bernoulli head learns when to press");
    Check(after.meanDistance <= 0.25 && after.meanDistance < before.meanDistance,
        "switch: the Gaussian head learns to track the target");
}

void SameSeedSameTraining()
{
    auto const run = []() {
        SwitchTask task(SwitchSettings().seed);
        auto learner = LoadLearner<SwitchLearner>(PPOKIT_SWITCH_TASK, PPOKIT_SWITCH_GRAPH, SwitchSettings());
        SwitchLearner::Buffer buffer(128);
        RolloutCollector<SwitchTask, SwitchLearner, SwitchLearner::Buffer> collector(task, learner);
        collector.Collect(buffer);
        auto const statistics = learner.Update(buffer);
        return std::array<double, 3>{statistics.policyLoss, statistics.valueLoss, learner.LogStd(0)};
    };
    Check(run() == run(), "a seed fully determines weights, sampling and shuffling");
}

/** @brief Switch settings for a critic that shares the actor's trunk: one learning rate. */
PpoSettings SharedSettings()
{
    PpoSettings settings = SwitchSettings();
    settings.criticLearningRate = settings.actorLearningRate;
    settings.valueCoefficient = 0.5;
    settings.maxGradientNorm = 0.5; // As the torch PPO of RLMission and RLSNA.
    return settings;
}

/** @brief Values of every tensor, to compare two learners bit for bit. */
std::vector<std::vector<float>> TensorsOf(ppokit::graph::Graph& network)
{
    std::vector<std::vector<float>> tensors;
    for (ppokit::graph::Parameter* parameter : network.Parameters())
    {
        tensors.push_back(parameter->value);
    }
    return tensors;
}

/**
 * @brief A critic sharing the actor's trunk and a Gaussian head whose log-std depends on
 *        the state: one loss, one Adam; both heads still learn the switch, and the trunk
 *        stays one (the critic's copy follows the actor's).
 */
void SharedTrunkLearns()
{
    SwitchTask task(SharedSettings().seed);
    auto learner = LoadLearner<SwitchLearner>(PPOKIT_SWITCH_TASK, PPOKIT_SWITCH_SHARED_GRAPH, SharedSettings());
    Check(learner.Joint(), "shared trunk: the learner optimizes actor and critic together");
    SwitchScore const before = ScoreSwitch(learner, 9001);
    SwitchLearner::Buffer buffer(512);
    RolloutCollector<SwitchTask, SwitchLearner, SwitchLearner::Buffer> collector(task, learner);
    // One learning rate for both networks is slower on this task than the critic's own 1e-2.
    for (int update = 0; update < 160; ++update)
    {
        collector.Collect(buffer);
        auto const statistics = learner.Update(buffer);
        Check(std::isfinite(statistics.policyLoss) && std::isfinite(statistics.valueLoss), "shared: finite losses");
    }
    SwitchScore const after = ScoreSwitch(learner, 9001);
    std::cout << "graph shared trunk: correct switch " << before.correctSwitch << " -> " << after.correctSwitch
              << ", mean distance " << before.meanDistance << " -> " << after.meanDistance << '\n';
    Check(after.correctSwitch >= 0.9, "shared: the Bernoulli head learns when to press");
    Check(after.meanDistance <= 0.25 && after.meanDistance < before.meanDistance,
        "shared: the Gaussian head learns to track the target");
    std::vector<std::vector<float>> const actor = TensorsOf(learner.Actor());
    std::vector<std::vector<float>> const critic = TensorsOf(learner.Critic());
    Check(actor[0] == critic[0] && actor[1] == critic[1], "shared: the critic's trunk is the actor's");
    double const logStd = learner.LogStd(0);
    Check(std::isfinite(logStd) && logStd >= -5.0 && logStd <= 1.0, "shared: the computed log-std stays in bounds");
}

/** @brief A shared-trunk training saved and resumed gives the next update bit for bit. */
void SharedResumeIsExact()
{
    std::filesystem::path const folder = std::filesystem::temp_directory_path() / "ppokit_shared_resume";
    std::filesystem::create_directories(folder);
    std::filesystem::path const checkpoint = folder / "latest.ppokit";
    auto const task = ppokit::schema::LoadTaskSchema(PPOKIT_SWITCH_TASK);
    auto const graphs = ppokit::schema::LoadBlockGraph(PPOKIT_SWITCH_SHARED_GRAPH, task);
    auto const identity = ppokit::schema::ArtifactIdentityOf(task, graphs);
    auto const run = [&](SwitchLearner& learner, SwitchTask& environment) {
        SwitchLearner::Buffer buffer(256);
        RolloutCollector<SwitchTask, SwitchLearner, SwitchLearner::Buffer> collector(environment, learner);
        collector.Collect(buffer);
        learner.Update(buffer);
    };
    SwitchTask straightTask(5);
    auto straight = LoadLearner<SwitchLearner>(PPOKIT_SWITCH_TASK, PPOKIT_SWITCH_SHARED_GRAPH, SharedSettings());
    run(straight, straightTask);
    straight.SaveTraining(checkpoint, identity);
    SwitchTask resumedTask = straightTask;
    run(straight, straightTask);
    auto resumed = LoadLearner<SwitchLearner>(PPOKIT_SWITCH_TASK, PPOKIT_SWITCH_SHARED_GRAPH, SharedSettings());
    resumed.LoadTraining(checkpoint, identity);
    run(resumed, resumedTask);
    Check(TensorsOf(resumed.Actor()) == TensorsOf(straight.Actor())
            && TensorsOf(resumed.Critic()) == TensorsOf(straight.Critic()) && resumed.Updates() == 2,
        "shared: a resumed training continues bit for bit (weights, one Adam, random state)");
    std::filesystem::remove_all(folder);
}

/** @brief Two learning rates cannot drive the one optimizer of a shared trunk. */
void SharedWithTwoLearningRates()
{
    LoadLearner<SwitchLearner>(PPOKIT_SWITCH_TASK, PPOKIT_SWITCH_SHARED_GRAPH, SwitchSettings());
}

bool ThrowsWith(void (*action)(), std::string const& expected)
{
    try
    {
        action();
    }
    catch (std::exception const& error)
    {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

/** @brief A learner whose sizes disagree with task.json is refused. */
void BuildWithWrongActionSize()
{
    LoadLearner<ppokit::graph::GraphLearner<2, 3>>(PPOKIT_SWITCH_TASK, PPOKIT_SWITCH_GRAPH, SwitchSettings());
}

/** @brief The Blue attention fixture reads critic groups: the owned learner cannot feed them. */
void LayoutWithCriticGroups()
{
    auto const task = ppokit::schema::LoadTaskSchema(PPOKIT_BLUE_TASK);
    auto const graphs = ppokit::schema::LoadBlockGraph(PPOKIT_BLUE_ATTENTION_GRAPH, task);
    ppokit::schema::MakePolicyLayout(graphs, task);
}

void LayoutErrors()
{
    Check(ThrowsWith(BuildWithWrongActionSize, "some action components have no head"),
        "a missing action component is refused");
    Check(ThrowsWith(LayoutWithCriticGroups, "critic input 'own' reads a critic group"),
        "a critic group needs the hosted mode");
}

/** @brief Set inputs get their mask from the mask feature of each entity. */
void BinderMasks()
{
    ppokit::graph::InputSlice slice;
    slice.offset = 1;
    slice.shape = ppokit::graph::Shape::Set(3, 2);
    slice.maskFeature = 1;
    ppokit::graph::InputBinder binder({slice}, 7);
    std::array<float, 7> const observation{9.F, 0.3F, 1.F, 0.7F, 0.F, -0.2F, 1.F};
    auto const& signals = binder.Bind(observation.data());
    Check(signals[0].values == observation.data() + 1, "binder: values point into the observation");
    Check(signals[0].mask[0] == 1 && signals[0].mask[1] == 0 && signals[0].mask[2] == 1,
        "binder: presence read from the mask feature");
    slice.maskFeature.reset();
    bool refused = false;
    try
    {
        ppokit::graph::InputBinder invalid({slice}, 7);
    }
    catch (std::invalid_argument const&)
    {
        refused = true;
    }
    Check(refused, "binder: a set without mask feature is refused");
    bool sizeRefused = false;
    try { (void)ppokit::graph::Shape::Set(std::numeric_limits<std::size_t>::max(), 2).Size(); }
    catch (std::length_error const&) { sizeRefused = true; }
    Check(sizeRefused, "set product cannot wrap");
    slice.shape = ppokit::graph::Shape::Vector(2);
    slice.offset = std::numeric_limits<std::size_t>::max();
    refused = false;
    try { ppokit::graph::InputBinder invalid({slice}, 7); }
    catch (std::invalid_argument const&) { refused = true; }
    Check(refused, "a wrapped slice cannot pass through the binder");
}

} // namespace

/** @brief The KL guard: off by default; with a tiny target, actor steps are skipped. */
void KlGuard()
{
    auto const skips = [](double targetKl) {
        auto settings = CorridorSettings();
        settings.targetKl = targetKl;
        settings.actorLearningRate = 3e-2;
        CorridorTask task;
        auto learner = LoadLearner<CorridorLearner>(PPOKIT_CORRIDOR_TASK, PPOKIT_CORRIDOR_BLOCKS, settings);
        CorridorLearner::Buffer buffer(256);
        RolloutCollector<CorridorTask, CorridorLearner, CorridorLearner::Buffer> collector(task, learner);
        collector.Collect(buffer);
        return learner.Update(buffer).actorKlSkips;
    };
    Check(skips(0.0) == 0, "without a target KL the actor always steps");
    // The first minibatch has not moved the policy yet (KL 0) and steps; the later ones are guarded.
    Check(skips(1e-12) > 0, "a tiny target KL skips actor steps");
}

/** @brief The switch task where "position" is active for signal > 0 and "press" for target <= 0. */
SwitchLearner SwitchWithActivities(PpoSettings const& settings, bool onZero = false)
{
    std::string const text = R"({
      "schema": "ppokit.task/1", "name": "switch", "namespace": "ppokit::tests::switch_task",
      "observation": { "state": { "fields": ["signal", "target"] } },
      "actions": {
        "move": { "head": "gaussian", "names": ["position"], "bounds": [-1, 1], "map": "clip",
                  "active": { "position": { "when": "signal", "above": 0 } } },
        "press": { "head": "bernoulli", "names": ["press"],
                   "active": { "press": { "when": "target", "at_most": 0 } } }
      },
      "rewards": { "Tracking": {}, "Switch": {} }
    })";
    std::string contract = text;
    if (onZero)
    {
        contract.insert(contract.find("\"names\": [\"press\""), "\"on_zero\": true, ");
    }
    auto const task = ppokit::schema::ParseTaskSchema(contract, "task.json");
    auto const graphs = ppokit::schema::LoadBlockGraph(PPOKIT_SWITCH_GRAPH, task);
    return SwitchLearner(ppokit::schema::BuildGraph(graphs.actor), ppokit::schema::BuildGraph(graphs.critic),
        ppokit::schema::MakePolicyLayout(graphs, task), settings);
}

/** @brief Values of every actor parameter, to see whether an update moved them. */
std::vector<float> ActorValues(SwitchLearner& learner)
{
    std::vector<float> values;
    for (ppokit::graph::Parameter const* parameter : learner.Actor().Parameters())
    {
        values.insert(values.end(), parameter->value.begin(), parameter->value.end());
    }
    return values;
}

/** @brief Only opted-in Bernoulli heads choose one on a zero logit, and only when active. */
void BernoulliTies()
{
    for (bool onZero : {false, true})
    {
        auto learner = SwitchWithActivities(SwitchSettings(), onZero);
        for (auto* parameter : learner.Actor().Parameters())
        {
            std::fill(parameter->value.begin(), parameter->value.end(), 0.F);
        }
        auto& bias = learner.Actor().Parameters().back()->value;
        Check(bias.size() == 1, "switch last parameter is the Bernoulli bias");
        for (float logit : {-1.F, 0.F, 1.F})
        {
            bias[0] = logit;
            bool const pressed = logit > 0.F || (logit == 0.F && onZero);
            Check(learner.Mean({1.F, -1.F})[1] == static_cast<float>(pressed), "active Bernoulli threshold");
            Check(learner.Mean({1.F, 1.F})[1] == 0.F, "masked Bernoulli stays zero for either tie rule");
        }
    }
}

/** @brief Rejected updates preserve both the collection and the next optimizer/RNG result. */
void IncompleteRollouts()
{
    SwitchLearner subject = SwitchWithActivities(SwitchSettings());
    SwitchLearner reference = SwitchWithActivities(SwitchSettings());
    SwitchLearner::Buffer buffer(2);
    SwitchLearner::Buffer expected(2);
    SwitchLearner::Buffer::Row row;
    row.reward = 1.0;
    for (std::size_t filled = 0; filled < buffer.Capacity(); ++filled)
    {
        bool rejected = false;
        try
        {
            (void)subject.Update(buffer);
        }
        catch (std::invalid_argument const&)
        {
            rejected = true;
        }
        Check(rejected && subject.Updates() == 0 && buffer.Size() == filled,
            "empty and partial updates preserve their buffer and counter");
        Check(ActorValues(subject) == ActorValues(reference), "a rejected update preserves the actor");
        buffer.Append(row);
        expected.Append(row);
        row.reward = -1.0;
    }
    auto const actual = subject.Update(buffer);
    auto const control = reference.Update(expected);
    Check(actual.updates == 1 && actual.valueLoss == control.valueLoss && buffer.Size() == 0
            && ActorValues(subject) == ActorValues(reference),
        "the completed rollout follows the uninterrupted update");
}

/** @brief Two terminal samples; one optionally produces a finite loss gradient whose Adam variance overflows. */
SwitchLearner::Buffer TwoSampleRollout(int overflowRow = -1)
{
    SwitchLearner::Buffer buffer(2);
    for (int index = 0; index < 2; ++index)
    {
        SwitchLearner::Buffer::Row row;
        row.observation = {0.25F, -0.5F};
        row.action = {-0.25F, 1.F};
        row.reward = index == 0 ? 1.0 : -0.5;
        if (index == overflowRow)
        {
            row.reward = 1e30;
        }
        row.end = StepEnd::Terminated;
        buffer.Append(row);
    }
    return buffer;
}

/** @brief Read the serialized weights, moments, optimizer ages and RNG, excluding the timestamped header. */
template<typename Learner>
std::string TrainingPayload(Learner& learner, ppokit::artifacts::ArtifactIdentity const& identity,
    std::filesystem::path const& path)
{
    learner.SaveTraining(path, identity);
    std::ifstream input(path, std::ios::binary);
    std::string header;
    if (!std::getline(input, header))
    {
        throw std::runtime_error("missing training checkpoint header");
    }
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

/** @brief Runtime-sized and fixed adapters must produce the same decisions, update and checkpoint. */
template<typename Fixed>
void RuntimeDimensionsMatch(char const* taskPath, char const* graphPath, PpoSettings const& settings)
{
    auto const task = ppokit::schema::LoadTaskSchema(taskPath);
    auto const graphs = ppokit::schema::LoadBlockGraph(graphPath, task);
    auto const identity = ppokit::schema::ArtifactIdentityOf(task, graphs);
    auto fixed = LoadLearner<Fixed>(taskPath, graphPath, settings);
    typename Fixed::Observation observation{};
    typename Fixed::Action action{};
    ppokit::graph::GraphLearnerCore dynamic(ppokit::schema::BuildGraph(graphs.actor),
        ppokit::schema::BuildGraph(graphs.critic), ppokit::schema::MakePolicyLayout(graphs, task),
        settings, task.observation.size, task.actionSize);
    typename Fixed::Buffer fixedBuffer(4);
    ppokit::runtime::DynamicRolloutBuffer dynamicBuffer(4, task.observation.size, task.actionSize);
    ppokit::runtime::DynamicTransition dynamicRow;
    dynamicRow.observation.resize(observation.size());
    dynamicRow.action.resize(action.size());
    for (std::size_t index = 0; index < 4; ++index)
    {
        observation[0] = static_cast<float>(index) * 0.125F;
        observation[1] = -0.5F;
        dynamic.Mean(observation.data(), action.data());
        Check(action == fixed.Mean(observation), "runtime-sized deterministic action matches");
        auto const decision = fixed.Decide(observation);
        double logProbability = 0.0;
        double value = 0.0;
        dynamic.Decide(observation.data(), action.data(), logProbability, value);
        Check(action == decision.action && logProbability == decision.logProbability && value == decision.value,
            "runtime-sized sampled action, likelihood and value match");
        typename Fixed::Buffer::Row row;
        row.observation = observation;
        row.action = action;
        row.logProbability = logProbability;
        row.value = value;
        row.nextValue = value;
        row.reward = static_cast<double>(index) - 1.0;
        row.end = index == 1 ? StepEnd::Terminated : (index == 3 ? StepEnd::Truncated : StepEnd::Running);
        fixedBuffer.Append(row);
        std::copy(observation.begin(), observation.end(), dynamicRow.observation.begin());
        std::copy(action.begin(), action.end(), dynamicRow.action.begin());
        dynamicRow.logProbability = row.logProbability;
        dynamicRow.value = row.value;
        dynamicRow.nextValue = row.nextValue;
        dynamicRow.reward = row.reward;
        dynamicRow.end = row.end;
        dynamicBuffer.Append(dynamicRow);
    }
    auto const expected = fixed.Update(fixedBuffer);
    auto const actual = dynamic.Update(dynamicBuffer);
    Check(expected.policyLoss == actual.policyLoss && expected.valueLoss == actual.valueLoss
            && expected.entropy == actual.entropy && expected.explainedVariance == actual.explainedVariance
            && expected.approximateKl == actual.approximateKl && expected.updates == actual.updates,
        "runtime-sized rollout preserves PPO statistics");
    auto const path = std::filesystem::temp_directory_path() / "ppokit-runtime-dimensions.ppokit";
    auto const payload = TrainingPayload(fixed, identity, path);
    Check(payload == TrainingPayload(dynamic, identity, path), "runtime-sized update preserves all checkpoint tensors and RNG");
    fixed.LoadTraining(path, identity);
    std::filesystem::remove(path);
    dynamicRow.action.clear();
    bool rejected = false;
    try { dynamicBuffer.Append(dynamicRow); }
    catch (std::invalid_argument const&) { rejected = true; }
    Check(rejected && dynamicBuffer.Size() == 0, "runtime-sized rollout rejects mismatched dimensions before append");
}

/** @brief Even a failure in the last minibatch must allow an exact retry for joint and independent Adam. */
void FailedUpdateIsRetryable(char const* graphPath)
{
    auto settings = SharedSettings();
    settings.minibatch = 1;
    settings.epochs = 1;
    settings.maxGradientNorm = 0.0;
    auto const task = ppokit::schema::LoadTaskSchema(PPOKIT_SWITCH_TASK);
    auto const graph = ppokit::schema::LoadBlockGraph(graphPath, task);
    auto const identity = ppokit::schema::ArtifactIdentityOf(task, graph);
    auto subject = LoadLearner<SwitchLearner>(PPOKIT_SWITCH_TASK, graphPath, settings);
    auto control = LoadLearner<SwitchLearner>(PPOKIT_SWITCH_TASK, graphPath, settings);
    auto warm = TwoSampleRollout();
    auto controlWarm = warm;
    subject.Update(warm);
    control.Update(controlWarm);
    auto const tick = std::chrono::steady_clock::now().time_since_epoch().count();
    auto const path = std::filesystem::temp_directory_path() / ("ppokit-rollback-" + std::to_string(tick) + ".ppokit");
    auto const before = TrainingPayload(control, identity, path);
    ppokit::graph::GraphProbes probes(subject.Actor());
    subject.SetProbes(&probes, nullptr);

    // Each failed attempt starts at the same shuffle. One of the two bad rows is necessarily visited last.
    for (int badRow : {0, 1})
    {
        auto bad = TwoSampleRollout(badRow);
        bool rejected = false;
        try
        {
            subject.Update(bad);
        }
        catch (std::runtime_error const& error)
        {
            rejected = std::string(error.what()).find("non-finite optimizer step") != std::string::npos;
        }
        Check(rejected && bad.Full() && subject.Updates() == 1, "a failed optimizer step retains the full rollout");
        Check(TrainingPayload(subject, identity, path) == before, "a failed update restores the complete training state");
        for (auto const& block : probes.Summary())
        {
            Check(block.samples == 0, "a failed update discards partial probe measurements");
        }
    }
    auto corrected = TwoSampleRollout();
    auto expected = corrected;
    subject.Update(corrected);
    control.Update(expected);
    Check(subject.Updates() == 2 && !corrected.Full()
            && TrainingPayload(subject, identity, path) == TrainingPayload(control, identity, path),
        "the corrected rollout reproduces the uninterrupted training state");
    std::filesystem::remove(path);
}

/**
 * @brief Inactive components (task.json "active") are 0, draw no random number, add nothing
 *        to the log-probability, and an update on inactive samples leaves the actor as it is.
 */
void ActivitiesMaskComponents()
{
    std::array<float, 2> const inactive{-0.5F, 0.3F};
    std::array<float, 2> const active{0.5F, -0.3F};
    SwitchLearner masked = SwitchWithActivities(SwitchSettings());
    SwitchLearner reference = SwitchWithActivities(SwitchSettings());
    auto const none = masked.Decide(inactive);
    Check(none.action[0] == 0.F && none.action[1] == 0.F, "inactive components are 0");
    Check(none.logProbability == 0.0, "inactive components add nothing to the log-probability");
    auto const after = masked.Decide(active);
    auto const first = reference.Decide(active);
    Check(after.action == first.action && after.logProbability == first.logProbability,
        "an inactive component draws no random number");
    Check(first.logProbability != 0.0, "active components are scored");
    auto const deterministic = masked.Mean(inactive);
    Check(deterministic[0] == 0.F && deterministic[1] == 0.F, "the deterministic action is 0 when inactive");

    std::vector<float> const before = ActorValues(masked);
    SwitchLearner::Buffer buffer(128);
    while (!buffer.Full())
    {
        auto const decision = masked.Decide(inactive);
        SwitchLearner::Buffer::Row row;
        row.observation = inactive;
        row.action = decision.action;
        row.logProbability = decision.logProbability;
        row.value = decision.value;
        row.reward = 1.0;
        row.end = StepEnd::Terminated;
        buffer.Append(row);
    }
    masked.Update(buffer);
    Check(ActorValues(masked) == before, "samples whose components are all inactive do not move the actor");
}

/** @brief Whether a task whose "move" group has this "active" section is refused with `expected`. */
bool ActivityRefused(std::string const& active, std::string const& expected)
{
    std::string const text = R"({ "schema": "ppokit.task/1", "name": "t",
      "observation": { "state": { "fields": ["signal"] } },
      "actions": { "move": { "head": "gaussian", "names": ["position"], "active": )"
        + active + R"( } },
      "rewards": { "R": {} } })";
    try
    {
        ppokit::schema::ParseTaskSchema(text, "task.json");
    }
    catch (ppokit::schema::SchemaError const& error)
    {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

/** @brief "active" errors name the place: unknown component, unknown feature, no test. */
void ActivityErrors()
{
    Check(ActivityRefused(R"({ "speed": { "when": "signal", "above": 0 } })", "not a component of this group"),
        "an unknown component is reported");
    Check(ActivityRefused(R"({ "position": { "when": "missing", "above": 0 } })", "'missing' is not a named feature"),
        "an unknown feature is reported");
    Check(ActivityRefused(R"({ "position": { "when": "signal" } })", "give either"), "a missing test is reported");
    Check(ActivityRefused(R"({ "position": { "when": "signal", "above": 0, "at_most": 1 } })", "give either"),
        "two tests are reported");
}

int main()
{
    try
    {
        RuntimeDimensionsMatch<CorridorLearner>(PPOKIT_CORRIDOR_TASK, PPOKIT_CORRIDOR_BLOCKS, CorridorSettings());
        RuntimeDimensionsMatch<SwitchLearner>(PPOKIT_SWITCH_TASK, PPOKIT_SWITCH_GRAPH, SwitchSettings());
        RuntimeDimensionsMatch<SwitchLearner>(PPOKIT_SWITCH_TASK, PPOKIT_SWITCH_SHARED_GRAPH, SharedSettings());
        BinderMasks();
        LayoutErrors();
        LearnsTheCorridor();
        KlGuard();
        LearnsBothHeads();
        SameSeedSameTraining();
        SharedTrunkLearns();
        SharedResumeIsExact();
        Check(ThrowsWith(&SharedWithTwoLearningRates, "set critic_learning_rate to actor_learning_rate"),
            "shared: one optimizer, one learning rate");
        ActivitiesMaskComponents();
        IncompleteRollouts();
        FailedUpdateIsRetryable(PPOKIT_SWITCH_GRAPH);
        FailedUpdateIsRetryable(PPOKIT_SWITCH_SHARED_GRAPH);
        BernoulliTies();
        ActivityErrors();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0)
    {
        std::cerr << failures << " graph learner check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit graph learner tests passed\n";
    return EXIT_SUCCESS;
}
