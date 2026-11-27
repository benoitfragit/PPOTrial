/**
 * @file master_checkpoint_parity_tests.cpp
 * @brief The T-6 checkpoint of projects/T6 (PPOKit graph engine) is master's trained T-6
 *        (t6texan2_joint_v5, update 870, FlightLearner): same actor means, log-std, gear
 *        logit and critic value, and the same deterministic flight on evaluation episodes.
 * @details Audit P0-04. The master checkpoint is kept as a fixture
 * (fixtures/master_t6texan2_joint_v5_flight_policy.txt, master's latest = its BEST
 * flight_870_092d842f035af06e, 19/20 stops). Master side: FlightLearner loads it (actor,
 * log-std, gear logit); its critic is read into the rl_tools critic that graph.json generates,
 * which network_parity_tests proves is FlightLearner's. PPOKit side: the graph learner of
 * projects/T6 (task.json, network.json) loads checkpoints/latest.ppokit. Compared:
 * - probe observations spread over [-0.9, 0.9];
 * - every step of deterministic evaluation episodes (master's EvaluationStart at the
 *   checkpoint's saved level), flown by master's actions: at each step both policies see the
 *   same observation, and their outputs must agree; the episode outcomes are those of
 *   master's policy, so the PPOKit policy, giving the same actions, flies them too.
 */
#include "flight_curriculum.hpp"
#include "flight_environment.hpp"
#include "flight_learner.hpp"
#include "flight_rng_checkpoint.hpp"
#include "test_support.hpp"

#include "ppokit/graph/graph_learner.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/task_schema.hpp"
#include "ppokit/schema/train_schema.hpp"

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include "flight_networks.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

using namespace ppo::examples::flight;

namespace {

namespace rlt = rl_tools;
namespace generated = task::networks;
namespace fs = std::filesystem;
using Device = rlt::devices::DEVICE_FACTORY<>;
using Index = typename Device::index_t;
constexpr Index SingleSample = 1;
using ActorNetwork = generated::Actor::Network<Index, SingleSample, generated::Actor::DefaultWidth, rlt::MatrixStaticTag>;
using CriticNetwork =
    generated::Critic::Network<Index, SingleSample, generated::Critic::DefaultWidth, rlt::MatrixStaticTag>;
template<Index Columns>
using Row = rlt::MatrixStatic<rlt::matrix::Specification<float, Index, 1, Columns>>;
using Graph = ppokit::graph::GraphLearner<ObservationSize, ContinuousActions + 1>;

/** @brief Tolerance of an output, relative to max(1, |master|): float summation order only. */
constexpr double Tolerance = 1e-4;
/** @brief Evaluation episodes flown (of master's 20) and their step budget. */
constexpr std::size_t Episodes = 20;
constexpr std::size_t EpisodesOfCampaign = 20;
constexpr std::size_t MaximumSteps = 12000;

double worst = 0.0;

bool Close(double actual, double expected) noexcept
{
    double const error = std::abs(actual - expected) / std::max(1.0, std::abs(expected));
    worst = std::max(worst, error);
    return error <= Tolerance;
}

template<typename Matrix>
void ReadMatrix(std::istream& stream, Matrix& matrix)
{
    for (Index row = 0; row < Matrix::ROWS; ++row)
    {
        for (Index column = 0; column < Matrix::COLS; ++column)
        {
            float value{};
            Check(static_cast<bool>(stream >> value), "the master checkpoint fills the generated networks");
            rlt::set(matrix, row, column, value);
        }
    }
}

template<typename Parameter>
void ReadParameter(std::istream& stream, Parameter& parameter)
{
    ReadMatrix(stream, parameter.parameters);
    ReadMatrix(stream, parameter.gradient_first_order_moment);
    ReadMatrix(stream, parameter.gradient_second_order_moment);
}

template<typename Network>
void ReadNetwork(std::istream& stream, Network& network)
{
    ReadParameter(stream, network.input_layer.weights);
    ReadParameter(stream, network.input_layer.biases);
    for (auto& layer : network.hidden_layers)
    {
        ReadParameter(stream, layer.weights);
        ReadParameter(stream, layer.biases);
    }
    ReadParameter(stream, network.output_layer.weights);
    ReadParameter(stream, network.output_layer.biases);
}

/** @brief Master's critic, read from its checkpoint into the generated rl_tools critic. */
std::unique_ptr<CriticNetwork> MasterCritic(fs::path const& checkpoint, std::uint64_t expectedUpdate)
{
    std::ifstream stream(checkpoint);
    std::string magic;
    std::size_t observations{};
    std::size_t actions{};
    std::size_t outputs{};
    std::size_t width{};
    std::uint64_t updates{};
    std::string signature;
    std::uint64_t actorAge{};
    std::uint64_t criticAge{};
    stream >> magic >> observations >> actions >> outputs >> width >> updates >> std::quoted(signature) >> actorAge
        >> criticAge;
    Check(static_cast<bool>(stream) && updates == expectedUpdate, "the historical checkpoint has the expected update");
    detail::ReadCheckpointRng(stream);
    auto actor = std::make_unique<ActorNetwork>();
    auto critic = std::make_unique<CriticNetwork>();
    ReadNetwork(stream, *actor);
    ReadParameter(stream, actor->log_std);
    ReadNetwork(stream, *critic);
    stream >> std::ws;
    Check(stream.eof(), "the master checkpoint is read to its end");
    return critic;
}

double CriticValue(CriticNetwork& critic, Observation const& observation)
{
    Device device;
    Row<generated::Critic::Inputs> input;
    Row<1> output;
    for (Index index = 0; index < generated::Critic::Inputs; ++index)
    {
        rlt::set(input, 0, index, observation[index]);
    }
    rlt::forward(device, critic, input, output);
    return static_cast<double>(rlt::get(output, 0, 0));
}

/** @brief The graph learner of projects/T6 with its checkpoint loaded. */
std::unique_ptr<Graph> LoadGraphLearner(fs::path const& project, fs::path const& weights, std::uint64_t expectedUpdate,
    ppokit::core::PpoSettings const& settings = {})
{
    ppokit::schema::TaskSchema const task = ppokit::schema::LoadTaskSchema(project / "task.json");
    // The networks of network.json; its observers (the Visualiseur DLL) change nothing of them.
    std::ifstream file(project / "network.json");
    nlohmann::ordered_json network = nlohmann::ordered_json::parse(file);
    network.erase("observers");
    network.erase("components");
    ppokit::schema::BlockGraphSchema const graphs =
        ppokit::schema::ParseBlockGraph(network.dump(), "network.json", task);
    auto learner = std::make_unique<Graph>(ppokit::schema::BuildGraph(graphs.actor),
        ppokit::schema::BuildGraph(graphs.critic), ppokit::schema::MakePolicyLayout(graphs, task),
        settings);
    learner->LoadTraining(weights / "latest.ppokit", ppokit::schema::ArtifactIdentityOf(task, graphs));
    Check(learner->Updates() == expectedUpdate, "the PPOKit checkpoint has the expected update");
    return learner;
}

/** @brief Both policies on one observation: means, gear logit, log-std, value. */
bool SameOutputs(FlightLearner& master, CriticNetwork& critic, Graph& ppokit, Observation const& observation)
{
    PolicyDistribution const expected = master.Evaluate(observation);
    typename Graph::Observation input{};
    std::copy(observation.begin(), observation.end(), input.begin());
    // Mean runs the actor; its head outputs are compared raw: Mean zeroes the brake while it
    // is inactive (in flight), where the environment ignores it, as FlightLearner's Sample does.
    (void)ppokit.Mean(input);
    float const* const means = ppokit.Actor().Output(0).values;
    bool same = true;
    for (std::size_t axis = 0; axis < ContinuousActions; ++axis)
    {
        same = Close(means[axis], expected.output[axis]) && same;
        same = Close(ppokit.LogStd(axis), expected.logStandardDeviations[axis]) && same;
    }
    // The gear head's logit (master's last output).
    same = Close(ppokit.Actor().Output(1).values[0], expected.output[ContinuousActions]) && same;
    same = Close(ppokit.Value(input), CriticValue(critic, observation)) && same;
    return same;
}

Observation Probe(int seed)
{
    Observation observation{};
    for (std::size_t index = 0; index < observation.size(); ++index)
    {
        auto const phase = static_cast<double>(seed * 31 + static_cast<int>(index) * 7);
        observation[index] = static_cast<float>(0.9 * std::sin(phase));
    }
    return observation;
}

/** @brief One update from identical stored weights, optimizer and shuffle stream on fixed transition rows. */
void FrozenUpdate(fs::path const& project, fs::path const& fixture, fs::path const& weights, fs::path const& output)
{
    constexpr std::size_t samples = 64;
    Config config = LoadConfig(project / "scenario.json");
    auto settings = ppokit::schema::LoadTrainingPlan(project / "training.json",
        ppokit::schema::LoadTaskSchema(project / "task.json")).ppo.Settings();
    config.training.rolloutSteps = samples;
    config.training.minibatch = settings.minibatch = 16;
    FlightLearner legacy(config);
    legacy.Load(fixture);
    auto const update = legacy.Stats().updates;
    auto const graph = LoadGraphLearner(project, weights, update, settings);
    Graph::Buffer buffer(samples);
    for (std::size_t index = 0; index < samples; ++index)
    {
        auto const observation = Probe(static_cast<int>(index));
        auto const next = Probe(static_cast<int>(index + 1));
        auto action = legacy.Choose(observation, true);
        Graph::Buffer::Row row;
        row.observation = observation;
        for (std::size_t axis = 0; axis < ContinuousActions; ++axis)
        {
            float const latent = static_cast<float>(action.raw[axis] + 0.04 * std::sin(index + axis));
            row.action[axis] = latent;
            action.raw[axis] = latent;
        }
        if (action.gearActive && index % 5 == 0) action.gearDown = !action.gearDown;
        row.action[GearOutput] = action.gearDown ? 1.F : 0.F;
        action.logProbability = LogProbability(action.policy, action);
        row.logProbability = action.logProbability;
        StepResult step;
        step.reward = 0.2 * std::cos(static_cast<double>(index));
        step.terminated = index == 17;
        step.truncated = index == 41;
        row.reward = step.reward;
        row.value = graph->Value(observation);
        row.nextValue = step.terminated ? 0.0 : graph->Value(next);
        row.end = step.terminated ? ppokit::runtime::StepEnd::Terminated :
            (step.truncated ? ppokit::runtime::StepEnd::Truncated : ppokit::runtime::StepEnd::Running);
        legacy.Record(observation, action, step, next);
        buffer.Append(row);
    }
    legacy.Optimize();
    auto const stats = graph->Update(buffer);
    Check(legacy.Stats().updates == update + 1 && stats.updates == update + 1, "both engines perform one update");
    Check(legacy.Stats().actorSteps > 0 && legacy.Stats().criticSteps == samples / settings.minibatch * settings.epochs,
        "the fixture exercised both optimizers");
    fs::create_directories(output);
    legacy.Save(output / "flight.txt");
    auto const header = ppokit::artifacts::ReadArtifactHeader(weights / "latest.ppokit");
    graph->SaveTraining(output / "latest.ppokit", header.identity);
    std::cout << std::setprecision(17) << "Frozen update, critic loss (legacy / graph): "
              << legacy.Stats().valueLoss << " / " << stats.valueLoss << '\n';
}

void MasterCheckpointParity(fs::path const& project, fs::path const& fixture, fs::path const& weights,
    std::uint64_t update, double expectedStops, double expectedTouchdowns)
{
    Config const config = LoadConfig(project / "scenario.json");
    FlightLearner master(config);
    master.Load(fixture);
    Check(master.Stats().updates == update, "historical learner loads the expected update");
    std::unique_ptr<CriticNetwork> const critic = MasterCritic(fixture, update);
    std::unique_ptr<Graph> const ppokit = LoadGraphLearner(project, weights, update);

    bool probes = true;
    for (int seed = 0; seed < 64; ++seed)
    {
        probes = SameOutputs(master, *critic, *ppokit, Probe(seed)) && probes;
    }
    Check(probes, "probe observations: same means, log-std, gear logit and value");

    // Master's evaluation episodes at the checkpoint's level (11 of the curriculum).
    FlightCurriculum curriculum(config.training.curriculum);
    std::ifstream level(weights / "ppokit_curriculum_level.txt");
    std::size_t savedLevel = 0;
    Check(static_cast<bool>(level >> savedLevel), "read the archived curriculum level");
    curriculum.Restore(savedLevel);
    FlightEnvironment environment(config);
    std::size_t steps = 0;
    std::size_t stops = 0;
    std::size_t touchdowns = 0;
    double stopSum = 0.0;
    double touchdownSum = 0.0;
    bool flown = true;
    for (std::size_t episode = 0; episode < Episodes; ++episode)
    {
        EpisodeStart const start = EvaluationStart(
            episode, EpisodesOfCampaign, config.training.curriculum.rolloutStartShare, curriculum.Difficulty());
        Observation observation = environment.Reset(static_cast<std::uint32_t>(4342 + episode), start);
        for (std::size_t step = 0; step < MaximumSteps; ++step)
        {
            flown = SameOutputs(master, *critic, *ppokit, observation) && flown;
            PolicyAction const action = master.Choose(observation, true);
            auto graphAction = action;
            auto const raw = ppokit->Mean(observation);
            for (std::size_t i = 0; i < ContinuousActions; ++i)
            {
                graphAction.raw[i] = raw[i];
            }
            graphAction.gearDown = raw[ContinuousActions] != 0.F;
            auto const expected = ToControls(action, config.initial);
            auto const actual = ToControls(graphAction, config.initial);
            Check(Applied(expected) == Applied(actual) && expected.gear == actual.gear,
                "both policies produce the same physical commands on the flight");
            StepResult const result = environment.Step(expected);
            observation = environment.ObservationNow();
            ++steps;
            if (result.terminated || result.truncated)
            {
                break;
            }
        }
        MissionState const& mission = environment.Mission();
        if (mission.outcome == Outcome::Stopped)
        {
            ++stops;
            stopSum += mission.stopFromThreshold;
        }
        if (mission.touchedDown)
        {
            ++touchdowns;
            touchdownSum += mission.touchdownFromThreshold;
        }
    }
    Check(flown, "every step of master's evaluation flights: same outputs");
    std::cout << steps << " steps of " << Episodes << " evaluation episodes, " << stops
              << " stops; largest relative difference " << worst << '\n';
    Check(steps > 1000, "the evaluation episodes are flown");
    // Master's BEST record of this checkpoint (best/current.json): 19 stops and 19 touchdowns
    // of 20, their distances past the threshold summed. The record was written by the Windows
    // CI: JSBSim's arithmetic differs in the last bits between compilers, hence 1 cm on sums of
    // 19 distances (Linux gives 0.25 mm and 0.002 mm of difference).
    std::cout << std::setprecision(17) << "stops " << stops << " (" << stopSum << " m), touchdowns " << touchdowns
              << " (" << touchdownSum << " m)\n";
    Check(stops == 19 && touchdowns == 19, "the campaign of master's BEST record: 19 stops, 19 touchdowns");
    Check(Near(stopSum, expectedStops, 1e-2) && Near(touchdownSum, expectedTouchdowns, 1e-2),
        "the stop and touchdown distances of master's BEST record");
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if (argc == 1)
        {
            fs::path const project(PPO_FLIGHT_T6_PROJECT);
            MasterCheckpointParity(project, PPO_FLIGHT_MASTER_CHECKPOINT, project / "checkpoints", 870,
                10390.561858707588, 1083.373831656396);
        }
        else if (argc == 6 && std::string(argv[1]) == "--frozen")
        {
            FrozenUpdate(argv[2], argv[3], argv[4], argv[5]);
        }
        else if (argc == 5)
        {
            std::ifstream manifest(argv[4]);
            auto const best = nlohmann::json::parse(manifest);
            MasterCheckpointParity(argv[1], argv[2], argv[3], best.at("update").get<std::uint64_t>(),
                best.at("stopSumMeters").get<double>(), best.at("touchdownSumMeters").get<double>());
        }
        else
        {
            throw std::invalid_argument("expected PROJECT V3_FILE CHECKPOINT_DIRECTORY BEST_MANIFEST");
        }
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "Flight checkpoint matches the archived campaign\n";
    return 0;
}
