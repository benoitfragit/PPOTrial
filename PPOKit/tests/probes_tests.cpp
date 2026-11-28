/**
 * @file PPOKit/tests/probes_tests.cpp
 * @brief Network probes: they never change training, they measure every block (present
 *        entities only for sets), the journal keeps them, and the studio reads them in
 *        plain words.
 */
#include "ppokit/editor/probe_series.hpp"
#include "ppokit/graph/graph_learner.hpp"
#include "ppokit/graph/probes.hpp"
#include "ppokit/runtime/run_log.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/train_schema.hpp"
#include "support/corridor_task.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>

namespace {

using namespace ppokit;
using CorridorLearner = graph::GraphLearner<2, 1>;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

CorridorLearner MakeLearner()
{
    auto const task = schema::LoadTaskSchema(PPOKIT_CORRIDOR_TASK);
    auto const graphs = schema::LoadBlockGraph(PPOKIT_CORRIDOR_BLOCKS, task);
    core::PpoSettings settings;
    settings.actorLearningRate = 3e-3;
    settings.criticLearningRate = 3e-3;
    settings.epochs = 3;
    settings.seed = 11;
    return CorridorLearner(schema::BuildGraph(graphs.actor), schema::BuildGraph(graphs.critic),
        schema::MakePolicyLayout(graphs, task), settings);
}

/** @brief Two identical learners, one probed: same statistics and same policy afterwards. */
void ProbesNeverChangeTraining()
{
    tests::CorridorTask firstTask;
    tests::CorridorTask secondTask;
    auto plain = MakeLearner();
    auto probed = MakeLearner();
    graph::GraphProbes actorProbes(probed.Actor());
    graph::GraphProbes criticProbes(probed.Critic());
    probed.SetProbes(&actorProbes, &criticProbes);
    CorridorLearner::Buffer firstBuffer(256);
    CorridorLearner::Buffer secondBuffer(256);
    runtime::RolloutCollector<tests::CorridorTask, CorridorLearner, CorridorLearner::Buffer> first(firstTask, plain);
    runtime::RolloutCollector<tests::CorridorTask, CorridorLearner, CorridorLearner::Buffer> second(secondTask, probed);
    bool same = true;
    for (int update = 0; update < 3; ++update)
    {
        first.Collect(firstBuffer);
        second.Collect(secondBuffer);
        auto const a = plain.Update(firstBuffer);
        auto const b = probed.Update(secondBuffer);
        same = same && a.policyLoss == b.policyLoss && a.valueLoss == b.valueLoss && a.approximateKl == b.approximateKl;
    }
    auto const observation = CorridorLearner::Observation{0.3F, -0.2F};
    Check(same && plain.Mean(observation) == probed.Mean(observation), "probes never change the training");

    auto const actor = actorProbes.Summary();
    Check(!actor.empty(), "every actor block is probed");
    bool measured = true;
    for (auto const& block : actor)
    {
        measured = measured && block.samples > 0 && std::isfinite(block.mean) && block.deviation >= 0.0;
        if (block.hasGradient)
        {
            measured = measured && block.gradientNorm > 0.0;
        }
    }
    Check(measured, "outputs and gradients are measured during the probed update");
    Check(!criticProbes.Summary().empty() && criticProbes.Summary().back().samples > 0, "the critic is probed too");
}

/** @brief Sets: only present entities count. */
void SetsCountPresentEntities()
{
    graph::Graph network;
    auto const input = network.AddInput("tracks", graph::Shape::Set(3, 2));
    auto const pool = network.AddNode("pool", std::make_unique<graph::MaskedPoolNode>(3, 2, graph::PoolMode::Mean),
        {input});
    network.AddOutput(pool);
    graph::GraphProbes probes(network);
    float const values[6] = {1.F, 1.F, 5.F, 5.F, 0.F, 0.F};
    std::uint8_t const mask[3] = {1, 0, 0};
    network.Forward({graph::Signal{values, mask}});
    probes.ObserveForward();
    auto const summary = probes.Summary();
    Check(summary.size() == 1 && summary[0].block == "pool" && summary[0].samples == 2 && summary[0].mean == 1.0,
        "a vector output after pooling: its two values");
    Check(!summary[0].hasGradient, "a block without parameters has no gradient size");

    graph::Graph encoder;
    auto const tracks = encoder.AddInput("tracks", graph::Shape::Set(3, 2));
    auto const encoded = encoder.AddNode("encoder",
        std::make_unique<graph::EntityEncoderNode>(3, 2, 4, graph::Activation::Relu), {tracks});
    encoder.AddOutput(encoded);
    std::mt19937 rng(3);
    encoder.Initialize(rng);
    graph::GraphProbes encoderProbes(encoder);
    encoder.Forward({graph::Signal{values, mask}});
    encoderProbes.ObserveForward();
    Check(encoderProbes.Summary()[0].samples == 4, "a set output: one present entity x 4 features");
}

void JournalAndDiagnostics()
{
    std::filesystem::path const path = std::filesystem::temp_directory_path() / "ppokit_probes_tests.jsonl";
    runtime::RunHeader header;
    header.task = "corridor";
    header.channels = {{"reward", "Reward", 1.0}};
    {
        runtime::RunLogWriter writer(path, header);
        runtime::BlockProbe healthy;
        healthy.block = "trunk";
        healthy.type = "mlp";
        healthy.samples = 100;
        healthy.deviation = 0.4;
        healthy.deadShare = 0.1;
        healthy.hasGradient = true;
        healthy.gradientNorm = 0.05;
        runtime::BlockProbe pooled = healthy;
        pooled.block = "pool";
        pooled.hasGradient = false;
        writer.WriteProbes({10, "actor", {healthy, pooled}});
        healthy.deadShare = 0.9;
        healthy.gradientNorm = 0.0;
        writer.WriteProbes({20, "actor", {healthy, pooled}});
    }
    auto const log = runtime::LoadRunLog(path);
    Check(log.probes.size() == 2 && log.probes[1].blocks[0].deadShare == 0.9 && !log.probes[0].blocks[1].hasGradient,
        "the journal keeps the probes");
    Check(editor::ProbeNetworks(log) == std::vector<std::string>{"actor"}
            && editor::ProbeBlocks(log, "actor") == std::vector<std::string>{"trunk", "pool"},
        "networks and blocks in graph order");
    auto const dead = editor::ProbeSeries(log, "actor", "trunk", editor::ProbeStatistic::DeadShare);
    Check(dead.updates == std::vector<double>{10.0, 20.0} && dead.values[1] == 0.9, "a curve per block");
    auto const gradient = editor::ProbeSeries(log, "actor", "pool", editor::ProbeStatistic::GradientNorm);
    Check(std::isnan(gradient.values[0]), "no gradient size for a block without parameters: a gap, not a zero");
    auto const remarks = editor::CheckProbes(log, "actor");
    bool deadRemark = false;
    bool vanishingRemark = false;
    for (auto const& remark : remarks)
    {
        deadRemark = deadRemark || remark.message.find("90 % of the outputs of 'trunk'") != std::string::npos;
        vanishingRemark = vanishingRemark || remark.message.find("Almost no gradient reaches 'trunk'") == 0;
    }
    Check(deadRemark && vanishingRemark && remarks.size() == 2, "the latest update is read in plain words");
    Check(editor::CheckProbes(log, "critic").empty(), "no probes, no remark");

    // Large outputs are normal for a linear output layer, a warning for a tanh layer;
    // zeros of a block without parameters come from the observation.
    {
        runtime::RunLogWriter writer(path, header);
        runtime::BlockProbe tanh;
        tanh.block = "layer 1";
        tanh.type = "dense tanh";
        tanh.samples = 100;
        tanh.deviation = 0.9;
        tanh.saturatedShare = 0.8;
        tanh.hasGradient = true;
        tanh.gradientNorm = 0.1;
        runtime::BlockProbe output = tanh;
        output.block = "output";
        output.type = "dense linear";
        runtime::BlockProbe concat = tanh;
        concat.block = "features";
        concat.type = "concat";
        concat.hasGradient = false;
        concat.deadShare = 0.7;
        writer.WriteProbes({10, "critic", {tanh, output, concat}});
    }
    auto const critic = editor::CheckProbes(runtime::LoadRunLog(path), "critic");
    Check(critic.size() == 1 && critic[0].block == "layer 1",
        "saturation is reported for tanh layers only, and pass-through blocks are not judged");
    std::filesystem::remove(path);
}

void Schedule()
{
    auto const plan = schema::ParseProbeEvery(R"({ "schema": "ppokit.train/1", "probes": { "every": 5 } })", "train.json");
    Check(plan == 5 && schema::ProbesDue(5, 10) && !schema::ProbesDue(5, 11) && !schema::ProbesDue(0, 10),
        "probes every 5 updates");
    try
    {
        schema::ParseProbeEvery(R"({ "schema": "ppokit.train/1", "probes": { "every": 0 } })", "train.json");
        Check(false, "every 0 is refused");
    }
    catch (schema::SchemaError const& error)
    {
        Check(std::string(error.what()).find("probes.every") != std::string::npos, "the message names probes.every");
    }
}

} // namespace

int main()
{
    try
    {
        ProbesNeverChangeTraining();
        SetsCountPresentEntities();
        JournalAndDiagnostics();
        Schedule();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0)
    {
        std::cerr << failures << " probe check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit probe tests passed\n";
    return EXIT_SUCCESS;
}
