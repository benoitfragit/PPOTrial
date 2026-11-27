/**
 * @file PPOKit/tests/artifact_tests.cpp
 * @brief Versioned artifacts: header round trip, refusals that say what changed, an exact
 *        resume (the next update gives the same weights bit for bit), policy checkpoints,
 *        and damaged files that leave the learner untouched.
 */
#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/graph/graph_learner.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "support/corridor_task.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

namespace {

using namespace ppokit;
using Learner = graph::GraphLearner<2, 1>;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

template<typename Body>
std::string ErrorOf(Body&& body)
{
    try
    {
        body();
    }
    catch (std::exception const& error)
    {
        return error.what();
    }
    return {};
}

bool Contains(std::string const& text, std::string const& part)
{
    return text.find(part) != std::string::npos;
}

/** @brief The corridor graph with a trunk of `width`, and its identity. */
struct Networks final
{
    schema::TaskSchema task;
    schema::BlockGraphSchema graphs;
    artifacts::ArtifactIdentity identity;
};

Networks CorridorNetworks(int width, std::string const& extra = {})
{
    std::string const text = R"({ "schema": "ppokit.graph/2",
      "actor": { "nodes": { "state": { "type": "input", "from": "observation.state" },
                            "trunk": { "type": "mlp", "layers": 2, "width": )" + std::to_string(width)
        + R"(, "activation": "tanh" },)" + extra + R"(
                            "policy": { "type": "gaussian_head" } },
                 "edges": [ ["state", "trunk"], )" + (extra.empty() ? std::string(R"(["trunk", "policy"])")
                                                               : std::string(R"(["trunk", "extra"], ["extra", "policy"])"))
        + R"( ] },
      "critic": { "nodes": { "state": { "type": "input", "from": "observation.state" },
                             "trunk": { "type": "mlp", "layers": 2, "width": 16, "activation": "tanh" },
                             "value": { "type": "value" } },
                  "edges": [ ["state", "trunk"], ["trunk", "value"] ] } })";
    Networks networks;
    networks.task = schema::LoadTaskSchema(PPOKIT_CORRIDOR_TASK);
    networks.graphs = schema::ParseBlockGraph(text, "graph.json", networks.task);
    networks.identity = schema::ArtifactIdentityOf(networks.task, networks.graphs);
    return networks;
}

Learner MakeLearner(Networks const& networks, std::uint32_t seed)
{
    core::PpoSettings settings;
    settings.actorLearningRate = 3e-3;
    settings.criticLearningRate = 3e-3;
    settings.seed = seed;
    return Learner(schema::BuildGraph(networks.graphs.actor), schema::BuildGraph(networks.graphs.critic),
        schema::MakePolicyLayout(networks.graphs, networks.task), settings);
}

/** @brief One collection on a fresh corridor, then one update. */
void TrainOnce(Learner& learner)
{
    tests::CorridorTask task;
    Learner::Buffer buffer(128);
    runtime::RolloutCollector<tests::CorridorTask, Learner, Learner::Buffer> collector(task, learner);
    collector.Collect(buffer);
    learner.Update(buffer);
}

bool SameParameters(graph::Graph& a, graph::Graph& b)
{
    auto const left = a.Parameters();
    auto const right = b.Parameters();
    bool same = left.size() == right.size();
    for (std::size_t tensor = 0; same && tensor < left.size(); ++tensor)
    {
        same = left[tensor]->value == right[tensor]->value;
    }
    return same;
}

std::filesystem::path Temporary(char const* name)
{
    return std::filesystem::temp_directory_path() / name;
}

void Header()
{
    auto const networks = CorridorNetworks(16);
    artifacts::ArtifactHeader header;
    header.identity = networks.identity;
    header.updates = 12;
    header.seed = 7;
    header.created = artifacts::UtcNow();
    auto const read = artifacts::ParseHeaderLine(artifacts::HeaderLine(header), "latest.ppokit");
    Check(read.identity.graphHash == header.identity.graphHash && read.identity.taskHash == header.identity.taskHash
            && read.updates == 12 && read.identity.actorBlocks.size() == 3 && read.created.size() == 20,
        "the header reads back, hashes exact");
    Check(Contains(ErrorOf([] { artifacts::ParseHeaderLine("hello", "x.txt"); }), "not a PPOKit artifact"),
        "a foreign file is named as such");

    auto remapped = header.identity;
    remapped.semanticHash = *remapped.semanticHash ^ 1;
    Check(Contains(ErrorOf([&] { artifacts::RequireCompatible(header.identity, remapped, "policy"); }),
              "policy semantics differ"), "same-shape models with different semantics are refused");
    auto historical = header.identity;
    historical.semanticHash.reset();
    Check(Contains(ErrorOf([&] { artifacts::RequireCompatible(historical, header.identity, "policy"); }),
              "migrate"), "ambiguous historical graph headers require the original contract");
    auto const wider = CorridorNetworks(32);
    std::string const widthMessage =
        ErrorOf([&] { artifacts::RequireCompatible(networks.identity, wider.identity, "latest.ppokit"); });
    Check(Contains(widthMessage, "the same blocks with other widths, settings or wiring"),
        "another width is explained: " + widthMessage);
    auto const extra = CorridorNetworks(16, R"( "extra": { "type": "dense", "width": 8, "activation": "tanh" },)");
    std::string const blockMessage =
        ErrorOf([&] { artifacts::RequireCompatible(networks.identity, extra.identity, "latest.ppokit"); });
    Check(Contains(blockMessage, "(added: extra:dense)"), "an added block is named: " + blockMessage);
    auto otherTask = networks.identity;
    otherTask.taskHash ^= 1;
    otherTask.observationSize = 3;
    std::string const taskMessage =
        ErrorOf([&] { artifacts::RequireCompatible(otherTask, networks.identity, "latest.ppokit"); });
    Check(Contains(taskMessage, "task.json changed its shapes or names (observation 3 -> 2 values"),
        "a task change gives the sizes: " + taskMessage);
}

/** @brief Stop after an update, resume in another process (another seed): the next update is identical. */
void ExactResume()
{
    auto const networks = CorridorNetworks(16);
    auto const path = Temporary("ppokit_artifact_training.ppokit");
    Learner original = MakeLearner(networks, 7);
    TrainOnce(original);
    TrainOnce(original);
    original.SaveTraining(path, networks.identity);
    TrainOnce(original);

    Learner resumed = MakeLearner(networks, 999);
    resumed.LoadTraining(path, networks.identity);
    TrainOnce(resumed);
    Check(SameParameters(original.Actor(), resumed.Actor()) && SameParameters(original.Critic(), resumed.Critic()),
        "the update after a resume gives the same weights, bit for bit");
    Check(artifacts::ReadArtifactHeader(path).updates == 2, "the header records the update count");
    auto const continued = Temporary("ppokit_artifact_continued.ppokit");
    resumed.SaveTraining(continued, networks.identity);
    auto const resumedHeader = artifacts::ReadArtifactHeader(continued);
    Check(resumedHeader.updates == 3 && resumedHeader.seed == 7,
        "a resumed artifact retains its original seed, not the constructor seed");
    std::filesystem::remove(continued);

    Learner other = MakeLearner(CorridorNetworks(32), 7);
    Check(Contains(ErrorOf([&] { other.LoadTraining(path, CorridorNetworks(32).identity); }), "other widths"),
        "a checkpoint of another network is refused with the reason");
    std::filesystem::remove(path);
}

/** @brief Corruption late in the critic state cannot install earlier actor tensors or Adam state. */
void InvalidOptimizerCheckpoint()
{
    auto const networks = CorridorNetworks(16);
    auto const path = Temporary("ppokit_bad_adam.ppokit");
    Learner source = MakeLearner(networks, 7);
    TrainOnce(source);
    source.SaveTraining(path, networks.identity);
    std::ifstream input(path);
    std::string const original((std::istreambuf_iterator<char>(input)), {});
    input.close();
    auto const critic = original.find("critic_adam ");
    auto const second = original.find("second ", critic);
    auto const value = original.find(' ', original.find('\n', second) + 1) + 1;
    auto const end = original.find_first_of(" \n", value);
    for (std::string const& bad : {original.substr(0, value) + "-1" + original.substr(end),
             original.substr(0, value) + "nan" + original.substr(end),
             original.substr(0, critic + 12) + "-1" + original.substr(original.find('\n', critic)),
             original + "\ntrailing-data\n"})
    {
        std::ofstream(path, std::ios::trunc) << bad;
        Learner candidate = MakeLearner(networks, 999);
        Learner control = MakeLearner(networks, 999);
        Check(!ErrorOf([&] { candidate.LoadTraining(path, networks.identity); }).empty(), "bad Adam state rejected");
        Check(candidate.Updates() == control.Updates(), "failed restore preserves the update count");
        TrainOnce(candidate);
        TrainOnce(control);
        Check(SameParameters(candidate.Actor(), control.Actor()) && SameParameters(candidate.Critic(), control.Critic()),
            "rejected restore preserves weights, optimizers and the sampling sequence");
    }
    std::filesystem::remove(path);
}

void Policies()
{
    auto const networks = CorridorNetworks(16);
    auto const path = Temporary("ppokit_artifact_policy.ppokit");
    Learner trained = MakeLearner(networks, 7);
    TrainOnce(trained);
    trained.SavePolicy(path, networks.identity);
    Learner deployed = MakeLearner(networks, 123);
    deployed.LoadPolicy(path, networks.identity);
    bool same = true;
    for (float position = -1.F; position <= 1.F; position += 0.25F)
    {
        Learner::Observation const observation{position, 0.5F * position};
        same = same && trained.Mean(observation) == deployed.Mean(observation);
    }
    Check(same, "a policy checkpoint gives the same deterministic actions");
    Check(Contains(ErrorOf([&] { deployed.LoadTraining(path, networks.identity); }), "load it with LoadPolicy"),
        "a policy cannot resume training, and the message says what to do");

    // A damaged file is refused and leaves the learner as it was.
    std::ifstream in(path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    Learner::Observation const probe{0.3F, -0.2F};
    auto const before = deployed.Mean(probe);
    std::ofstream(path, std::ios::binary | std::ios::trunc) << text << "\ntrailing-data\n";
    Check(Contains(ErrorOf([&] { deployed.LoadPolicy(path, networks.identity); }), "trailing data"),
        "a policy with an unexpected suffix is rejected");
    Check(deployed.Mean(probe) == before, "the rejected policy leaves its actor untouched");
    text.resize(text.size() / 2);
    std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
    Check(Contains(ErrorOf([&] { deployed.LoadPolicy(path, networks.identity); }), "damaged checkpoint"),
        "a truncated checkpoint is reported as damaged");
    Check(deployed.Mean(probe) == before, "and the learner is unchanged");
    std::filesystem::remove(path);
}

/**
 * @brief The committed checkpoint of the corridor still loads: the artifact format of
 *        earlier PPOKit versions stays readable.
 */
void CommittedCheckpoint()
{
    auto const task = schema::LoadTaskSchema(PPOKIT_CORRIDOR_TASK);
    auto const graphs = schema::LoadBlockGraph(PPOKIT_CORRIDOR_BLOCKS, task);
    Learner learner = MakeLearner({task, graphs, {}}, 1);
    learner.LoadTraining(PPOKIT_CORRIDOR_CHECKPOINT, schema::ArtifactIdentityOf(task, graphs));
    Check(artifacts::ReadArtifactHeader(PPOKIT_CORRIDOR_CHECKPOINT).updates == 3,
        "the committed corridor checkpoint loads with this PPOKit");
}

/** @brief Writes the committed checkpoint again (only when its format changes on purpose). */
void WriteCommittedCheckpoint(char const* path)
{
    auto const task = schema::LoadTaskSchema(PPOKIT_CORRIDOR_TASK);
    auto const graphs = schema::LoadBlockGraph(PPOKIT_CORRIDOR_BLOCKS, task);
    Learner learner = MakeLearner({task, graphs, {}}, 7);
    for (int update = 0; update < 3; ++update)
    {
        TrainOnce(learner);
    }
    learner.SaveTraining(path, schema::ArtifactIdentityOf(task, graphs));
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 3 && std::string(argv[1]) == "--write-fixture")
    {
        WriteCommittedCheckpoint(argv[2]);
        return EXIT_SUCCESS;
    }
    try
    {
        Header();
        CommittedCheckpoint();
        ExactResume();
        InvalidOptimizerCheckpoint();
        Policies();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0)
    {
        std::cerr << failures << " artifact check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit artifact tests passed\n";
    return EXIT_SUCCESS;
}
