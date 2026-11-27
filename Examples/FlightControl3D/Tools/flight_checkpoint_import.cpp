/**
 * @file Examples/FlightControl3D/Tools/flight_checkpoint_import.cpp
 * @brief Converts a FlightControl3D training checkpoint (rl_tools learner,
 *        "PPO_FLIGHT_POLICY_V3") into a PPOKit graph-learner training checkpoint, so a run
 *        started with FlightControlHeadless continues under ppokit-run at the same update.
 *
 * Usage: flight_checkpoint_import V3_FILE LEVEL_FILE FLIGHT_PROJECT NEW_DIRECTORY [centreline1200]
 *
 * Both learners have the same networks (two tanh layers of 128 units on the 35
 * observations), with weights stored output-major. The rl_tools actor has one output layer
 * of 6 rows (5 Gaussian means, then the gear logit) and a log-std of 6 values (5 used);
 * the graph has a Gaussian head and a Bernoulli head. Weights, biases, log-std and both
 * Adam moments are copied; rl_tools counts Adam age from 1, the graph counts steps from 0.
 * Both learners use mt19937. Preserve its recurrence window across standard libraries.
 * Import to a new directory and validate the complete checkpoint before publishing it.
 */
#include "flight_config.hpp"
#include "flight_curriculum.hpp"
#include "flight_rng_checkpoint.hpp"
#include "flight_learner.hpp"
#include "flight_policy.hpp"
#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/component/native_module.hpp"
#include "ppokit/graph/checkpoint_io.hpp"
#include "ppokit/graph/graph_learner.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/project.hpp"
#include "ppokit/schema/task_schema.hpp"
#include "ppokit/schema/train_schema.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace flight = ppo::examples::flight;
namespace graph = ppokit::graph;
namespace schema = ppokit::schema;

using Tensor = std::vector<float>;
using Learner = graph::GraphLearner<flight::ObservationSize, flight::ContinuousActions + 1>;

constexpr std::size_t Width = 128;
constexpr std::size_t Means = flight::ContinuousActions;
constexpr std::size_t ActorOutputs = Means + 1;
/** @brief A parameter of the rl_tools checkpoint: value and its two Adam moments. */
struct Stored final
{
    Tensor value;
    Tensor first;
    Tensor second;
};

/** @brief One dense layer of the rl_tools checkpoint. */
struct StoredLayer final
{
    Stored weights;
    Stored biases;
};

/** @brief Everything the rl_tools checkpoint holds, including its random state. */
struct OldCheckpoint final
{
    std::string signature;
    std::mt19937 rng;
    std::uint64_t updates = 0;
    std::size_t actorAge = 0;
    std::size_t criticAge = 0;
    std::vector<StoredLayer> actor;
    Stored logStd;
    std::vector<StoredLayer> critic;
};

/** @brief Reads `count` values; the checkpoint writes each matrix on one line. */
Tensor ReadValues(std::istream& in, std::size_t count)
{
    Tensor values(count);
    for (float& value : values)
    {
        if (!(in >> value) || !std::isfinite(value))
        {
            throw std::runtime_error("flight checkpoint: missing or non-finite value");
        }
    }
    return values;
}

Stored ReadStored(std::istream& in, std::size_t count)
{
    Stored stored;
    stored.value = ReadValues(in, count);
    stored.first = ReadValues(in, count);
    stored.second = ReadValues(in, count);
    return stored;
}

/** @brief Input, hidden and output layers of an MLP, in checkpoint order. */
std::vector<StoredLayer> ReadMlp(std::istream& in, std::size_t outputs)
{
    std::size_t const sizes[][2] = {{flight::ObservationSize, Width}, {Width, Width}, {Width, outputs}};
    std::vector<StoredLayer> layers;
    for (auto const& size : sizes)
    {
        StoredLayer layer;
        layer.weights = ReadStored(in, size[0] * size[1]);
        layer.biases = ReadStored(in, size[1]);
        layers.push_back(std::move(layer));
    }
    return layers;
}

/** @brief Reads a "PPO_FLIGHT_POLICY_V3" file of width 128 (FlightLearner::Save). */
OldCheckpoint ReadOldCheckpoint(fs::path const& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("cannot open " + path.string());
    }
    std::string magic;
    std::size_t observations = 0;
    std::size_t means = 0;
    std::size_t outputs = 0;
    std::size_t width = 0;
    OldCheckpoint old;
    in >> magic >> observations >> means >> outputs >> width >> old.updates >> std::quoted(old.signature) >> old.actorAge
        >> old.criticAge;
    if (!in || magic != "PPO_FLIGHT_POLICY_V3" || observations != flight::ObservationSize || means != Means
        || outputs != ActorOutputs || width != Width || old.actorAge == 0 || old.criticAge == 0)
    {
        throw std::runtime_error(path.string() + ": not a FlightControl3D V3 checkpoint of width 128");
    }
    old.rng = flight::detail::ReadCheckpointRng(in);
    old.actor = ReadMlp(in, ActorOutputs);
    old.logStd = ReadStored(in, ActorOutputs);
    old.critic = ReadMlp(in, 1);
    in >> std::ws;
    if (!in.eof())
    {
        throw std::runtime_error(path.string() + ": trailing data after the critic");
    }
    return old;
}

/** @brief Rows [first, first + count) of an output-major weight matrix with `inputs` columns. */
Tensor Rows(Tensor const& weights, std::size_t inputs, std::size_t first, std::size_t count)
{
    auto const begin = weights.begin() + static_cast<std::ptrdiff_t>(first * inputs);
    return Tensor(begin, begin + static_cast<std::ptrdiff_t>(count * inputs));
}

/** @brief Values [first, first + count) of a vector. */
Tensor Slice(Tensor const& values, std::size_t first, std::size_t count)
{
    auto const begin = values.begin() + static_cast<std::ptrdiff_t>(first);
    return Tensor(begin, begin + static_cast<std::ptrdiff_t>(count));
}

/** @brief One tensor of the graph: value and moments, cut from a stored parameter. */
struct Converted final
{
    Tensor value;
    Tensor first;
    Tensor second;
};

Converted Whole(Stored const& stored)
{
    return {stored.value, stored.first, stored.second};
}

Converted RowsOf(Stored const& stored, std::size_t inputs, std::size_t first, std::size_t count)
{
    return {Rows(stored.value, inputs, first, count), Rows(stored.first, inputs, first, count),
        Rows(stored.second, inputs, first, count)};
}

Converted SliceOf(Stored const& stored, std::size_t first, std::size_t count)
{
    return {Slice(stored.value, first, count), Slice(stored.first, first, count), Slice(stored.second, first, count)};
}

/** @brief Trunk layers (weights, biases) of a stored MLP: its first two layers. */
std::vector<Converted> Trunk(std::vector<StoredLayer> const& layers)
{
    return {Whole(layers[0].weights), Whole(layers[0].biases), Whole(layers[1].weights), Whole(layers[1].biases)};
}

/**
 * @brief The actor tensors in the order of the graph's parameters: trunk, then the heads in
 *        the order the graph lists them (told apart by their sizes).
 */
std::vector<Converted> ActorTensors(OldCheckpoint const& old, std::vector<graph::Parameter*> const& parameters)
{
    std::vector<Converted> gaussian = {RowsOf(old.actor[2].weights, Width, 0, Means),
        SliceOf(old.actor[2].biases, 0, Means), SliceOf(old.logStd, 0, Means)};
    std::vector<Converted> bernoulli = {
        RowsOf(old.actor[2].weights, Width, Means, 1), SliceOf(old.actor[2].biases, Means, 1)};
    std::vector<Converted> tensors = Trunk(old.actor);
    if (parameters.size() != tensors.size() + gaussian.size() + bernoulli.size())
    {
        throw std::runtime_error("the actor of network.json is not a 2x128 trunk with a Gaussian and a Bernoulli head");
    }
    bool const gaussianFirst = parameters[tensors.size()]->value.size() == Means * Width;
    std::vector<Converted>& firstHead = gaussianFirst ? gaussian : bernoulli;
    std::vector<Converted>& secondHead = gaussianFirst ? bernoulli : gaussian;
    tensors.insert(tensors.end(), firstHead.begin(), firstHead.end());
    tensors.insert(tensors.end(), secondHead.begin(), secondHead.end());
    return tensors;
}

std::vector<Converted> CriticTensors(OldCheckpoint const& old)
{
    std::vector<Converted> tensors = Trunk(old.critic);
    tensors.push_back(Whole(old.critic[2].weights));
    tensors.push_back(Whole(old.critic[2].biases));
    return tensors;
}

/** @brief Copies the values into the graph; every size must match. */
void Assign(std::vector<Converted> const& tensors, std::vector<graph::Parameter*> const& parameters, char const* name)
{
    if (tensors.size() != parameters.size())
    {
        throw std::runtime_error(std::string("the ") + name + " of network.json has another number of tensors");
    }
    for (std::size_t index = 0; index < tensors.size(); ++index)
    {
        if (tensors[index].value.size() != parameters[index]->value.size())
        {
            throw std::runtime_error(std::string("tensor ") + std::to_string(index) + " of the " + name + " has "
                + std::to_string(parameters[index]->value.size()) + " values, the checkpoint "
                + std::to_string(tensors[index].value.size()));
        }
        parameters[index]->value = tensors[index].value;
    }
}

graph::checkpoint::Tensors FirstMoments(std::vector<Converted> const& tensors)
{
    graph::checkpoint::Tensors moments;
    for (Converted const& tensor : tensors)
    {
        moments.push_back(tensor.first);
    }
    return moments;
}

graph::checkpoint::Tensors SecondMoments(std::vector<Converted> const& tensors)
{
    graph::checkpoint::Tensors moments;
    for (Converted const& tensor : tensors)
    {
        moments.push_back(tensor.second);
    }
    return moments;
}

/** @brief The target project: task, networks (with their converted values), settings. */
struct Target final
{
    schema::Project project;
    schema::TaskSchema task;
    schema::BlockGraphSchema graphs;
    graph::Graph actor;
    graph::Graph critic;
    ppokit::core::PpoSettings settings;
};

Target LoadTarget(fs::path const& root)
{
    Target target{schema::LoadProject(root), {}, {}, {}, {}, {}};
    target.task = schema::LoadTaskSchema(target.project.Task());
    target.graphs = schema::LoadBlockGraph(target.project.Network(), target.task);
    target.actor = schema::BuildGraph(target.graphs.actor);
    target.critic = schema::BuildGraph(target.graphs.critic);
    target.settings = schema::LoadTrainingPlan(target.project.Training(), target.task).ppo.Settings();
    return target;
}

/**
 * @brief Writes the graph-learner training checkpoint (GraphLearner::SaveTraining layout)
 *        and the policy checkpoint (actor only).
 */
void WriteCheckpoints(fs::path const& directory, Target& target, OldCheckpoint const& old, std::vector<Converted> const& actor,
    std::vector<Converted> const& critic)
{
    ppokit::artifacts::ArtifactHeader header;
    header.identity = schema::ArtifactIdentityOf(target.task, target.graphs);
    header.updates = old.updates;
    header.seed = target.settings.seed;
    header.created = ppokit::artifacts::UtcNow();
    std::vector<graph::Parameter*> const actorParameters = target.actor.Parameters();
    std::vector<graph::Parameter*> const criticParameters = target.critic.Parameters();
    header.kind = ppokit::artifacts::ArtifactKind::Training;
    ppokit::artifacts::WriteFileAtomically(
        directory / "latest.ppokit",
        [&](std::ostream& out) {
            out << ppokit::artifacts::HeaderLine(header);
            graph::checkpoint::WriteTensors(out, "actor", actorParameters);
            graph::checkpoint::WriteTensors(out, "critic", criticParameters);
            out << "actor_adam " << old.actorAge - 1 << '\n';
            graph::checkpoint::WriteMoments(out, "first", FirstMoments(actor));
            graph::checkpoint::WriteMoments(out, "second", SecondMoments(actor));
            out << "critic_adam " << old.criticAge - 1 << '\n';
            graph::checkpoint::WriteMoments(out, "first", FirstMoments(critic));
            graph::checkpoint::WriteMoments(out, "second", SecondMoments(critic));
            graph::checkpoint::WriteRandomState(out, old.rng);
        },
        ppokit::artifacts::AtomicWriteOptions{}, "converted FlightControl3D checkpoint");
    header.kind = ppokit::artifacts::ArtifactKind::Policy;
    ppokit::artifacts::WriteFileAtomically(
        directory / "policy.ppokit",
        [&](std::ostream& out) {
            out << ppokit::artifacts::HeaderLine(header);
            graph::checkpoint::WriteTensors(out, "actor", actorParameters);
        },
        ppokit::artifacts::AtomicWriteOptions{}, "converted FlightControl3D policy");
}

/** @brief A historical curriculum file contains a single in-range level. */
std::size_t ReadLevel(fs::path const& file, flight::Config const& config)
{
    std::ifstream in(file);
    std::size_t level{};
    if (!(in >> level) || !(in >> std::ws).eof() || level >= flight::CurriculumLevels(config.training.curriculum))
    {
        throw std::runtime_error(file.string() + ": invalid curriculum level");
    }
    return level;
}

void Import(fs::path const& source, fs::path const& levelFile, fs::path const& project, fs::path const& output,
    bool sourceUsesCurrentGuidance)
{
    Target target = LoadTarget(project);
    auto const config = flight::LoadConfig(target.project.SettingsFile().value());
    OldCheckpoint const old = ReadOldCheckpoint(source);
    if (old.signature != config.checkpointSignature || target.task.contract != flight::EnvironmentContract(config))
    {
        throw std::runtime_error("historical aircraft/scenario signature differs from this project");
    }
    // The V3 signature covered neither lateral guidance nor the curriculum. Its source
    // profile must be declared when importing post-change archives.
    if (sourceUsesCurrentGuidance == config.runtime.legacyCentreline)
        throw std::runtime_error("source guidance profile differs from the destination project");
    auto const level = ReadLevel(levelFile, config);
    auto const actor = ActorTensors(old, target.actor.Parameters());
    auto const critic = CriticTensors(old);
    Assign(actor, target.actor.Parameters(), "actor");
    Assign(critic, target.critic.Parameters(), "critic");
    fs::path const stage = ppokit::artifacts::SuffixedPath(output, ".importing");
    if (fs::exists(output) || !fs::create_directory(stage))
    {
        throw std::runtime_error("checkpoint import requires a new output directory");
    }
    try
    {
        WriteCheckpoints(stage, target, old, actor, critic);
        std::ofstream state(stage / "ppokit_curriculum_level.txt");
        state << level << '\n';
        state.close();
        if (!state)
        {
            throw std::runtime_error("could not write the curriculum level");
        }
        Learner restored(std::move(target.actor), std::move(target.critic),
            schema::MakePolicyLayout(target.graphs, target.task), target.settings);
        restored.LoadTraining(stage / "latest.ppokit", schema::ArtifactIdentityOf(target.task, target.graphs));
        fs::rename(stage, output);
    }
    catch (...)
    {
        std::error_code ignored;
        fs::remove_all(stage, ignored);
        throw;
    }
}

} // namespace

int main(int argc, char** argv)
{
    if ((argc != 5 && argc != 6) || (argc == 6 && std::string(argv[5]) != "centreline1200"))
    {
        std::cerr << "usage: flight_checkpoint_import V3_FILE LEVEL_FILE FLIGHT_PROJECT NEW_DIRECTORY [centreline1200]\n";
        return 2;
    }
    try
    {
        ppokit::component::SetKnownModules(schema::ReadProgramRegistry(PPOKIT_PROGRAM_REGISTRY));
        Import(argv[1], argv[2], argv[3], argv[4], argc == 6);
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
