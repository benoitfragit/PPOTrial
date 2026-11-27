/**
 * @file Examples/Airspace/Tools/airspace_checkpoint_import.cpp
 * @brief Converts a legacy LibTorch Airspace checkpoint (RLMission, RLSNA model.pt, exported
 *        by export_legacy_checkpoint.py) into PPOKit artifacts of the agent's graph.json,
 *        then checks them against the legacy forward pass.
 *
 * Usage:
 * @code
 * airspace_checkpoint_import --agent <RLMission|RLSNA> --tensors <export.json>
 *     [--updates <n>] [--to-training <training.ppokit>] [--to-policy <policy.ppokit>]
 *     [--reference <legacy test vectors.json>]
 * @endcode
 *
 * graph.json reproduces the legacy networks (critic trunk shared with the actor, keys of the
 * attention, log-std computed from the state), so the conversion copies tensors:
 * - actor and critic parameters, torch Linear (out, in) = PPOKit Dense output-major;
 * - the one legacy Adam (one step count, both moments) as the joint Adam of the shared
 *   learner, whose tensor order is the legacy registration order;
 * - no random state exists in the legacy file: the engine starts from the training seed.
 * `--updates` records the agent's model version in the header (the TrainingAgent counts its
 * versions in settings/Training/<agent>.json; the legacy file holds no update count).
 *
 * Checks, before anything is kept: the PPOKit learner loads the training artifact (identity,
 * tensor sizes, joint Adam) and the policy artifact; with `--reference`, the loaded networks
 * give the legacy means, log-std, fire logits and values, observations with no present track
 * included (RLMission's graph masks with LibTorch's -1e9 fill). Observations with a
 * fractional presence are reported apart: the legacy mean pool weighted tracks by their
 * presence, PPOKit's mask is a flag; the environment never writes a fractional presence.
 * An existing destination is never overwritten.
 *
 * Re-stamping (the networks of graph.json changed their masking, not their parameters):
 * @code
 * airspace_checkpoint_import --agent RLMission --restamp <artifact.ppokit> --to <new.ppokit>
 *     --former-graph <graph.json the artifact was written for>
 * @endcode
 * The artifact must carry the identity of the former graph; the new one carries the
 * identity of the current graph.json, with the same tensors byte for byte, and is reloaded
 * by the learner before it is kept.
 */
#include "interception.hpp"
#include "missionlayout.hpp"
#include "targetselector.hpp"

#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/graph/checkpoint_io.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using Json = nlohmann::json;
using Tensors = ppokit::graph::checkpoint::Tensors;

/** @brief Relative tolerance of a converted output against the legacy float32 forward pass. */
constexpr double OutputTolerance = 5e-4;

struct Options final
{
    std::string agent;
    std::filesystem::path tensors;
    std::filesystem::path restamp;
    std::filesystem::path to;
    std::filesystem::path formerGraph;
    std::uint64_t updates = 0;
    std::filesystem::path toTraining;
    std::filesystem::path toPolicy;
    std::filesystem::path reference;
};

Options ParseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index + 1 < argc; index += 2)
    {
        std::string const name = argv[index];
        std::string const value = argv[index + 1];
        if (name == "--agent")
        {
            options.agent = value;
        }
        else if (name == "--tensors")
        {
            options.tensors = value;
        }
        else if (name == "--updates")
        {
            options.updates = std::stoull(value);
        }
        else if (name == "--to-training")
        {
            options.toTraining = value;
        }
        else if (name == "--to-policy")
        {
            options.toPolicy = value;
        }
        else if (name == "--reference")
        {
            options.reference = value;
        }
        else if (name == "--restamp")
        {
            options.restamp = value;
        }
        else if (name == "--to")
        {
            options.to = value;
        }
        else if (name == "--former-graph")
        {
            options.formerGraph = value;
        }
        else
        {
            throw std::invalid_argument("unknown option " + name);
        }
    }
    bool const agentKnown = options.agent == "RLMission" || options.agent == "RLSNA";
    bool const output = !options.toTraining.empty() || !options.toPolicy.empty();
    bool const converts = !options.tensors.empty() && output;
    bool const restamps = !options.restamp.empty() && !options.to.empty() && !options.formerGraph.empty();
    if (argc % 2 == 0 || !agentKnown || converts == restamps)
    {
        throw std::invalid_argument("usage: airspace_checkpoint_import --agent <RLMission|RLSNA> --tensors "
                                    "<export.json> [--updates <n>] [--to-training <file>] [--to-policy <file>] "
                                    "[--reference <vectors.json>]\n   or: airspace_checkpoint_import --agent "
                                    "<RLMission|RLSNA> --restamp <artifact.ppokit> --to <new.ppokit> "
                                    "--former-graph <graph.json>");
    }
    return options;
}

Json ReadJson(std::filesystem::path const& path)
{
    std::ifstream stream(path);
    if (!stream)
    {
        throw std::runtime_error(path.string() + ": cannot open");
    }
    return Json::parse(stream);
}

Tensors TensorsOf(Json const& values)
{
    Tensors tensors;
    for (Json const& tensor : values)
    {
        tensors.push_back(tensor.get<std::vector<float>>());
    }
    return tensors;
}

/** @brief Sizes of `tensors` must be those of the network's parameters, in order. */
void RequireSizes(Tensors const& tensors, std::vector<ppokit::graph::Parameter*> const& parameters, char const* label)
{
    bool fits = tensors.size() == parameters.size();
    for (std::size_t index = 0; fits && index < tensors.size(); ++index)
    {
        fits = tensors[index].size() == parameters[index]->value.size();
    }
    if (!fits)
    {
        throw std::runtime_error(std::string("the exported ") + label + " tensors do not fit graph.json");
    }
}

void RequireAbsent(std::filesystem::path const& path)
{
    if (!path.empty() && std::filesystem::exists(path))
    {
        throw std::runtime_error(path.string() + " exists: remove it first (nothing is overwritten)");
    }
}

/** @brief The exported legacy state, checked against the networks of graph.json. */
struct LegacyState final
{
    Tensors actor;
    Tensors critic;
    std::size_t adamSteps = 0;
    Tensors first;
    Tensors second;
};

template<typename Learner>
LegacyState ReadLegacyState(Json const& document, Learner& learner)
{
    LegacyState state;
    state.actor = TensorsOf(document.at("actor"));
    state.critic = TensorsOf(document.at("critic"));
    state.adamSteps = document.at("adam_steps").get<std::size_t>();
    state.first = TensorsOf(document.at("first"));
    state.second = TensorsOf(document.at("second"));
    RequireSizes(state.actor, learner.Actor().Parameters(), "actor");
    RequireSizes(state.critic, learner.Critic().Parameters(), "critic");
    if (!learner.Joint() || state.first.size() != state.second.size())
    {
        throw std::runtime_error("graph.json must share the critic trunk (one Adam), as the legacy agent");
    }
    // The settings of the legacy Adam must be PPOKit's: lr of the training settings, standard betas.
    std::vector<double> const betas = document.at("betas").get<std::vector<double>>();
    bool const standard = betas.size() == 2 && betas[0] == 0.9 && betas[1] == 0.999
        && document.at("eps").get<double>() == 1e-8 && document.at("weight_decay").get<double>() == 0.0;
    if (!standard)
    {
        throw std::runtime_error("the legacy Adam settings differ from PPOKit's Adam");
    }
    return state;
}

/** @brief Writes the training artifact: header, actor, critic, joint Adam, fresh random state. */
void WriteTraining(std::filesystem::path const& path, ppokit::artifacts::ArtifactIdentity const& identity,
    LegacyState const& state, std::uint64_t updates, std::uint32_t seed)
{
    ppokit::artifacts::ArtifactHeader header;
    header.kind = ppokit::artifacts::ArtifactKind::Training;
    header.identity = identity;
    header.updates = updates;
    header.seed = seed;
    header.created = ppokit::artifacts::UtcNow();
    ppokit::artifacts::WriteFileAtomically(
        path,
        [&](std::ostream& out) {
            out << ppokit::artifacts::HeaderLine(header);
            ppokit::graph::checkpoint::WriteMoments(out, "actor", state.actor);
            ppokit::graph::checkpoint::WriteMoments(out, "critic", state.critic);
            out << "adam " << state.adamSteps << '\n';
            ppokit::graph::checkpoint::WriteMoments(out, "first", state.first);
            ppokit::graph::checkpoint::WriteMoments(out, "second", state.second);
            ppokit::graph::checkpoint::WriteRandomState(out, std::mt19937(seed));
        },
        ppokit::artifacts::AtomicWriteOptions{}, "Airspace training checkpoint");
}

/** @brief Writes the policy artifact: header and actor. */
void WritePolicy(std::filesystem::path const& path, ppokit::artifacts::ArtifactIdentity const& identity,
    LegacyState const& state, std::uint64_t updates, std::uint32_t seed)
{
    ppokit::artifacts::ArtifactHeader header;
    header.kind = ppokit::artifacts::ArtifactKind::Policy;
    header.identity = identity;
    header.updates = updates;
    header.seed = seed;
    header.created = ppokit::artifacts::UtcNow();
    ppokit::artifacts::WriteFileAtomically(
        path,
        [&](std::ostream& out) {
            out << ppokit::artifacts::HeaderLine(header);
            ppokit::graph::checkpoint::WriteMoments(out, "actor", state.actor);
        },
        ppokit::artifacts::AtomicWriteOptions{}, "Airspace policy checkpoint");
}

/** @brief True for a reference case outside the environment's domain: a fractional presence. */
bool SentinelCase(Json const& reference)
{
    return reference.at("category").get<std::string>() == "fractional_presence";
}

/** @brief Largest error of `actual` against `expected`, relative to max(1, |expected|). */
double RelativeError(std::vector<double> const& actual, std::vector<double> const& expected)
{
    if (actual.size() != expected.size())
    {
        return std::numeric_limits<double>::infinity();
    }
    double worst = 0.0;
    for (std::size_t index = 0; index < std::min(actual.size(), expected.size()); ++index)
    {
        worst = std::max(worst, std::abs(actual[index] - expected[index]) / std::max(1.0, std::abs(expected[index])));
    }
    return worst;
}

/** @brief Outputs of the loaded networks for one observation, as the legacy vectors name them. */
template<typename Learner>
Json OutputsOf(Learner& learner, std::vector<float> const& values)
{
    typename Learner::Observation observation{};
    if (values.size() != observation.size())
    {
        throw std::runtime_error("reference observation of the wrong size");
    }
    std::copy(values.begin(), values.end(), observation.begin());
    (void)learner.Mean(observation); // Runs the actor.
    Json outputs;
    ppokit::graph::Graph& actor = learner.Actor();
    std::size_t const components = actor.At(actor.OutputId(0)).OutputShape().width;
    std::vector<double> means;
    std::vector<double> logStd;
    for (std::size_t component = 0; component < components; ++component)
    {
        means.push_back(actor.Output(0).values[component]);
        logStd.push_back(learner.LogStd(component));
    }
    outputs["mu"] = means;
    outputs["log_std"] = logStd;
    if (actor.OutputCount() > 1)
    {
        outputs["fire_logit"] = std::vector<double>{actor.Output(1).values[0]};
    }
    outputs["value"] = std::vector<double>{learner.Value(observation)};
    return outputs;
}

/**
 * @brief Compares the loaded networks with the legacy forward on every reference case.
 * @param withCritic False for a policy artifact (no critic: its value is not compared).
 * @return False when a case outside the sentinel cases differs beyond OutputTolerance.
 */
template<typename Learner>
bool MatchesReference(Learner& learner, Json const& vectors, bool withCritic)
{
    double worst = 0.0;
    double worstSentinel = 0.0;
    std::size_t compared = 0;
    std::size_t sentinels = 0;
    for (Json const& reference : vectors.at("cases"))
    {
        Json const outputs = OutputsOf(learner, reference.at("observation").get<std::vector<float>>());
        double error = 0.0;
        for (auto const& [name, expected] : reference.at("legacy").items())
        {
            if (name == "value" && !withCritic)
            {
                continue;
            }
            error = std::max(error, RelativeError(outputs.at(name).get<std::vector<double>>(),
                                        expected.get<std::vector<double>>()));
        }
        if (SentinelCase(reference))
        {
            worstSentinel = std::max(worstSentinel, error);
            ++sentinels;
            continue;
        }
        if (error > OutputTolerance)
        {
            std::cerr << "case " << reference.at("id").get<std::string>() << ": relative error " << error << '\n';
        }
        worst = std::max(worst, error);
        ++compared;
    }
    std::cout << "  legacy forward: " << compared << " cases, largest relative error " << worst << "; " << sentinels
              << " cases with no present track or a fractional presence (legacy -1e9 sentinels), largest "
              << worstSentinel << '\n';
    return compared > 0 && worst <= OutputTolerance;
}

/** @brief The tensors of a loaded network, to compare with the exported ones. */
Tensors ValuesOf(std::vector<ppokit::graph::Parameter*> const& parameters)
{
    Tensors values;
    for (ppokit::graph::Parameter const* parameter : parameters)
    {
        values.push_back(parameter->value);
    }
    return values;
}

/** @brief Writes, reloads and checks the artifacts of one agent. */
template<std::size_t ObservationSize, std::size_t ActionSize>
void Convert(ppo::model::GraphNetworkFiles const& files, Options const& options)
{
    RequireAbsent(options.toTraining);
    RequireAbsent(options.toPolicy);
    ppokit::core::PpoSettings settings;
    settings.criticLearningRate = settings.actorLearningRate; // One Adam: the shared trunk.
    auto network = ppo::model::LoadGraphNetwork<ObservationSize, ActionSize>(files, settings);
    Json const document = ReadJson(options.tensors);
    LegacyState const state = ReadLegacyState(document, *network.learner);
    std::uint32_t const seed = settings.seed;
    std::vector<std::filesystem::path> written;
    try
    {
        if (!options.toTraining.empty())
        {
            WriteTraining(options.toTraining, network.identity, state, options.updates, seed);
            written.push_back(options.toTraining);
            auto loaded = ppo::model::LoadGraphNetwork<ObservationSize, ActionSize>(files, settings);
            loaded.learner->LoadTraining(options.toTraining, network.identity);
            bool const same = ValuesOf(loaded.learner->Actor().Parameters()) == state.actor
                && ValuesOf(loaded.learner->Critic().Parameters()) == state.critic;
            if (!same)
            {
                throw std::runtime_error(options.toTraining.string() + ": the reloaded tensors differ");
            }
            std::cout << options.toTraining.string() << ": training artifact, Adam step " << state.adamSteps
                      << ", reloaded by the learner\n";
            if (!options.reference.empty() && !MatchesReference(*loaded.learner, ReadJson(options.reference), true))
            {
                throw std::runtime_error(options.toTraining.string() + ": outputs differ from the legacy agent");
            }
        }
        if (!options.toPolicy.empty())
        {
            WritePolicy(options.toPolicy, network.identity, state, options.updates, seed);
            written.push_back(options.toPolicy);
            auto loaded = ppo::model::LoadGraphNetwork<ObservationSize, ActionSize>(files, settings);
            loaded.learner->LoadPolicy(options.toPolicy, network.identity);
            std::cout << options.toPolicy.string() << ": policy artifact, reloaded by the learner\n";
            if (!options.reference.empty() && !MatchesReference(*loaded.learner, ReadJson(options.reference), false))
            {
                throw std::runtime_error(options.toPolicy.string() + ": outputs differ from the legacy agent");
            }
        }
    }
    catch (...)
    {
        // A rejected conversion leaves nothing behind.
        for (std::filesystem::path const& path : written)
        {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
        }
        throw;
    }
}

/** @brief The text after the header line of an artifact: its tensors, unchanged. */
std::string ArtifactBody(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    std::string header;
    if (!stream || !std::getline(stream, header))
    {
        throw std::runtime_error(path.string() + ": cannot read");
    }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

/**
 * @brief Writes `options.to`: the tensors of `options.restamp`, written for the networks of
 *        `options.formerGraph`, under the identity of the current graph.json; reloaded first.
 */
template<std::size_t ObservationSize, std::size_t ActionSize>
void Restamp(ppo::model::GraphNetworkFiles const& files, Options const& options)
{
    RequireAbsent(options.to);
    ppokit::core::PpoSettings settings;
    settings.criticLearningRate = settings.actorLearningRate;
    ppo::model::GraphNetworkFiles const former{files.task, options.formerGraph};
    auto const formerNetwork = ppo::model::LoadGraphNetwork<ObservationSize, ActionSize>(former, settings);
    ppokit::artifacts::ArtifactHeader header = ppokit::artifacts::ReadArtifactHeader(options.restamp);
    ppokit::artifacts::RequireCompatible(header.identity, formerNetwork.identity, options.restamp.string());
    auto network = ppo::model::LoadGraphNetwork<ObservationSize, ActionSize>(files, settings);
    header.identity = network.identity;
    header.created = ppokit::artifacts::UtcNow();
    std::string const body = ArtifactBody(options.restamp);
    ppokit::artifacts::WriteFileAtomically(
        options.to, [&](std::ostream& out) { out << ppokit::artifacts::HeaderLine(header) << body; },
        ppokit::artifacts::AtomicWriteOptions::Binary(), "re-stamped Airspace checkpoint");
    try
    {
        if (header.kind == ppokit::artifacts::ArtifactKind::Training)
        {
            network.learner->LoadTraining(options.to, network.identity);
        }
        else
        {
            network.learner->LoadPolicy(options.to, network.identity);
        }
    }
    catch (...)
    {
        std::error_code ignored;
        std::filesystem::remove(options.to, ignored);
        throw;
    }
    std::cout << options.to.string() << ": " << ppokit::artifacts::ArtifactKindName(header.kind)
              << " artifact of update " << header.updates << ", re-stamped and reloaded\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Options const options = ParseOptions(argc, argv);
        constexpr std::size_t MissionObservation = rl::MissionObservationLayout::kStateDimension;
        constexpr std::size_t MissionActions = rl::MissionObservationLayout::kActionCount;
        using Interception = rl::StatesAndActions;
        constexpr std::size_t InterceptionObservation = std::tuple_size<Interception::STATES>::value;
        constexpr std::size_t InterceptionActions = std::tuple_size<Interception::ACTIONS>::value;
        bool const mission = options.agent == "RLMission";
        if (!options.restamp.empty() && mission)
        {
            Restamp<MissionObservation, MissionActions>(rl::MissionNetworkFiles(), options);
        }
        else if (!options.restamp.empty())
        {
            Restamp<InterceptionObservation, InterceptionActions>(rl::InterceptionNetworkFiles(), options);
        }
        else if (mission)
        {
            Convert<MissionObservation, MissionActions>(rl::MissionNetworkFiles(), options);
        }
        else
        {
            Convert<InterceptionObservation, InterceptionActions>(rl::InterceptionNetworkFiles(), options);
        }
        return EXIT_SUCCESS;
    }
    catch (std::exception const& error)
    {
        std::cerr << "airspace_checkpoint_import: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
