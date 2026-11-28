/**
 * @file Examples/BlueDefense/Agents/BluePatrol/Training/generalized_artifact.hpp
 * @brief Blue MAPPO checkpoints in the PPOKit artifact format (`.ppokit`).
 * @details The same text format as every PPOKit example (ppokit/artifacts/artifact.hpp,
 * ppokit/graph/checkpoint_io.hpp): one JSON header line (kind, identity of the task and
 * networks, updates, seed, date), then labelled sections of tensors written with 9
 * significant digits, so a resume is exact bit for bit. The networks stay the rl_tools
 * networks of the generalized learner (backend "rltools"); the tensors are their matrices,
 * row-major, in the order of ForEachActorParameter and ForEachCriticParameter.
 *
 * A training checkpoint holds:
 * @code
 * {"schema":"ppokit.artifact/1","kind":"training",...}
 * contract 10 <observation version> <raw> <compact> <actions> <actor hidden> <actor layers>
 *          <critic input> <critic hidden> <critic layers> <guidance semantic version>
 * team <P> <N> <blue capacity> <red capacity> <blue tracks> <objective tracks> <lidar>
 * actor <tensors>          parameters of the shared actor
 * critic <tensors>         parameters of the centralized critic
 * actor_adam <age>         then "settings" (alpha, betas, epsilon, weight decays, bias factor),
 * first <tensors>          the first moments and
 * second <tensors>         the second moments of the actor's Adam
 * critic_adam <age>        the same for the critic
 * rng                      then the sampling engine's state
 * @endcode
 * A policy checkpoint stops after the actor. The contract and team lines keep the strict
 * checks of the former V7/V10 files (layout, sizes, capacities, guidance semantics); the
 * header lets PPOKit refuse a checkpoint of other networks in plain words. Reading stages
 * every value before applying it: a damaged file leaves the networks as they were.
 */
#pragma once

#include "generalized_checkpoint.hpp"
#include "generalized_critic_network.hpp"
#include "patrol_training_checkpoint.hpp"

#include "blue_actor_graph_files.hpp"
#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/graph/checkpoint_io.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ppo::examples::bluedefense::defenders {

/** @brief Builds the identity of the Blue networks from the embedded task.json and graph_blocks.json. */
inline ppokit::artifacts::ArtifactIdentity MakeBluePatrolArtifactIdentity()
{
    ppokit::schema::TaskSchema const task = ppokit::schema::ParseTaskSchema(embedded::BlueTaskJson, "task.json");
    ppokit::schema::BlockGraphSchema const graphs =
        ppokit::schema::ParseBlockGraph(embedded::BlueGraphBlocksJson, "graph_blocks.json", task);
    return ppokit::schema::ArtifactIdentityOf(task, graphs, "rltools");
}

/**
 * @brief Identity of the Blue networks (computed once): the task and block graphs embedded
 *        in the binary, stored by rl_tools. A checkpoint of another task or other networks
 *        is refused with the differences in plain words.
 */
inline ppokit::artifacts::ArtifactIdentity const& BluePatrolArtifactIdentity()
{
    static ppokit::artifacts::ArtifactIdentity const identity = MakeBluePatrolArtifactIdentity();
    return identity;
}

/** @brief True when `path` starts with a PPOKit artifact header (a JSON line), false for a V7/V10 file. */
inline bool IsPpokitArtifact(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    char first = '\0';
    return stream.get(first) && first == '{';
}

/** @brief Visits the four parameters of one attention block (encoder, then query). */
template<typename Attention, typename Visitor>
void ForEachAttentionParameter(Attention& attention, Visitor& visitor)
{
    visitor(attention.encoderW);
    visitor(attention.encoderB);
    visitor(attention.queryW);
    visitor(attention.queryB);
}

/**
 * @brief Visits the actor's parameters in checkpoint order: the red then the friendly
 *        attention block, the dense trunk and the state-independent log-std.
 */
template<typename Actor, typename Visitor>
void ForEachActorParameter(Actor& actor, Visitor& visitor)
{
    ForEachAttentionParameter(actor.red, visitor);
    ForEachAttentionParameter(actor.friendlies, visitor);
    visitor(actor.input_layer.weights);
    visitor(actor.input_layer.biases);
    for (auto& layer : actor.hidden_layers)
    {
        visitor(layer.weights);
        visitor(layer.biases);
    }
    visitor(actor.output_layer.weights);
    visitor(actor.output_layer.biases);
    visitor(actor.log_std);
}

/** @brief Visits the critic's parameters in checkpoint order, layer by layer. */
template<typename Critic, typename Visitor>
void ForEachCriticParameter(Critic& critic, Visitor& visitor)
{
    visitor(critic.input_layer.weights);
    visitor(critic.input_layer.biases);
    for (auto& layer : critic.hidden_layers)
    {
        visitor(layer.weights);
        visitor(layer.biases);
    }
    visitor(critic.output_layer.weights);
    visitor(critic.output_layer.biases);
}

namespace generalized_artifact_detail {

/** @brief Which matrix of an rl_tools Adam parameter a section holds. */
enum class Field : std::uint8_t
{
    Value,
    FirstMoment,
    SecondMoment,
};

template<typename Parameter>
auto const& MatrixOf(Parameter const& parameter, Field field) noexcept
{
    if (field == Field::FirstMoment)
    {
        return parameter.gradient_first_order_moment;
    }
    if (field == Field::SecondMoment)
    {
        return parameter.gradient_second_order_moment;
    }
    return parameter.parameters;
}

template<typename Parameter>
auto& MutableMatrixOf(Parameter& parameter, Field field) noexcept
{
    if (field == Field::FirstMoment)
    {
        return parameter.gradient_first_order_moment;
    }
    if (field == Field::SecondMoment)
    {
        return parameter.gradient_second_order_moment;
    }
    return parameter.parameters;
}

/** @brief Copies one field of every visited parameter, row-major, into tensors. */
struct Collect final
{
    Field field;
    ppokit::graph::checkpoint::Tensors tensors;

    template<typename Parameter>
    void operator()(Parameter const& parameter)
    {
        auto const& matrix = MatrixOf(parameter, field);
        using Spec = typename std::decay_t<decltype(matrix)>::SPEC;
        std::vector<float> values;
        values.reserve(static_cast<std::size_t>(Spec::ROWS) * static_cast<std::size_t>(Spec::COLS));
        for (typename Spec::TI row = 0; row < Spec::ROWS; ++row)
        {
            for (typename Spec::TI column = 0; column < Spec::COLS; ++column)
            {
                values.push_back(static_cast<float>(rlt::get(matrix, row, column)));
            }
        }
        tensors.push_back(std::move(values));
    }
};

/** @brief Number of values of every visited parameter, for ReadTensors. */
struct Sizes final
{
    std::vector<std::size_t> sizes;

    template<typename Parameter>
    void operator()(Parameter const& parameter)
    {
        using Spec = typename std::decay_t<decltype(parameter.parameters)>::SPEC;
        sizes.push_back(static_cast<std::size_t>(Spec::ROWS) * static_cast<std::size_t>(Spec::COLS));
    }
};

/** @brief Writes staged tensors (sizes already checked) back into one field, and zeroes the gradients. */
struct Apply final
{
    Field field;
    ppokit::graph::checkpoint::Tensors const& tensors;
    std::size_t next = 0;

    template<typename Parameter>
    void operator()(Parameter& parameter)
    {
        auto& matrix = MutableMatrixOf(parameter, field);
        using Spec = typename std::decay_t<decltype(matrix)>::SPEC;
        std::vector<float> const& values = tensors[next];
        std::size_t index = 0;
        for (typename Spec::TI row = 0; row < Spec::ROWS; ++row)
        {
            for (typename Spec::TI column = 0; column < Spec::COLS; ++column)
            {
                rlt::set(matrix, row, column, values[index]);
                rlt::set(parameter.gradient, row, column, 0.F);
                ++index;
            }
        }
        ++next;
    }
};

template<typename Network>
ppokit::graph::checkpoint::Tensors CollectActor(Network const& actor, Field field)
{
    Collect collect{field, {}};
    ForEachActorParameter(actor, collect);
    return std::move(collect.tensors);
}

template<typename Network>
ppokit::graph::checkpoint::Tensors CollectCritic(Network const& critic, Field field)
{
    Collect collect{field, {}};
    ForEachCriticParameter(critic, collect);
    return std::move(collect.tensors);
}

/** @brief The contract line: everything the former V7/V10 headers checked. */
template<typename Critic>
void WriteContract(std::ostream& out)
{
    out << "contract " << kGeneralizedPolicyCheckpointVersion << ' ' << GeneralizedObservationLayout::Version << ' '
        << GeneralizedObservationLayout::Size << ' ' << ObservationLayout::Size << ' '
        << GeneralizedObservationLayout::ActionSize << ' ' << TrainingParameters::ACTOR_HIDDEN_DIM << ' '
        << TrainingParameters::ACTOR_NUM_LAYERS << ' ' << Critic::INPUT_DIM << ' '
        << GeneralizedCriticConfig::HiddenDim << ' ' << GeneralizedCriticConfig::Layers << ' '
        << kGeneralizedUnifiedGuidanceSemanticVersion << '\n';
}

/**
 * @brief Checks the contract line against the compiled networks.
 * @throws std::runtime_error naming `source` when one value differs.
 */
template<typename Critic>
void ReadContract(std::istream& in, std::string const& source)
{
    ppokit::graph::checkpoint::RequireLabel(in, "contract", source);
    unsigned version{};
    unsigned observationVersion{};
    std::size_t raw{};
    std::size_t compact{};
    std::size_t actions{};
    std::size_t actorHidden{};
    std::size_t actorLayers{};
    std::size_t criticInput{};
    std::size_t criticHidden{};
    std::size_t criticLayers{};
    std::string guidance;
    bool const read = static_cast<bool>(in >> version >> observationVersion >> raw >> compact >> actions
        >> actorHidden >> actorLayers >> criticInput >> criticHidden >> criticLayers >> guidance);
    bool const matches = read && version == kGeneralizedPolicyCheckpointVersion
        && observationVersion == GeneralizedObservationLayout::Version && raw == GeneralizedObservationLayout::Size
        && compact == ObservationLayout::Size && actions == GeneralizedObservationLayout::ActionSize
        && actorHidden == TrainingParameters::ACTOR_HIDDEN_DIM && actorLayers == TrainingParameters::ACTOR_NUM_LAYERS
        && criticInput == Critic::INPUT_DIM && criticHidden == GeneralizedCriticConfig::HiddenDim
        && criticLayers == GeneralizedCriticConfig::Layers && guidance == kGeneralizedUnifiedGuidanceSemanticVersion;
    if (!matches)
    {
        throw std::runtime_error(source + ": incompatible generalized blue MAPPO contract (observation layout, "
                                          "network sizes or guidance semantics)");
    }
}

/** @brief The team line: the episode's P/N and the compiled capacities. */
inline void WriteTeam(std::ostream& out, GeneralizedCheckpointMetadata metadata)
{
    out << "team ";
    generalized_checkpoint_detail::WriteMetadata(out, metadata);
    out << '\n';
}

inline GeneralizedCheckpointMetadata ReadTeam(std::istream& in, std::string const& source)
{
    ppokit::graph::checkpoint::RequireLabel(in, "team", source);
    return generalized_checkpoint_detail::ReadMetadata(in);
}

/** @brief One rl_tools Adam: its age, then its settings, then both moments. */
template<typename Optimizer>
void WriteAdam(std::ostream& out, char const* label, Optimizer const& optimizer,
    ppokit::graph::checkpoint::Tensors const& first, ppokit::graph::checkpoint::Tensors const& second)
{
    auto const& settings = optimizer.parameters;
    out << label << ' ' << optimizer.age << "\nsettings " << settings.alpha << ' ' << settings.beta_1 << ' '
        << settings.beta_2 << ' ' << settings.epsilon << ' ' << settings.weight_decay << ' '
        << settings.weight_decay_input << ' ' << settings.weight_decay_output << ' ' << settings.bias_lr_factor
        << '\n';
    ppokit::graph::checkpoint::WriteMoments(out, "first", first);
    ppokit::graph::checkpoint::WriteMoments(out, "second", second);
}

/** @brief A staged Adam: optimizer settings and age, then both moments. */
template<typename Optimizer>
struct StagedAdam final
{
    Optimizer optimizer;
    ppokit::graph::checkpoint::Tensors first;
    ppokit::graph::checkpoint::Tensors second;
};

template<typename Optimizer>
StagedAdam<Optimizer> ReadAdam(std::istream& in, char const* label, Optimizer const& current,
    std::vector<std::size_t> const& sizes, std::string const& source)
{
    ppokit::graph::checkpoint::RequireLabel(in, label, source);
    StagedAdam<Optimizer> staged{current, {}, {}};
    auto& settings = staged.optimizer.parameters;
    typename Optimizer::TI age{};
    in >> age;
    ppokit::graph::checkpoint::RequireLabel(in, "settings", source);
    bool const read = static_cast<bool>(in >> settings.alpha >> settings.beta_1 >> settings.beta_2
        >> settings.epsilon >> settings.weight_decay >> settings.weight_decay_input >> settings.weight_decay_output
        >> settings.bias_lr_factor);
    if (!read)
    {
        throw std::runtime_error(source + ": damaged checkpoint: unreadable '" + label + "' settings");
    }
    training_checkpoint_detail::RequireValidPatrolOptimizer(settings, age);
    staged.optimizer.age = age;
    staged.first = ppokit::graph::checkpoint::ReadTensors(in, "first", sizes, source);
    staged.second = ppokit::graph::checkpoint::ReadTensors(in, "second", sizes, source);
    return staged;
}

inline ppokit::artifacts::ArtifactHeader MakeHeader(ppokit::artifacts::ArtifactKind kind, std::uint64_t updates,
    std::uint32_t seed)
{
    ppokit::artifacts::ArtifactHeader header;
    header.kind = kind;
    header.identity = BluePatrolArtifactIdentity();
    header.updates = updates;
    header.seed = seed;
    header.created = ppokit::artifacts::UtcNow();
    return header;
}

} // namespace generalized_artifact_detail

/**
 * @brief Atomically writes an inference artifact: header, contract, team and actor.
 * @param metadata P/N the actor was trained or evaluated with.
 * @param updates Completed PPO updates of the actor, recorded in the header.
 * @param seed Training seed, recorded in the header.
 */
template<typename Critic, typename Actor>
void SaveGeneralizedPolicyArtifact(std::filesystem::path const& path, Actor const& actor,
    GeneralizedCheckpointMetadata metadata, std::uint64_t updates, std::uint32_t seed)
{
    namespace detail = generalized_artifact_detail;
    ppokit::artifacts::ArtifactHeader const header =
        detail::MakeHeader(ppokit::artifacts::ArtifactKind::Policy, updates, seed);
    ppokit::graph::checkpoint::Tensors const values = detail::CollectActor(actor, detail::Field::Value);
    SavePatrolAtomically(path, [&](std::ostream& out) {
        out << ppokit::artifacts::HeaderLine(header);
        detail::WriteContract<Critic>(out);
        detail::WriteTeam(out, metadata);
        ppokit::graph::checkpoint::WriteMoments(out, "actor", values);
        if (!out)
        {
            throw std::runtime_error("Failed to serialize the generalized blue actor artifact");
        }
    });
}

/**
 * @brief Reads the actor of a policy or training artifact (its Adam moments are left unchanged).
 * @return The P/N the actor was saved with.
 * @throws ppokit::artifacts::ArtifactMismatch for other task or networks, std::runtime_error
 *         for another contract or a damaged file.
 */
template<typename Critic, typename Actor>
GeneralizedCheckpointMetadata LoadGeneralizedPolicyArtifact(std::filesystem::path const& path, Actor& actor)
{
    namespace detail = generalized_artifact_detail;
    std::string const source = path.filename().string();
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error(source + ": cannot open the blue actor checkpoint");
    }
    ppokit::artifacts::ArtifactHeader const header =
        ppokit::artifacts::ReadCompatibleHeader(in, BluePatrolArtifactIdentity(), source, false);
    detail::ReadContract<Critic>(in, source);
    GeneralizedCheckpointMetadata const metadata = detail::ReadTeam(in, source);
    detail::Sizes sizes;
    ForEachActorParameter(actor, sizes);
    ppokit::graph::checkpoint::Tensors const values =
        ppokit::graph::checkpoint::ReadTensors(in, "actor", sizes.sizes, source);
    if (header.kind == ppokit::artifacts::ArtifactKind::Policy)
    {
        ppokit::artifacts::RequirePayloadEnd(in, source);
    }
    detail::Apply apply{detail::Field::Value, values};
    ForEachActorParameter(actor, apply);
    return metadata;
}

/**
 * @brief Atomically writes a training artifact at an update boundary.
 * @param updates Completed PPO updates.
 * @param metadata P/N of the saved episode.
 * @param seed Training seed, recorded in the header.
 */
template<typename Actor, typename Critic, typename ActorOptimizer, typename CriticOptimizer, typename RandomEngine>
void SaveGeneralizedTrainingArtifact(std::filesystem::path const& path, Actor const& actor, Critic const& critic,
    ActorOptimizer const& actorOptimizer, CriticOptimizer const& criticOptimizer, RandomEngine const& randomEngine,
    std::uint64_t updates, GeneralizedCheckpointMetadata metadata, std::uint32_t seed)
{
    namespace detail = generalized_artifact_detail;
    using Field = detail::Field;
    ppokit::artifacts::ArtifactHeader const header =
        detail::MakeHeader(ppokit::artifacts::ArtifactKind::Training, updates, seed);
    SavePatrolAtomically(path, [&](std::ostream& out) {
        out << ppokit::artifacts::HeaderLine(header);
        detail::WriteContract<Critic>(out);
        detail::WriteTeam(out, metadata);
        ppokit::graph::checkpoint::WriteMoments(out, "actor", detail::CollectActor(actor, Field::Value));
        ppokit::graph::checkpoint::WriteMoments(out, "critic", detail::CollectCritic(critic, Field::Value));
        detail::WriteAdam(out, "actor_adam", actorOptimizer, detail::CollectActor(actor, Field::FirstMoment),
            detail::CollectActor(actor, Field::SecondMoment));
        detail::WriteAdam(out, "critic_adam", criticOptimizer, detail::CollectCritic(critic, Field::FirstMoment),
            detail::CollectCritic(critic, Field::SecondMoment));
        ppokit::graph::checkpoint::WriteRandomState(out, randomEngine);
        if (!out)
        {
            throw std::runtime_error("Failed to serialize the generalized blue training artifact");
        }
    });
}

/**
 * @brief Restores a training artifact: parameters, both Adams (settings, ages, moments) and the RNG.
 * @param metadata Receives the P/N the state was saved with.
 * @return The completed PPO updates of the header.
 * @throws ppokit::artifacts::ArtifactMismatch, or std::runtime_error for a policy artifact,
 *         another contract or a damaged file; nothing is changed then.
 */
template<typename Actor, typename Critic, typename ActorOptimizer, typename CriticOptimizer, typename RandomEngine>
std::uint64_t LoadGeneralizedTrainingArtifact(std::filesystem::path const& path, Actor& actor, Critic& critic,
    ActorOptimizer& actorOptimizer, CriticOptimizer& criticOptimizer, RandomEngine& randomEngine,
    GeneralizedCheckpointMetadata& metadata, std::uint32_t* restoredSeed = nullptr)
{
    namespace detail = generalized_artifact_detail;
    using Field = detail::Field;
    std::string const source = path.filename().string();
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error(source + ": cannot open the blue training checkpoint");
    }
    ppokit::artifacts::ArtifactHeader const header =
        ppokit::artifacts::ReadCompatibleHeader(in, BluePatrolArtifactIdentity(), source, true);
    detail::ReadContract<Critic>(in, source);
    GeneralizedCheckpointMetadata const team = detail::ReadTeam(in, source);
    detail::Sizes actorSizes;
    ForEachActorParameter(actor, actorSizes);
    detail::Sizes criticSizes;
    ForEachCriticParameter(critic, criticSizes);
    // Everything is staged before anything is applied.
    ppokit::graph::checkpoint::Tensors const actorValues =
        ppokit::graph::checkpoint::ReadTensors(in, "actor", actorSizes.sizes, source);
    ppokit::graph::checkpoint::Tensors const criticValues =
        ppokit::graph::checkpoint::ReadTensors(in, "critic", criticSizes.sizes, source);
    auto const actorAdam = detail::ReadAdam(in, "actor_adam", actorOptimizer, actorSizes.sizes, source);
    auto const criticAdam = detail::ReadAdam(in, "critic_adam", criticOptimizer, criticSizes.sizes, source);
    RandomEngine const stagedEngine = ppokit::graph::checkpoint::ReadRandomState(in, source);
    ppokit::artifacts::RequirePayloadEnd(in, source);

    detail::Apply applyActor{Field::Value, actorValues};
    ForEachActorParameter(actor, applyActor);
    detail::Apply actorFirst{Field::FirstMoment, actorAdam.first};
    ForEachActorParameter(actor, actorFirst);
    detail::Apply actorSecond{Field::SecondMoment, actorAdam.second};
    ForEachActorParameter(actor, actorSecond);
    detail::Apply applyCritic{Field::Value, criticValues};
    ForEachCriticParameter(critic, applyCritic);
    detail::Apply criticFirst{Field::FirstMoment, criticAdam.first};
    ForEachCriticParameter(critic, criticFirst);
    detail::Apply criticSecond{Field::SecondMoment, criticAdam.second};
    ForEachCriticParameter(critic, criticSecond);
    actorOptimizer = actorAdam.optimizer;
    criticOptimizer = criticAdam.optimizer;
    randomEngine = stagedEngine;
    metadata = team;
    if (restoredSeed != nullptr)
    {
        *restoredSeed = header.seed;
    }
    return header.updates;
}

} // namespace ppo::examples::bluedefense::defenders
