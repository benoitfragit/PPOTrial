/**
 * @file PPOKit/include/ppokit/artifacts/artifact.hpp
 * @brief Versioned artifacts: the identity of the networks a checkpoint was trained with.
 * @details Every PPOKit checkpoint starts with one JSON line (`ppokit.artifact/1`): its
 * kind (training: everything to resume; policy: the actor for inference), the backend,
 * the identity of the task and of the networks, the update count, the seed and the date.
 * The identity holds hashes (task shapes, graph, components) and a readable summary
 * (sizes, blocks), so that a refused checkpoint says what changed:
 *
 *   "model/latest.ppokit: trained for another network: the actor had blocks own:input,
 *    trunk:mlp; graph.json now has own:input, trunk:mlp, extra:dense (added: extra)"
 *
 * The payload after the header belongs to the learner that wrote it.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iosfwd>
#include <stdexcept>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ppokit::artifacts {

enum class ArtifactKind : std::uint8_t
{
    /** @brief Actor, critic, optimizer and random state: resumes training exactly. */
    Training,
    /** @brief The actor alone: evaluation and inference. */
    Policy,
};

/** @brief What a checkpoint's numbers mean: the task and networks they were trained for. */
struct ArtifactIdentity final
{
    std::string task;
    /** @brief ShapeHash of task.json: observation, critic input, actions and reward names. */
    std::uint64_t taskHash = 0;
    /** @brief Hash of the actor and critic blocks, their parameters and wiring, and their components. */
    std::uint64_t graphHash = 0;
    /** @brief Engine of the networks, e.g. "graph" (PPOKit graph engine) or "rltools". */
    std::string backend;
    std::string scalar = "f32";
    std::size_t observationSize = 0;
    std::size_t actionSize = 0;
    /** @brief "name:type" of every block, in graph order, for readable refusals. */
    std::vector<std::string> actorBlocks;
    std::vector<std::string> criticBlocks;
    /** @brief "name@version" of each component block and its hash (DLL + manifest + configuration). */
    std::vector<std::pair<std::string, std::uint64_t>> components;
    /**
     * @brief The environment the networks were trained in (task.json "contract"), empty when
     *        the task did not say (and in checkpoints written before it existed).
     */
    std::string contract;
    /** @brief Versioned semantic identity, absent in historical format-1 headers. */
    std::optional<std::uint64_t> semanticHash;
};

struct ArtifactHeader final
{
    ArtifactKind kind = ArtifactKind::Training;
    ArtifactIdentity identity;
    std::uint64_t updates = 0;
    std::uint32_t seed = 0;
    /** @brief UTC date of writing, "2026-11-22T10:15:00Z". */
    std::string created;
};

/** @brief A checkpoint refused because it does not fit the current task or networks. */
class ArtifactMismatch final : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/** @brief "training" or "policy". */
char const* ArtifactKindName(ArtifactKind kind) noexcept;

/** @brief The header line, newline included. */
std::string HeaderLine(ArtifactHeader const& header);

/**
 * @brief Reads the header line.
 * @throws std::runtime_error naming `source` when the line is not a PPOKit artifact header.
 */
ArtifactHeader ParseHeaderLine(std::string_view line, std::string const& source);

/**
 * @brief Reads and validates a checkpoint header before decoding its backend-specific payload.
 * @param requireTraining True for resume; false for inference (either kind contains an actor).
 * @throws ArtifactMismatch or std::runtime_error without modifying any model state.
 */
ArtifactHeader ReadCompatibleHeader(std::istream& input, ArtifactIdentity const& expected,
    std::string const& source, bool requireTraining);

/** @brief Refuses trailing non-whitespace after the last required checkpoint section. */
void RequirePayloadEnd(std::istream& input, std::string const& source);

/** @brief Reads only the header of an artifact file. */
ArtifactHeader ReadArtifactHeader(std::filesystem::path const& path);

/**
 * @brief Refuses a checkpoint trained for another task, other networks or another
 *        environment.
 * @details Structural compatibility: task, shapes, networks, engine. Semantic compatibility:
 * the contract; a checkpoint of another environment (same shapes, e.g. the T-6 in the PC-7
 * project) is refused, a transfer is explicit (ppokit-run transfer). A checkpoint without a
 * contract (written before it existed, or for a task without one) is accepted: nothing tells
 * its environment.
 * @param saved Identity in the checkpoint.
 * @param current Identity of what the program built from its task.json and graph.json.
 * @throws ArtifactMismatch listing every difference in plain words.
 */
void RequireCompatible(ArtifactIdentity const& saved, ArtifactIdentity const& current, std::string const& source);

/**
 * @brief Writes `target`: the artifact `source` with its contract replaced by `contract`,
 *        its tensors byte for byte (an explicit transfer to another environment).
 * @throws std::runtime_error when `target` exists (nothing is overwritten) or `source`
 *         cannot be read.
 */
void RetargetArtifact(std::filesystem::path const& source, std::filesystem::path const& target,
    std::string const& contract);

/** @brief The current UTC date and time, as written in headers. */
std::string UtcNow();

} // namespace ppokit::artifacts
