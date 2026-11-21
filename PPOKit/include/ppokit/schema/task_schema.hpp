/**
 * @file PPOKit/include/ppokit/schema/task_schema.hpp
 * @brief Description of a learning task read from `task.json`: observation, critic input,
 *        actions and reward channels.
 * @details A task file states the *shape* of what the agent sees and does, so that
 * PPOKit can generate typed layouts, check checkpoints and build networks without
 * the example hard-coding offsets. Minimal example:
 *
 * @code{.json}
 * {
 *   "schema": "ppokit.task/1",
 *   "name": "flight",
 *   "namespace": "ppo::examples::flight::task",
 *   "agents": { "min": 1, "max": 1 },
 *   "observation": {
 *     "runway": { "fields": ["along", "cross", "height"] },
 *     "lidar":  { "size": 61 },
 *     "red":    { "set": { "capacity": 10, "features": 13, "mask": "last" } }
 *   },
 *   "actions": {
 *     "controls": { "head": "gaussian", "names": ["elevator", "aileron"], "bounds": [-1, 1], "map": "clip" }
 *   },
 *   "rewards": { "Path": {}, "Crash": { "label": "Crash penalty" } }
 * }
 * @endcode
 *
 * Groups and channels keep the order in which they are written. Unknown keys are
 * errors, so a typo never silently changes a layout.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace ppokit::schema {

/** @brief Raised for an unreadable or invalid task file; the message names the faulty key. */
class SchemaError final : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/** @brief Contiguous block of the observation (or critic input) vector. */
struct ObservationGroup final
{
    /** @brief Group name, an identifier such as "runway" or "red_tracks". */
    std::string name;
    /** @brief Optional names of the features, in order; empty for an anonymous vector. */
    std::vector<std::string> featureNames;
    /** @brief Values per element (a plain group has a single element). */
    std::size_t features = 0;
    /** @brief Number of elements; greater than one only for a set of entities. */
    std::size_t capacity = 1;
    /** @brief True for a set of entities (tracks, team mates), possibly partly empty. */
    bool isSet = false;
    /** @brief Index, within an element, of the presence flag of a masked set. */
    std::optional<std::size_t> maskFeature;
    /** @brief Offset of the group in the flat vector. */
    std::size_t offset = 0;

    /** @brief Number of values of the group: capacity x features. */
    std::size_t Size() const noexcept { return capacity * features; }
};

/** @brief Ordered groups forming one flat input vector. */
struct InputLayout final
{
    std::vector<ObservationGroup> groups;
    /** @brief Total number of values. */
    std::size_t size = 0;
};

/** @brief Probability distribution of an action group. */
enum class ActionHead : std::uint8_t
{
    /** @brief Diagonal Gaussian, one mean per component (ppokit/heads/gaussian.hpp). */
    Gaussian,
    /** @brief Yes/no decision, one logit per component (ppokit/heads/bernoulli.hpp). */
    Bernoulli,
};

/** @brief How a sampled action is turned into the command applied to the environment. */
enum class ActionMapping : std::uint8_t
{
    /** @brief The latent sample is clamped to the bounds; PPO scores the latent sample. */
    Clip,
    /** @brief The sample is applied as is. */
    None,
};

/**
 * @brief Contiguous block of the action vector sharing one head.
 * @details The actor outputs one value per component whatever the head: a Gaussian mean
 * or a Bernoulli logit. Bounds and mapping only apply to Gaussian components.
 */
struct ActionGroup final
{
    std::string name;
    ActionHead head = ActionHead::Gaussian;
    /** @brief Names of the components, in order. */
    std::vector<std::string> componentNames;
    double lowerBound = -1.0;
    double upperBound = 1.0;
    ActionMapping mapping = ActionMapping::Clip;
    /** @brief Offset of the group in the flat action vector. */
    std::size_t offset = 0;

    std::size_t Size() const noexcept { return componentNames.size(); }
};

/** @brief Named reward channel; the total reward is the weighted sum of the channels. */
struct RewardChannel final
{
    /** @brief Identifier, used for code and JSON keys. */
    std::string name;
    /** @brief Human-readable label for logs and the studio; defaults to the name. */
    std::string label;
    /** @brief Default weight of the channel. */
    double weight = 1.0;
};

/** @brief Complete, validated description of a task. */
struct TaskSchema final
{
    std::string name;
    /** @brief C++ namespace of the generated layout header. */
    std::string cppNamespace;
    std::size_t minimumAgents = 1;
    std::size_t maximumAgents = 1;
    InputLayout observation;
    /** @brief Critic input; absent when the critic reads the actor observation. */
    std::optional<InputLayout> critic;
    std::vector<ActionGroup> actions;
    std::size_t actionSize = 0;
    std::vector<RewardChannel> rewards;
    /** @brief Hash of everything that fixes tensor shapes and orders (see ShapeHash). */
    std::uint64_t shapeHash = 0;
};

/**
 * @brief Parses and validates a task description.
 * @param text JSON text.
 * @param sourceName File name quoted in error messages.
 * @throws SchemaError with the path of the faulty key, e.g. "task.json: observation.red.set.capacity: ...".
 */
TaskSchema ParseTaskSchema(std::string_view text, std::string_view sourceName);

/** @brief Reads and parses a task file. @throws SchemaError, including when the file is missing. */
TaskSchema LoadTaskSchema(std::filesystem::path const& path);

/**
 * @brief FNV-1a 64 hash of the shapes and names that a trained network depends on.
 * @details Covers observation and critic groups, action groups and reward channel
 * names, in order. Labels, default weights and the namespace are excluded: changing
 * them does not invalidate a checkpoint.
 */
std::uint64_t ShapeHash(TaskSchema const& schema);

} // namespace ppokit::schema
