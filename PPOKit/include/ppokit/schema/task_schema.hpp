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
 *   "rewards": { "Path": {}, "Crash": { "label": "Crash penalty" } },
 *   "scenario": {
 *     "wind_kt": { "type": "float", "range": [0, 25], "default": 0 },
 *     "start":   { "type": "enum", "values": ["rolling", "airborne"], "default": "airborne" },
 *     "gusts":   { "type": "bool", "default": false },
 *     "opponents": { "type": "int", "range": [0, 10] }
 *   },
 *   "metrics": {
 *     "stopped": { "kind": "flag", "label": "Stopped on the runway" },
 *     "stop_m":  { "kind": "number", "unit": "m" }
 *   }
 * }
 * @endcode
 *
 * Groups and channels keep the order in which they are written. Unknown keys are
 * errors, so a typo never silently changes a layout.
 *
 * An action component may be active only in some states, e.g. a brake used on the ground:
 *
 * @code
 * "controls": { "head": "gaussian", "names": ["elevator", "brake"],
 *               "active": { "brake": { "when": "rollout", "above": 0.5 } } }
 * @endcode
 *
 * "when" names a feature of the observation; "above" (feature > value) or "at_most"
 * (feature <= value) gives the test. An inactive component is not drawn (its action is 0)
 * and takes no part in the likelihood, the KL, the entropy or the gradients.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ppokit::schema {

/** @brief Raised for an unreadable or invalid task file; the message names the faulty key. */
class SchemaError final : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;

    /**
     * @brief Error located in a file: what() reads "<source>: <path>: <detail>".
     * @param source File name, e.g. "graph.json".
     * @param path Dotted JSON path, e.g. "actor.nodes.red_attend.query"; empty for the root.
     * @param detail What is wrong.
     */
    SchemaError(std::string const& source, std::string path, std::string detail)
        : std::runtime_error(source + (path.empty() ? "" : ": " + path) + ": " + detail), path_(std::move(path)),
          detail_(std::move(detail))
    {
    }

    /** @brief JSON path of the faulty value (empty when unknown or at the root); the editor highlights it. */
    std::string const& Path() const noexcept { return path_; }
    /** @brief The message without its location. */
    std::string const& Detail() const noexcept { return detail_; }

private:
    std::string path_;
    std::string detail_;
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
    /** @brief Unit, frame or normalization convention, when the producer declares it. */
    std::string convention;
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
    /** @brief Deterministic Bernoulli decision at a logit of exactly zero. */
    bool onZero = false;
    /** @brief Offset of the group in the flat action vector. */
    std::size_t offset = 0;

    std::size_t Size() const noexcept { return componentNames.size(); }
};

/** @brief Comparison of an ActionActivity. */
enum class ActivityTest : std::uint8_t
{
    /** @brief Active when the feature is greater than the threshold. */
    Above,
    /** @brief Active when the feature is lower than or equal to the threshold. */
    AtMost,
};

/**
 * @brief State in which an action component is active (task.json "active"). Not part of
 *        the shape hash: it changes no tensor, only which components a policy draws.
 */
struct ActionActivity final
{
    /** @brief Component name, as in "names". */
    std::string component;
    /** @brief Index of the component in the flat action vector. */
    std::size_t axis = 0;
    /** @brief Index of the tested feature in the flat observation vector. */
    std::size_t feature = 0;
    ActivityTest test = ActivityTest::Above;
    double threshold = 0.0;
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

/** @brief Kind of an episode metric. */
enum class MetricKind : std::uint8_t
{
    /** @brief 0 or 1 per episode (a landing, a crash); suites use its rate. */
    Flag,
    /** @brief A number per episode (a distance, a difficulty); NaN when it does not apply. */
    Number,
};

/**
 * @brief A value the environment reports at the end of each episode (task.json "metrics"),
 *        for the evaluation suites: their gates and the ranking of the BEST policy. Not part
 *        of the shape hash.
 */
struct EpisodeMetric final
{
    std::string name;
    std::string label;
    /** @brief Unit for the studio, e.g. "m"; empty when the value has none. */
    std::string unit;
    MetricKind kind = MetricKind::Number;
};

/** @brief Kind of value of a scenario parameter. */
enum class ParameterType : std::uint8_t
{
    Float,
    Int,
    Bool,
    /** @brief One of a fixed list of names, stored as its index. */
    Enum,
};

/**
 * @brief One parameter of the episode scenarios (wind, start placement, number of opponents...).
 * @details Curriculum stages sample these parameters (train.json). Values are carried as
 * doubles: an enum as its index, a bool as 0 or 1, an int as an integral value.
 */
struct ScenarioParameter final
{
    std::string name;
    ParameterType type = ParameterType::Float;
    /** @brief Inclusive range of a float or int parameter. */
    double minimum = 0;
    double maximum = 0;
    /** @brief Names of an enum parameter, in index order. */
    std::vector<std::string> values;
    /** @brief Value used by stages that do not sample the parameter. */
    std::optional<double> defaultValue;

    /** @brief Index of an enum name. @return Nothing for an unknown name. */
    std::optional<std::size_t> EnumIndex(std::string const& valueName) const;
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
    /**
     * @brief State of the world for the observers (Visualiseur blocks): positions, attitudes,
     *        zones... filled by the environment for each agent, never read by the networks.
     *        Absent when the task has none. Not part of ShapeHash: adding or changing it
     *        keeps every checkpoint valid.
     */
    std::optional<InputLayout> world;
    std::vector<ActionGroup> actions;
    std::size_t actionSize = 0;
    /** @brief Components active only in some states, in task.json order. */
    std::vector<ActionActivity> activities;
    std::vector<RewardChannel> rewards;
    /** @brief Scenario parameters, in written order; empty when the task has no variation. */
    std::vector<ScenarioParameter> scenario;
    /** @brief Episode metrics, in written order; empty when the environment reports none. */
    std::vector<EpisodeMetric> metrics;
    /**
     * @brief The environment the networks are trained in ("contract": e.g. an aircraft and the
     *        version of its observation contract), empty when the task does not say. Not part of
     *        ShapeHash: two projects of the same shapes in different environments (T-6, PC-7)
     *        keep their checkpoints apart through it (ArtifactIdentity::contract).
     */
    std::string contract;
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

/** @brief Version 1 of the policy contract: tensor order, masks, action semantics and observation conventions. */
std::uint64_t SemanticHash(TaskSchema const& schema);

} // namespace ppokit::schema
