/**
 * @file PPOKit/tests/task_schema_tests.cpp
 * @brief task.json parsing, validation messages, shape hash and generated layout header.
 */
#include "ppokit/schema/task_codegen.hpp"
#include "ppokit/schema/task_schema.hpp"
#include "sample_task_layout.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

namespace {

using ppokit::schema::ParseTaskSchema;
using ppokit::schema::SchemaError;

int failures = 0;

void Check(bool condition, char const* description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/** @brief Smallest valid task, extended by the tests through `extra` top-level members. */
std::string MinimalTask(std::string const& observation = R"({"own": {"size": 3}})",
    std::string const& actions = R"({"move": {"head": "gaussian", "size": 2}})",
    std::string const& rewards = R"({"Progress": {}})")
{
    return R"({"schema": "ppokit.task/1", "name": "minimal", "observation": )" + observation
        + R"(, "actions": )" + actions + R"(, "rewards": )" + rewards + "}";
}

/** @brief Checks that parsing fails with a message containing `expected`. */
void ExpectError(std::string const& text, std::string const& expected, char const* description)
{
    try
    {
        ParseTaskSchema(text, "task.json");
    }
    catch (SchemaError const& error)
    {
        bool const matches = std::string(error.what()).find(expected) != std::string::npos;
        if (!matches)
        {
            std::cerr << "  message was: " << error.what() << '\n';
        }
        Check(matches, description);
        return;
    }
    Check(false, description);
}

void ParsesLayoutsInWrittenOrder()
{
    auto const schema = ParseTaskSchema(MinimalTask(
        R"({"own": {"fields": ["b", "a"]}, "lidar": {"size": 4}, "tracks": {"set": {"capacity": 3, "features": 5, "mask": "last"}}})"),
        "task.json");
    Check(schema.observation.groups.size() == 3, "three observation groups");
    Check(schema.observation.groups[0].name == "own" && schema.observation.groups[1].name == "lidar",
        "groups keep the written order, not alphabetical order");
    Check(schema.observation.groups[1].offset == 2 && schema.observation.groups[2].offset == 6, "group offsets");
    Check(schema.observation.size == 2 + 4 + 15, "observation size includes every set entity");
    Check(schema.observation.groups[2].maskFeature == 4, "mask \"last\" is the final feature");
    Check(schema.cppNamespace == "ppokit::tasks::minimal", "default namespace");
    Check(schema.minimumAgents == 1 && schema.maximumAgents == 1, "single agent by default");
    Check(schema.actions[0].componentNames[1] == "move_1", "components named from the group when only a size is given");
    Check(!schema.critic, "critic reads the actor observation by default");
}

void ReportsFaultyKeys()
{
    auto const maximum = std::to_string(std::numeric_limits<std::size_t>::max());
    ExpectError(MinimalTask("{\"tracks\":{\"set\":{\"features\":2,\"capacity\":" + maximum + ",\"mask\":\"last\"}}}"),
        "exceeds the addressable size", "set width times capacity must fit");
    ExpectError(MinimalTask("{\"a\":{\"size\":" + maximum + "},\"b\":{\"size\":1}}"),
        "exceeds the addressable size", "groups must fit together");
    ExpectError(R"({"schema": "ppokit.task/2"})", "task.json: schema: expected \"ppokit.task/1\"", "wrong version");
    ExpectError(MinimalTask(R"({"own": {"size": 3, "colour": 1}})"), "observation.own.colour: unknown key",
        "unknown group key is named with its path");
    ExpectError(MinimalTask(R"({"red": {"set": {"features": 3}}})"), "observation.red.set: missing \"capacity\"",
        "set without capacity");
    ExpectError(MinimalTask(R"({"own": {"fields": ["x", "x"]}})"), "observation.own.fields[1]: duplicate name 'x'",
        "duplicate field");
    ExpectError(MinimalTask(R"({"a": {"fields": ["x"]}, "b": {"fields": ["x"]}})"),
        "observation.b.fields: feature 'x' already exists", "field names unique across groups");
    ExpectError(MinimalTask(R"({"own": {"size": 0}})"), "observation.own.size: expected a positive integer",
        "zero size");
    ExpectError(MinimalTask(R"({"own": {"size": 3}})", R"({"move": {"head": "beta", "size": 1}})"),
        "actions.move.head: expected \"gaussian\"", "unsupported head");
    ExpectError(MinimalTask(R"({"own": {"size": 3}})", R"({"move": {"head": "gaussian", "size": 1, "bounds": [1, 0]}})"),
        "actions.move.bounds: lower bound must be smaller", "inverted bounds");
    ExpectError(MinimalTask(R"({"own": {"size": 3}})", R"({"move": {"head": "gaussian", "size": 1}})",
                    R"({"Patrol tracking": {}})"),
        "rewards.Patrol tracking: channel names must be identifiers; put free text in \"label\"", "reward name");
    ExpectError(R"({"schema": "ppokit.task/1", "name": "x", "observation": {"a": {"size": 1}}, "rewards": {"R": {}}})",
        "task.json: actions: missing required key", "missing actions");
    ExpectError("{ not json", "task.json: invalid JSON", "syntax error");
}

/** @brief The minimal task with a "metrics" section. */
std::string TaskWithMetrics(std::string const& metrics)
{
    std::string text = MinimalTask();
    text.insert(text.size() - 1, R"(, "metrics": )" + metrics);
    return text;
}

void ShapeHashTracksShapesOnly()
{
    auto const reference = ParseTaskSchema(MinimalTask(), "task.json").shapeHash;
    auto const relabelled = ParseTaskSchema(MinimalTask(R"({"own": {"size": 3}})",
        R"({"move": {"head": "gaussian", "size": 2}})", R"({"Progress": {"label": "Other", "weight": 3}})"),
        "task.json").shapeHash;
    auto const resized = ParseTaskSchema(MinimalTask(R"({"own": {"size": 4}})"), "task.json").shapeHash;
    auto const renamed = ParseTaskSchema(MinimalTask(R"({"own": {"size": 3}})",
        R"({"move": {"head": "gaussian", "size": 2}})", R"({"Distance": {}})"), "task.json").shapeHash;
    Check(reference == relabelled, "labels and weights do not change the shape hash");
    Check(reference != resized, "a resized group changes the shape hash");
    Check(reference != renamed, "a renamed reward channel changes the shape hash");
    Check(reference == ParseTaskSchema(TaskWithMetrics(R"({"landed": {"kind": "flag"}})"), "task.json").shapeHash,
        "episode metrics do not change the shape hash");
}

/** @brief task.json "metrics": kinds, labels, units, and the faulty keys named. */
void ReadsEpisodeMetrics()
{
    auto const task = ParseTaskSchema(
        TaskWithMetrics(R"({"landed": {"kind": "flag"}, "stop_m": {"kind": "number", "label": "Stop", "unit": "m"}})"),
        "task.json");
    Check(task.metrics.size() == 2 && task.metrics[0].name == "landed" && task.metrics[0].label == "landed"
            && task.metrics[0].kind == ppokit::schema::MetricKind::Flag && task.metrics[1].unit == "m"
            && task.metrics[1].label == "Stop",
        "episode metrics keep their order, kind, label and unit");
    ExpectError(TaskWithMetrics(R"({"landed": {"kind": "bool"}})"), "metrics.landed.kind: expected \"flag\"",
        "an unknown metric kind");
    ExpectError(TaskWithMetrics(R"({"landed": {}})"), "metrics.landed.kind", "a metric without kind");
    ExpectError(TaskWithMetrics(R"({"stop m": {"kind": "number"}})"), "metric names must be identifiers",
        "a metric name that is not an identifier");
}

void GeneratesReadableHeader()
{
    Check(ppokit::schema::ToPascalCase("glide_path_error") == "GlidePathError", "snake_case to PascalCase");
    Check(ppokit::schema::ToPascalCase("Path") == "Path", "PascalCase is kept");
    auto const schema = ParseTaskSchema(MinimalTask(R"({"own": {"fields": ["speed"]}})"), "task.json");
    std::string const header = ppokit::schema::GenerateTaskLayoutHeader(schema, "task.json");
    Check(header.find("// Generated by ppokit-codegen from task.json.") == 0, "generated banner");
    Check(header.find("namespace ppokit::tasks::minimal {") != std::string::npos, "namespace");
    Check(header.find("enum class Slot : std::size_t") != std::string::npos, "slot enum for fully named layouts");
    Check(header == ppokit::schema::GenerateTaskLayoutHeader(schema, "task.json"), "deterministic output");
    bool rejected = false;
    try
    {
        ppokit::schema::GenerateTaskLayoutHeader(
            ParseTaskSchema(MinimalTask(R"({"size": {"size": 1}})"), "task.json"), "task.json");
    }
    catch (SchemaError const& error)
    {
        rejected = std::string(error.what()).find("reserved") != std::string::npos;
    }
    Check(rejected, "a group named like a generated member is rejected with a clear message");
}

// The fixture header is generated at build time by ppokit_generate_task_layout.
using Sample = ppokit::tests::sample::TaskLayout;
static_assert(Sample::MaximumAgents == 4);
static_assert(Sample::Observation::Size == 2 + 5 + 3 * 3);
static_assert(Sample::Observation::Lidar::Offset == 2 && Sample::Observation::RedTracks::Offset == 7);
static_assert(Sample::Observation::RedTracks::MaskFeature == 2);
static_assert(static_cast<std::size_t>(Sample::Observation::Own::Feature::HeadingError) == 1);
static_assert(Sample::Critic::Size == 2);
static_assert(Sample::Action::Size == 3 && Sample::Action::Brake::Offset == 2);
static_assert(!Sample::Action::Brake::Clipped && Sample::Action::Steer::Clipped);
static_assert(static_cast<std::size_t>(Sample::Action::Component::Brake0) == 2);
static_assert(Sample::Reward::Count == 2 && Sample::Reward::DefaultWeights[0] == 0.5);

void GeneratedFixtureMatchesRuntimeParse()
{
    Check(std::strcmp(Sample::Reward::Labels[0], "Patrol progress") == 0, "generated reward label");
    Check(std::strcmp(Sample::Action::ComponentNames[2], "brake_0") == 0, "generated component name");
    auto const schema = ppokit::schema::LoadTaskSchema(PPOKIT_SAMPLE_TASK);
    Check(schema.shapeHash == Sample::ShapeHash, "generated hash equals the runtime hash");
}

} // namespace

/** @brief Inputs of fixed shape may have a different physical meaning. */
void SemanticChanges()
{
    auto const original = ppokit::schema::LoadTaskSchema(PPOKIT_SAMPLE_TASK);
    auto const check = [&](auto changed) {
        Check(ppokit::schema::ShapeHash(original) == ppokit::schema::ShapeHash(changed)
                && ppokit::schema::SemanticHash(original) != ppokit::schema::SemanticHash(changed),
            "semantic edits preserve shape and change the policy contract");
    };
    auto task = original;
    task.actions.front().upperBound += 1.0;
    check(task);
    task = original;
    task.observation.groups.front().convention = "NED, metres";
    check(task);
    task = original;
    task.activities.push_back({"axis", 0, 0, ppokit::schema::ActivityTest::Above, 0.25});
    check(task);
}

int main()
{
    ReadsEpisodeMetrics();
    ParsesLayoutsInWrittenOrder();
    ReportsFaultyKeys();
    ShapeHashTracksShapesOnly();
    SemanticChanges();
    GeneratesReadableHeader();
    GeneratedFixtureMatchesRuntimeParse();
    if (failures != 0)
    {
        std::cerr << failures << " task schema check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit task schema tests passed\n";
    return EXIT_SUCCESS;
}
