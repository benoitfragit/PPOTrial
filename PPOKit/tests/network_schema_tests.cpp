/**
 * @file PPOKit/tests/network_schema_tests.cpp
 * @brief graph.json parsing, validation messages and generated rl_tools network header.
 */
#include "ppokit/schema/network_schema.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using ppokit::schema::CriticInput;
using ppokit::schema::NetworkModel;
using ppokit::schema::ParseNetworkSchema;
using ppokit::schema::ParseTaskSchema;
using ppokit::schema::SchemaError;
using ppokit::schema::TaskSchema;

int failures = 0;

void Check(bool condition, char const* description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/** @brief Task with 3 observations and 2 actions, optionally with a 4-value critic input. */
TaskSchema Task(bool withCritic)
{
    std::string text = R"({"schema": "ppokit.task/1", "name": "small", "observation": {"own": {"size": 3}},)"
                       R"("actions": {"move": {"head": "gaussian", "size": 2}}, "rewards": {"R": {}})";
    if (withCritic)
    {
        text += R"(, "critic": {"state": {"size": 4}})";
    }
    return ParseTaskSchema(text + "}", "task.json");
}

std::string Graph(std::string const& actor, std::string const& critic)
{
    return R"({"schema": "ppokit.graph/1", "actor": )" + actor + R"(, "critic": )" + critic + "}";
}

std::string const Mlp = R"({"model": "mlp", "hidden": {"layers": 2, "width": 16}, "activation": "relu"})";

void ExpectError(std::string const& graph, TaskSchema const& task, std::string const& expected,
    char const* description)
{
    try
    {
        ParseNetworkSchema(graph, "graph.json", task);
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

void SizesComeFromTheTask()
{
    auto const decentralized = ParseNetworkSchema(Graph(Mlp, Mlp), "graph.json", Task(false));
    Check(decentralized.actor.inputs == 3 && decentralized.actor.outputs == 2, "actor sizes from the task");
    Check(decentralized.critic.input == CriticInput::Observation && decentralized.critic.inputs == 3,
        "without critic input, the critic reads the observation");
    Check(decentralized.actor.mlp.hiddenLayers == 2 && decentralized.actor.mlp.width == 16, "hidden layers");
    Check(decentralized.cppNamespace == "ppokit::tasks::small::networks", "default namespace");
    auto const centralized = ParseNetworkSchema(Graph(R"({"model": "external"})", Mlp), "graph.json", Task(true));
    Check(centralized.critic.input == CriticInput::Critic && centralized.critic.inputs == 4,
        "a task with a critic input gets a centralized critic by default");
    Check(centralized.actor.model == NetworkModel::External, "external actor");
}

void ReportsFaultyKeys()
{
    auto const task = Task(false);
    ExpectError(R"({"schema": "ppokit.graph/0"})", task, "graph.json: schema: expected \"ppokit.graph/1\"",
        "wrong version");
    ExpectError(Graph(R"({"model": "mlp"})", Mlp), task, "actor.hidden: missing", "missing hidden layers");
    ExpectError(Graph(R"({"model": "mlp", "hidden": {"layers": 2, "width": 0}})", Mlp), task,
        "actor.hidden.width: expected a positive integer", "zero width");
    ExpectError(Graph(R"({"model": "mlp", "hidden": {"layers": 2, "width": 8}, "activation": "gelu"})", Mlp), task,
        "actor.activation: expected \"tanh\", \"relu\" or \"identity\"", "unsupported activation");
    ExpectError(Graph(Mlp, R"({"model": "mlp", "input": "critic", "hidden": {"layers": 1, "width": 8}})"), task,
        "critic.input: the task has no \"critic\" input", "critic input missing from the task");
    ExpectError(Graph(R"({"model": "external", "hidden": {"layers": 1, "width": 8}})", Mlp), task,
        "actor.hidden: unknown key", "an external network takes no MLP settings");
    ExpectError(Graph(R"({"model": "transformer"})", Mlp), task, "actor.model: expected \"mlp\" or \"external\"",
        "unknown model");
}

void GeneratesRlToolsTypes()
{
    auto const task = Task(true);
    auto const networks = ParseNetworkSchema(Graph(R"({"model": "external"})", Mlp), "graph.json", task);
    std::string const header = ppokit::schema::GenerateNetworksHeader(networks, task, "graph.json");
    Check(header.find("#error \"Include the rl_tools operation headers") != std::string::npos,
        "guard against a missing rl_tools include");
    Check(header.find("namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;") != std::string::npos,
        "follows the namespace wrapper of the including file");
    Check(header.find("struct Actor") == std::string::npos, "nothing generated for an external actor");
    Check(header.find("rlt::nn::activation_functions::RELU") != std::string::npos, "critic activation");
    Check(header.find("static constexpr std::size_t Inputs = 4;") != std::string::npos, "critic input size");
}

} // namespace

int main()
{
    SizesComeFromTheTask();
    ReportsFaultyKeys();
    GeneratesRlToolsTypes();
    if (failures != 0)
    {
        std::cerr << failures << " network schema check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit network schema tests passed\n";
    return EXIT_SUCCESS;
}
