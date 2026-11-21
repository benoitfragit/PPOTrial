/**
 * @file PPOKit/src/schema/network_schema.cpp
 * @brief Parsing of `graph.json` and generation of the rl_tools network types.
 * @details The generated types are exactly the rl_tools types the examples wrote
 * by hand (StructureSpecification + NeuralNetworkAdam), so a network built from
 * them computes bit for bit what the hand-written network computes.
 */
#include "ppokit/schema/network_schema.hpp"

#include "code_writer.hpp"
#include "json_validation.hpp"

#include <string>

namespace ppokit::schema {
namespace {

using detail::CodeWriter;
using detail::ErrorReporter;
using detail::IsQualifiedName;
using detail::Json;
using detail::RequireKnownKeys;
using detail::RequireObject;
using detail::RequirePositiveInteger;

/** @brief Format identifier accepted in the "schema" key. */
constexpr char const* NetworkSchemaVersion = "ppokit.graph/1";

NetworkModel ReadModel(Json const& network, std::string const& path, ErrorReporter const& errors)
{
    if (!network.contains("model"))
    {
        errors.Fail(path + ".model", "missing (\"mlp\" or \"external\")");
    }
    Json const& model = network.at("model");
    if (model == "mlp")
    {
        return NetworkModel::Mlp;
    }
    if (model == "external")
    {
        return NetworkModel::External;
    }
    errors.Fail(path + ".model", "expected \"mlp\" or \"external\"");
}

Activation ReadActivation(Json const& network, std::string const& path, ErrorReporter const& errors)
{
    if (!network.contains("activation"))
    {
        return Activation::Tanh;
    }
    Json const& activation = network.at("activation");
    if (activation == "tanh")
    {
        return Activation::Tanh;
    }
    if (activation == "relu")
    {
        return Activation::Relu;
    }
    if (activation == "identity")
    {
        return Activation::Identity;
    }
    errors.Fail(path + ".activation", "expected \"tanh\", \"relu\" or \"identity\"");
}

/** @brief Reads `{"layers": n, "width": w}`; rl_tools MLPs use one width for every hidden layer. */
void ReadHidden(Json const& network, MlpSpec& mlp, std::string const& path, ErrorReporter const& errors)
{
    std::string const hiddenPath = path + ".hidden";
    if (!network.contains("hidden"))
    {
        errors.Fail(hiddenPath, "missing, e.g. { \"layers\": 2, \"width\": 64 }");
    }
    Json const& hidden = network.at("hidden");
    RequireObject(hidden, hiddenPath, errors);
    RequireKnownKeys(hidden, {"layers", "width"}, hiddenPath, errors);
    if (!hidden.contains("layers") || !hidden.contains("width"))
    {
        errors.Fail(hiddenPath, "expected { \"layers\": ..., \"width\": ... } (one width for every hidden layer)");
    }
    mlp.hiddenLayers = RequirePositiveInteger(hidden.at("layers"), hiddenPath + ".layers", errors);
    mlp.width = RequirePositiveInteger(hidden.at("width"), hiddenPath + ".width", errors);
}

ActorSpec ReadActor(Json const& value, TaskSchema const& task, ErrorReporter const& errors)
{
    RequireObject(value, "actor", errors);
    Json const& actor = value;
    ActorSpec spec;
    spec.inputs = task.observation.size;
    spec.outputs = task.actionSize;
    spec.model = ReadModel(actor, "actor", errors);
    if (spec.model == NetworkModel::External)
    {
        RequireKnownKeys(actor, {"model"}, "actor", errors);
        return spec;
    }
    RequireKnownKeys(actor, {"model", "hidden", "activation", "log_std"}, "actor", errors);
    ReadHidden(actor, spec.mlp, "actor", errors);
    spec.mlp.activation = ReadActivation(actor, "actor", errors);
    if (actor.contains("log_std") && actor.at("log_std") != "state_independent")
    {
        errors.Fail("actor.log_std", "expected \"state_independent\" (one learned log-std per action component)");
    }
    return spec;
}

CriticSpec ReadCritic(Json const& value, TaskSchema const& task, ErrorReporter const& errors)
{
    RequireObject(value, "critic", errors);
    Json const& critic = value;
    CriticSpec spec;
    spec.model = ReadModel(critic, "critic", errors);
    // A task with a dedicated critic input is centralized by default.
    spec.input = task.critic ? CriticInput::Critic : CriticInput::Observation;
    if (critic.contains("input"))
    {
        Json const& input = critic.at("input");
        if (input == "observation")
        {
            spec.input = CriticInput::Observation;
        }
        else if (input == "critic" && task.critic)
        {
            spec.input = CriticInput::Critic;
        }
        else if (input == "critic")
        {
            errors.Fail("critic.input", "the task has no \"critic\" input in task.json");
        }
        else
        {
            errors.Fail("critic.input", "expected \"observation\" or \"critic\"");
        }
    }
    spec.inputs = spec.input == CriticInput::Critic ? task.critic->size : task.observation.size;
    if (spec.model == NetworkModel::External)
    {
        RequireKnownKeys(critic, {"model", "input"}, "critic", errors);
        return spec;
    }
    RequireKnownKeys(critic, {"model", "input", "hidden", "activation"}, "critic", errors);
    ReadHidden(critic, spec.mlp, "critic", errors);
    spec.mlp.activation = ReadActivation(critic, "critic", errors);
    return spec;
}

char const* ActivationName(Activation activation) noexcept
{
    switch (activation)
    {
    case Activation::Identity:
        return "IDENTITY";
    case Activation::Relu:
        return "RELU";
    case Activation::Tanh:
        break;
    }
    return "TANH";
}

/**
 * @brief Writes one MLP: its sizes, the rl_tools StructureSpecification and the network type.
 * @param networkTemplate rl_tools network family, e.g. "mlp_unconditional_stddev".
 */
void WriteMlp(CodeWriter& code, std::string const& name, std::string const& description, std::size_t inputs,
    std::size_t outputs, MlpSpec const& mlp, std::string const& networkTemplate)
{
    code.Line("/** @brief " + description + " */");
    code.Open("struct " + name + " final");
    code.Line("static constexpr std::size_t Inputs = " + std::to_string(inputs) + ";");
    code.Line("static constexpr std::size_t Outputs = " + std::to_string(outputs) + ";");
    code.Line("static constexpr std::size_t HiddenLayers = " + std::to_string(mlp.hiddenLayers) + ";");
    code.Line("/** @brief Width written in graph.json; another width can be chosen per instantiation. */");
    code.Line("static constexpr std::size_t DefaultWidth = " + std::to_string(mlp.width) + ";");
    code.Line();
    code.Line("/** @brief rl_tools layout; NUM_LAYERS counts the input and output layers. */");
    code.Line("template<typename Index, Index BatchSize, Index Width = DefaultWidth,");
    code.Line("    typename ContainerTag = rlt::MatrixDynamicTag>");
    code.Line("using Structure = rlt::nn_models::mlp::StructureSpecification<float, Index, Inputs, Outputs,");
    code.Line("    HiddenLayers + 1, Width, rlt::nn::activation_functions::" + std::string(ActivationName(mlp.activation))
        + ", rlt::nn::activation_functions::IDENTITY,");
    code.Line("    BatchSize, ContainerTag>;");
    code.Line();
    code.Line("/** @brief Network with its Adam moments. */");
    code.Line("template<typename Index, Index BatchSize, Index Width = DefaultWidth,");
    code.Line("    typename ContainerTag = rlt::MatrixDynamicTag>");
    code.Line("using Network = rlt::nn_models::" + networkTemplate + "::NeuralNetworkAdam<");
    code.Line("    rlt::nn_models::mlp::AdamSpecification<Structure<Index, BatchSize, Width, ContainerTag>>>;");
    code.Close(";");
}

} // namespace

NetworkSchema ParseNetworkSchema(std::string_view text, std::string_view sourceName, TaskSchema const& task)
{
    ErrorReporter const errors(sourceName);
    Json const root = detail::ParseRootObject(text, errors);
    RequireKnownKeys(root, {"schema", "namespace", "actor", "critic"}, "", errors);
    if (!root.contains("schema") || root.at("schema") != NetworkSchemaVersion)
    {
        errors.Fail("schema", std::string("expected \"") + NetworkSchemaVersion + "\"");
    }
    for (char const* required : {"actor", "critic"})
    {
        if (!root.contains(required))
        {
            errors.Fail(required, "missing required key");
        }
    }
    NetworkSchema schema;
    schema.cppNamespace = task.cppNamespace + "::networks";
    if (root.contains("namespace"))
    {
        Json const& name = root.at("namespace");
        if (!name.is_string() || !IsQualifiedName(name.get<std::string>()))
        {
            errors.Fail("namespace", "expected a C++ namespace such as \"my_project::networks\"");
        }
        schema.cppNamespace = name.get<std::string>();
    }
    schema.actor = ReadActor(root.at("actor"), task, errors);
    schema.critic = ReadCritic(root.at("critic"), task, errors);
    return schema;
}

NetworkSchema LoadNetworkSchema(std::filesystem::path const& path, TaskSchema const& task)
{
    return ParseNetworkSchema(detail::ReadTextFile(path), path.filename().string(), task);
}

std::string GenerateNetworksHeader(NetworkSchema const& networks, TaskSchema const& task, std::string_view sourceName)
{
    CodeWriter code;
    code.Line("// Generated by ppokit-codegen from " + std::string(sourceName) + " (task \"" + task.name + "\").");
    code.Line("// Do not edit: change the network file instead.");
    code.Line("//");
    code.Line("// Include the rl_tools operation headers first: this header then uses the same rl_tools");
    code.Line("// namespace, including a RL_TOOLS_NAMESPACE_WRAPPER chosen by the including file.");
    code.Line("#pragma once");
    code.Line();
    code.Line("#ifndef RL_TOOLS_RL_TOOLS_H");
    code.Line("#error \"Include the rl_tools operation headers before this generated networks header\"");
    code.Line("#endif");
    code.Line();
    code.Line("#include <cstddef>");
    code.Line();
    code.Line("namespace " + networks.cppNamespace + " {");
    code.Line();
    code.Line("namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;");
    if (networks.actor.model == NetworkModel::Mlp)
    {
        code.Line();
        WriteMlp(code, "Actor",
            "Actor: observation -> Gaussian means, with one learned log-std per action component.",
            networks.actor.inputs, networks.actor.outputs, networks.actor.mlp, "mlp_unconditional_stddev");
    }
    if (networks.critic.model == NetworkModel::Mlp)
    {
        char const* source = networks.critic.input == CriticInput::Critic ? "critic input" : "observation";
        code.Line();
        WriteMlp(code, "Critic", std::string("Critic: ") + source + " -> state value.", networks.critic.inputs, 1,
            networks.critic.mlp, "mlp");
    }
    code.Line();
    code.Line("} // namespace " + networks.cppNamespace);
    return code.Text();
}

} // namespace ppokit::schema
