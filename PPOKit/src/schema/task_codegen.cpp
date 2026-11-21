/**
 * @file PPOKit/src/schema/task_codegen.cpp
 * @brief Writes the constants-only C++ layout header of a task.
 * @details The output is deterministic (same schema, same bytes), documented and
 * formatted like hand-written code, because example authors read it to find offsets.
 */
#include "ppokit/schema/task_codegen.hpp"

#include "code_writer.hpp"

#include <cctype>
#include <cstdio>
#include <set>
#include <sstream>
#include <vector>

namespace ppokit::schema {
namespace {

using detail::CodeWriter;
using detail::DoubleLiteral;
using detail::Quoted;

/** @brief Names used by the generated structs themselves; groups may not take them. */
std::set<std::string> const ReservedMemberNames{"Size", "Slot", "SlotNames", "Component", "ComponentNames",
    "Channel", "Count", "Names", "Labels", "DefaultWeights"};

std::string HexLiteral(std::uint64_t value)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "0x%016llxULL", static_cast<unsigned long long>(value));
    return buffer;
}

/** @brief Converts names to PascalCase enumerators and rejects collisions. */
std::vector<std::string> UniquePascalNames(std::vector<std::string> const& names, std::string const& context)
{
    std::vector<std::string> converted;
    std::set<std::string> seen;
    for (auto const& name : names)
    {
        std::string pascal = ToPascalCase(name);
        if (!seen.insert(pascal).second)
        {
            throw SchemaError(context + ": names collide once converted to C++ ('" + pascal + "')");
        }
        converted.push_back(std::move(pascal));
    }
    return converted;
}

/** @brief Writes `enum class <type> : std::size_t { A, B, Count };`. */
void WriteEnum(CodeWriter& code, std::string const& type, std::vector<std::string> const& enumerators)
{
    code.Open("enum class " + type + " : std::size_t");
    for (auto const& enumerator : enumerators)
    {
        code.Line(enumerator + ",");
    }
    code.Line("Count");
    code.Close(";");
}

/** @brief Writes `enum class <type> : std::size_t { A, B };` without a Count enumerator. */
void WriteEnumWithoutCount(CodeWriter& code, std::string const& type, std::vector<std::string> const& enumerators)
{
    code.Open("enum class " + type + " : std::size_t");
    for (auto const& enumerator : enumerators)
    {
        code.Line(enumerator + ",");
    }
    code.Close(";");
}

/** @brief Writes `static constexpr std::array<char const*, N> <name>{"a", "b"};`. */
void WriteNameArray(CodeWriter& code, std::string const& name, std::string const& count,
    std::vector<std::string> const& values)
{
    code.Open("static constexpr std::array<char const*, " + count + "> " + name);
    for (auto const& value : values)
    {
        code.Line(Quoted(value) + ",");
    }
    code.Close(";");
}

std::string GroupStructName(std::string const& groupName, std::string const& context)
{
    std::string name = ToPascalCase(groupName);
    if (ReservedMemberNames.count(name) != 0)
    {
        throw SchemaError(context + "." + groupName + ": the group name '" + name + "' is reserved, rename the group");
    }
    return name;
}

void WriteInputGroup(CodeWriter& code, ObservationGroup const& group, std::string const& context)
{
    if (group.isSet)
    {
        code.Line("/** @brief Set \"" + group.name + "\": up to " + std::to_string(group.capacity) + " entities of "
            + std::to_string(group.features) + " features. */");
    }
    else
    {
        code.Line("/** @brief Group \"" + group.name + "\": " + std::to_string(group.features) + " features. */");
    }
    code.Open("struct " + GroupStructName(group.name, context) + " final");
    code.Line("static constexpr std::size_t Offset = " + std::to_string(group.offset) + ";");
    code.Line("static constexpr std::size_t Size = " + std::to_string(group.Size()) + ";");
    if (group.isSet)
    {
        code.Line("static constexpr std::size_t Capacity = " + std::to_string(group.capacity) + ";");
        code.Line("static constexpr std::size_t Features = " + std::to_string(group.features) + ";");
        if (group.maskFeature)
        {
            code.Line("/** @brief Feature of each entity that flags its presence. */");
            code.Line("static constexpr std::size_t MaskFeature = " + std::to_string(*group.maskFeature) + ";");
        }
    }
    if (!group.featureNames.empty())
    {
        WriteEnum(code, "Feature", UniquePascalNames(group.featureNames, context + "." + group.name));
    }
    code.Close(";");
}

/** @brief True when every value of the layout has its own name (no set, no anonymous vector). */
bool HasNamedSlots(InputLayout const& layout) noexcept
{
    for (auto const& group : layout.groups)
    {
        if (group.isSet || group.featureNames.empty())
        {
            return false;
        }
    }
    return true;
}

void WriteInputLayout(CodeWriter& code, std::string const& structName, std::string const& description,
    InputLayout const& layout, std::string const& context)
{
    code.Line("/** @brief " + description + " */");
    code.Open("struct " + structName + " final");
    code.Line("static constexpr std::size_t Size = " + std::to_string(layout.size) + ";");
    for (auto const& group : layout.groups)
    {
        code.Line();
        WriteInputGroup(code, group, context);
    }
    if (HasNamedSlots(layout))
    {
        std::vector<std::string> names;
        for (auto const& group : layout.groups)
        {
            names.insert(names.end(), group.featureNames.begin(), group.featureNames.end());
        }
        code.Line();
        code.Line("/** @brief Every slot of the flat vector, in order. */");
        WriteEnum(code, "Slot", UniquePascalNames(names, context));
        WriteNameArray(code, "SlotNames", "Size", names);
    }
    code.Close(";");
}

/** @brief Writes the struct of one action group; bounds and clipping only exist for Gaussian heads. */
void WriteActionGroup(CodeWriter& code, ActionGroup const& group)
{
    bool const gaussian = group.head == ActionHead::Gaussian;
    std::string const head =
        gaussian ? "Gaussian head (one mean per component)" : "Bernoulli head (one logit per component)";
    code.Line("/** @brief Group \"" + group.name + "\": " + head + ", " + std::to_string(group.Size())
        + " components. */");
    code.Open("struct " + GroupStructName(group.name, "actions") + " final");
    code.Line("static constexpr std::size_t Offset = " + std::to_string(group.offset) + ";");
    code.Line("static constexpr std::size_t Size = " + std::to_string(group.Size()) + ";");
    code.Line(std::string("static constexpr bool Gaussian = ") + (gaussian ? "true" : "false") + ";");
    if (gaussian)
    {
        code.Line("static constexpr double LowerBound = " + DoubleLiteral(group.lowerBound) + ";");
        code.Line("static constexpr double UpperBound = " + DoubleLiteral(group.upperBound) + ";");
        bool const clipped = group.mapping == ActionMapping::Clip;
        code.Line("/** @brief True when the sample is clipped to the bounds before use. */");
        code.Line(std::string("static constexpr bool Clipped = ") + (clipped ? "true" : "false") + ";");
    }
    code.Close(";");
}

void WriteActions(CodeWriter& code, TaskSchema const& schema)
{
    code.Line("/** @brief Action vector: one actor output per component (Gaussian mean or Bernoulli logit). */");
    code.Open("struct Action final");
    code.Line("static constexpr std::size_t Size = " + std::to_string(schema.actionSize) + ";");
    std::vector<std::string> names;
    for (auto const& group : schema.actions)
    {
        code.Line();
        WriteActionGroup(code, group);
        names.insert(names.end(), group.componentNames.begin(), group.componentNames.end());
    }
    code.Line();
    WriteEnum(code, "Component", UniquePascalNames(names, "actions"));
    WriteNameArray(code, "ComponentNames", "Size", names);
    code.Close(";");
}

/** @brief "wind_kt" -> "windKt": field names of the generated Scenario struct. */
std::string ToCamelCase(std::string const& identifier)
{
    std::string camel = ToPascalCase(identifier);
    if (!camel.empty())
    {
        camel.front() = static_cast<char>(std::tolower(static_cast<unsigned char>(camel.front())));
    }
    return camel;
}

/** @brief C++ type and conversion from a curriculum value (a double) for one parameter. */
std::string FieldType(ScenarioParameter const& parameter)
{
    switch (parameter.type)
    {
    case ParameterType::Int:
        return "int";
    case ParameterType::Bool:
        return "bool";
    case ParameterType::Enum:
        return ToPascalCase(parameter.name);
    case ParameterType::Float:
        break;
    }
    return "double";
}

std::string FieldConversion(ScenarioParameter const& parameter, std::string const& value)
{
    switch (parameter.type)
    {
    case ParameterType::Int:
        return "static_cast<int>(" + value + ")";
    case ParameterType::Bool:
        return value + " != 0.0";
    case ParameterType::Enum:
        return "static_cast<" + ToPascalCase(parameter.name) + ">(static_cast<std::size_t>(" + value + "))";
    case ParameterType::Float:
        break;
    }
    return value;
}

/** @brief Typed scenario struct, filled from the values drawn by ppokit::runtime::Curriculum. */
void WriteScenario(CodeWriter& code, TaskSchema const& schema)
{
    std::vector<std::string> names;
    std::set<std::string> reserved{"Count", "Names", "From"};
    for (auto const& parameter : schema.scenario)
    {
        names.push_back(parameter.name);
        if (parameter.type == ParameterType::Enum && reserved.count(ToPascalCase(parameter.name)) != 0)
        {
            throw SchemaError("scenario." + parameter.name + ": the name is reserved, rename the parameter");
        }
    }
    UniquePascalNames(names, "scenario");
    code.Line("/** @brief Scenario of one episode (task.json \"scenario\"), typed. */");
    code.Open("struct Scenario final");
    code.Line("static constexpr std::size_t Count = " + std::to_string(names.size()) + ";");
    WriteNameArray(code, "Names", "Count", names);
    for (auto const& parameter : schema.scenario)
    {
        if (parameter.type == ParameterType::Enum)
        {
            WriteEnumWithoutCount(code, ToPascalCase(parameter.name), UniquePascalNames(parameter.values,
                "scenario." + parameter.name));
        }
    }
    code.Line();
    for (auto const& parameter : schema.scenario)
    {
        code.Line(FieldType(parameter) + " " + ToCamelCase(parameter.name) + "{};");
    }
    code.Line();
    code.Line("/** @brief Converts curriculum values, in task.json order, into typed fields. */");
    code.Line("template<typename Values>");
    code.Open("static Scenario From(Values const& values)");
    code.Line("Scenario scenario;");
    for (std::size_t index = 0; index < schema.scenario.size(); ++index)
    {
        auto const& parameter = schema.scenario[index];
        code.Line("scenario." + ToCamelCase(parameter.name) + " = "
            + FieldConversion(parameter, "values[" + std::to_string(index) + "]") + ";");
    }
    code.Line("return scenario;");
    code.Close();
    code.Close(";");
}

void WriteRewards(CodeWriter& code, TaskSchema const& schema)
{
    std::vector<std::string> names;
    std::vector<std::string> labels;
    std::string weights;
    for (auto const& channel : schema.rewards)
    {
        names.push_back(channel.name);
        labels.push_back(channel.label);
        weights += (weights.empty() ? "" : ", ") + DoubleLiteral(channel.weight);
    }
    code.Line("/** @brief Reward channels; the reward is the weighted sum of the channels. */");
    code.Open("struct Reward final");
    WriteEnum(code, "Channel", UniquePascalNames(names, "rewards"));
    code.Line("static constexpr std::size_t Count = " + std::to_string(names.size()) + ";");
    WriteNameArray(code, "Names", "Count", names);
    WriteNameArray(code, "Labels", "Count", labels);
    code.Line("static constexpr std::array<double, Count> DefaultWeights{" + weights + "};");
    code.Close(";");
}

} // namespace

std::string ToPascalCase(std::string_view identifier)
{
    std::string result;
    bool upperNext = true;
    for (char const character : identifier)
    {
        if (character == '_')
        {
            upperNext = true;
            continue;
        }
        if (upperNext)
        {
            result += static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
            upperNext = false;
        }
        else
        {
            result += character;
        }
    }
    return result;
}

std::string GenerateTaskLayoutHeader(TaskSchema const& schema, std::string_view sourceName)
{
    CodeWriter code;
    code.Line("// Generated by ppokit-codegen from " + std::string(sourceName) + ".");
    code.Line("// Do not edit: change the task file instead.");
    code.Line("#pragma once");
    code.Line();
    code.Line("#include <array>");
    code.Line("#include <cstddef>");
    code.Line("#include <cstdint>");
    code.Line();
    code.Line("namespace " + schema.cppNamespace + " {");
    code.Line();
    code.Line("/** @brief Layout of the \"" + schema.name + "\" task. */");
    code.Open("struct TaskLayout final");
    code.Line("static constexpr char const* Name = " + Quoted(schema.name) + ";");
    code.Line("/** @brief Hash of every shape and name a trained network depends on. */");
    code.Line("static constexpr std::uint64_t ShapeHash = " + HexLiteral(schema.shapeHash) + ";");
    code.Line("static constexpr std::size_t MinimumAgents = " + std::to_string(schema.minimumAgents) + ";");
    code.Line("static constexpr std::size_t MaximumAgents = " + std::to_string(schema.maximumAgents) + ";");
    code.Line();
    WriteInputLayout(code, "Observation", "Actor observation.", schema.observation, "observation");
    if (schema.critic)
    {
        code.Line();
        WriteInputLayout(code, "Critic", "Critic input.", *schema.critic, "critic");
    }
    code.Line();
    WriteActions(code, schema);
    code.Line();
    WriteRewards(code, schema);
    if (!schema.scenario.empty())
    {
        code.Line();
        WriteScenario(code, schema);
    }
    code.Close(";");
    code.Line();
    code.Line("} // namespace " + schema.cppNamespace);
    return code.Text();
}

} // namespace ppokit::schema
