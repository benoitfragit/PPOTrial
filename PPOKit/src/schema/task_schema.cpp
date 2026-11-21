/**
 * @file PPOKit/src/schema/task_schema.cpp
 * @brief Parsing and validation of `task.json`.
 * @details Every error names the file and the dotted path of the faulty key, and says
 * what was expected, because the task file is written by hand by example authors.
 */
#include "ppokit/schema/task_schema.hpp"

#include "json_validation.hpp"

#include <set>
#include <utility>

namespace ppokit::schema {
namespace {

using detail::ChildPath;
using detail::ErrorReporter;
using detail::IsIdentifier;
using detail::IsQualifiedName;
using detail::Json;
using detail::RequireFiniteNumber;
using detail::RequireIdentifier;
using detail::RequireKnownKeys;
using detail::RequireObject;
using detail::RequirePositiveInteger;

/** @brief Format identifier accepted in the "schema" key. */
constexpr char const* TaskSchemaVersion = "ppokit.task/1";

/** @brief Reads a non-empty array of distinct identifiers. */
std::vector<std::string> RequireNameList(Json const& value, std::string const& path, ErrorReporter const& errors)
{
    if (!value.is_array() || value.empty())
    {
        errors.Fail(path, "expected a non-empty array of names");
    }
    std::vector<std::string> names;
    std::set<std::string> seen;
    for (std::size_t index = 0; index < value.size(); ++index)
    {
        std::string const itemPath = path + "[" + std::to_string(index) + "]";
        std::string name = RequireIdentifier(value[index], itemPath, errors);
        if (!seen.insert(name).second)
        {
            errors.Fail(itemPath, "duplicate name '" + name + "'");
        }
        names.push_back(std::move(name));
    }
    return names;
}

/** @brief Reads the "mask" of a set: "none", "last", a feature name or a feature index. */
std::optional<std::size_t> ReadMaskFeature(Json const& set, ObservationGroup const& group, std::string const& path,
    ErrorReporter const& errors)
{
    if (!set.contains("mask"))
    {
        return std::nullopt;
    }
    Json const& mask = set.at("mask");
    std::string const maskPath = path + ".mask";
    if (mask.is_string() && mask.get<std::string>() == "none")
    {
        return std::nullopt;
    }
    if (mask.is_string() && mask.get<std::string>() == "last")
    {
        return group.features - 1;
    }
    if (mask.is_string())
    {
        for (std::size_t index = 0; index < group.featureNames.size(); ++index)
        {
            if (group.featureNames[index] == mask.get<std::string>())
            {
                return index;
            }
        }
        errors.Fail(maskPath, "'" + mask.get<std::string>() + "' is not a feature of this set");
    }
    if (mask.is_number_unsigned() && mask.get<std::size_t>() < group.features)
    {
        return mask.get<std::size_t>();
    }
    errors.Fail(maskPath, "expected \"none\", \"last\", a feature name or a feature index");
}

/** @brief Reads the features of a group: either "fields" (names) or "size"/"features" (a count). */
void ReadFeatures(Json const& object, char const* countKey, ObservationGroup& group, std::string const& path,
    ErrorReporter const& errors)
{
    bool const hasNames = object.contains("fields");
    bool const hasCount = object.contains(countKey);
    if (hasNames == hasCount)
    {
        errors.Fail(path, std::string("give either \"fields\" (feature names) or \"") + countKey + "\" (a count)");
    }
    if (hasNames)
    {
        group.featureNames = RequireNameList(object.at("fields"), path + ".fields", errors);
        group.features = group.featureNames.size();
    }
    else
    {
        group.features = RequirePositiveInteger(object.at(countKey), path + "." + countKey, errors);
    }
}

ObservationGroup ReadGroup(std::string const& name, Json const& value, std::string const& path,
    ErrorReporter const& errors)
{
    RequireObject(value, path, errors);
    ObservationGroup group;
    group.name = name;
    if (value.contains("set"))
    {
        RequireKnownKeys(value, {"set"}, path, errors);
        std::string const setPath = path + ".set";
        Json const& set = RequireObject(value.at("set"), setPath, errors);
        RequireKnownKeys(set, {"capacity", "features", "fields", "mask"}, setPath, errors);
        if (!set.contains("capacity"))
        {
            errors.Fail(setPath, "missing \"capacity\" (maximum number of entities)");
        }
        group.isSet = true;
        group.capacity = RequirePositiveInteger(set.at("capacity"), setPath + ".capacity", errors);
        ReadFeatures(set, "features", group, setPath, errors);
        group.maskFeature = ReadMaskFeature(set, group, setPath, errors);
        return group;
    }
    RequireKnownKeys(value, {"fields", "size"}, path, errors);
    ReadFeatures(value, "size", group, path, errors);
    return group;
}

/** @brief Reads an ordered object of groups into a flat layout and assigns offsets. */
InputLayout ReadInputLayout(Json const& value, std::string const& path, ErrorReporter const& errors)
{
    RequireObject(value, path, errors);
    if (value.empty())
    {
        errors.Fail(path, "expected at least one group");
    }
    InputLayout layout;
    std::set<std::string> featureNames;
    for (auto const& item : value.items())
    {
        std::string const groupPath = path + "." + item.key();
        if (!IsIdentifier(item.key()))
        {
            errors.Fail(groupPath, "group names must be identifiers");
        }
        ObservationGroup group = ReadGroup(item.key(), item.value(), groupPath, errors);
        // Plain named fields become enumerators of one Slot enum: they must be unique.
        if (!group.isSet)
        {
            for (auto const& feature : group.featureNames)
            {
                if (!featureNames.insert(feature).second)
                {
                    errors.Fail(groupPath + ".fields", "feature '" + feature + "' already exists in another group");
                }
            }
        }
        group.offset = layout.size;
        layout.size += group.Size();
        layout.groups.push_back(std::move(group));
    }
    return layout;
}

ActionGroup ReadActionGroup(std::string const& name, Json const& value, std::string const& path,
    ErrorReporter const& errors)
{
    RequireObject(value, path, errors);
    RequireKnownKeys(value, {"head", "names", "size", "bounds", "map"}, path, errors);
    ActionGroup group;
    group.name = name;
    if (!value.contains("head") || value.at("head") != "gaussian")
    {
        errors.Fail(path + ".head", "expected \"gaussian\" (the only head available in this version)");
    }
    bool const hasNames = value.contains("names");
    if (hasNames == value.contains("size"))
    {
        errors.Fail(path, "give either \"names\" (component names) or \"size\" (a count)");
    }
    if (hasNames)
    {
        group.componentNames = RequireNameList(value.at("names"), path + ".names", errors);
    }
    else
    {
        std::size_t const size = RequirePositiveInteger(value.at("size"), path + ".size", errors);
        for (std::size_t index = 0; index < size; ++index)
        {
            group.componentNames.push_back(name + "_" + std::to_string(index));
        }
    }
    if (value.contains("bounds"))
    {
        Json const& bounds = value.at("bounds");
        if (!bounds.is_array() || bounds.size() != 2)
        {
            errors.Fail(path + ".bounds", "expected [lower, upper]");
        }
        group.lowerBound = RequireFiniteNumber(bounds[0], path + ".bounds[0]", errors);
        group.upperBound = RequireFiniteNumber(bounds[1], path + ".bounds[1]", errors);
        if (!(group.lowerBound < group.upperBound))
        {
            errors.Fail(path + ".bounds", "lower bound must be smaller than upper bound");
        }
    }
    if (value.contains("map"))
    {
        Json const& mapping = value.at("map");
        if (mapping == "clip")
        {
            group.mapping = ActionMapping::Clip;
        }
        else if (mapping == "none")
        {
            group.mapping = ActionMapping::None;
        }
        else
        {
            errors.Fail(path + ".map", "expected \"clip\" or \"none\"");
        }
    }
    return group;
}

void ReadActions(Json const& value, TaskSchema& schema, ErrorReporter const& errors)
{
    RequireObject(value, "actions", errors);
    if (value.empty())
    {
        errors.Fail("actions", "expected at least one action group");
    }
    std::set<std::string> componentNames;
    for (auto const& item : value.items())
    {
        std::string const path = "actions." + item.key();
        if (!IsIdentifier(item.key()))
        {
            errors.Fail(path, "action group names must be identifiers");
        }
        ActionGroup group = ReadActionGroup(item.key(), item.value(), path, errors);
        for (auto const& component : group.componentNames)
        {
            if (!componentNames.insert(component).second)
            {
                errors.Fail(path, "component '" + component + "' already exists in another group");
            }
        }
        group.offset = schema.actionSize;
        schema.actionSize += group.Size();
        schema.actions.push_back(std::move(group));
    }
}

void ReadRewards(Json const& value, TaskSchema& schema, ErrorReporter const& errors)
{
    RequireObject(value, "rewards", errors);
    if (value.empty())
    {
        errors.Fail("rewards", "expected at least one reward channel");
    }
    for (auto const& item : value.items())
    {
        std::string const path = "rewards." + item.key();
        if (!IsIdentifier(item.key()))
        {
            errors.Fail(path, "channel names must be identifiers; put free text in \"label\"");
        }
        RequireObject(item.value(), path, errors);
        RequireKnownKeys(item.value(), {"label", "weight"}, path, errors);
        RewardChannel channel;
        channel.name = item.key();
        channel.label = item.key();
        if (item.value().contains("label"))
        {
            if (!item.value().at("label").is_string() || item.value().at("label").get<std::string>().empty())
            {
                errors.Fail(path + ".label", "expected a non-empty string");
            }
            channel.label = item.value().at("label").get<std::string>();
        }
        if (item.value().contains("weight"))
        {
            channel.weight = RequireFiniteNumber(item.value().at("weight"), path + ".weight", errors);
        }
        schema.rewards.push_back(std::move(channel));
    }
}

void ReadAgents(Json const& value, TaskSchema& schema, ErrorReporter const& errors)
{
    RequireObject(value, "agents", errors);
    RequireKnownKeys(value, {"min", "max"}, "agents", errors);
    if (!value.contains("min") || !value.contains("max"))
    {
        errors.Fail("agents", "expected { \"min\": ..., \"max\": ... }");
    }
    schema.minimumAgents = RequirePositiveInteger(value.at("min"), "agents.min", errors);
    schema.maximumAgents = RequirePositiveInteger(value.at("max"), "agents.max", errors);
    if (schema.maximumAgents < schema.minimumAgents)
    {
        errors.Fail("agents.max", "must be greater than or equal to agents.min");
    }
}

/** @brief Appends a field of the canonical shape description, '|' separated. */
void AppendCanonical(std::string& text, std::string const& field)
{
    text += field;
    text += '|';
}

void AppendCanonicalLayout(std::string& text, char const* tag, InputLayout const& layout)
{
    AppendCanonical(text, tag);
    for (auto const& group : layout.groups)
    {
        AppendCanonical(text, group.name);
        AppendCanonical(text, std::to_string(group.features));
        AppendCanonical(text, std::to_string(group.capacity));
        AppendCanonical(text, group.isSet ? "set" : "plain");
        AppendCanonical(text, group.maskFeature ? std::to_string(*group.maskFeature) : "-");
        for (auto const& feature : group.featureNames)
        {
            AppendCanonical(text, feature);
        }
        AppendCanonical(text, ";");
    }
}

} // namespace

TaskSchema ParseTaskSchema(std::string_view text, std::string_view sourceName)
{
    ErrorReporter const errors(sourceName);
    Json const root = detail::ParseRootObject(text, errors);
    RequireKnownKeys(root, {"schema", "name", "namespace", "agents", "observation", "critic", "actions", "rewards"},
        "", errors);
    if (!root.contains("schema") || root.at("schema") != TaskSchemaVersion)
    {
        errors.Fail("schema", std::string("expected \"") + TaskSchemaVersion + "\"");
    }
    for (char const* required : {"name", "observation", "actions", "rewards"})
    {
        if (!root.contains(required))
        {
            errors.Fail(required, "missing required key");
        }
    }

    TaskSchema schema;
    schema.name = RequireIdentifier(root.at("name"), "name", errors);
    schema.cppNamespace = "ppokit::tasks::" + schema.name;
    if (root.contains("namespace"))
    {
        Json const& name = root.at("namespace");
        if (!name.is_string() || !IsQualifiedName(name.get<std::string>()))
        {
            errors.Fail("namespace", "expected a C++ namespace such as \"my_project::tasks\"");
        }
        schema.cppNamespace = name.get<std::string>();
    }
    if (root.contains("agents"))
    {
        ReadAgents(root.at("agents"), schema, errors);
    }
    schema.observation = ReadInputLayout(root.at("observation"), "observation", errors);
    if (root.contains("critic"))
    {
        schema.critic = ReadInputLayout(root.at("critic"), "critic", errors);
    }
    ReadActions(root.at("actions"), schema, errors);
    ReadRewards(root.at("rewards"), schema, errors);
    schema.shapeHash = ShapeHash(schema);
    return schema;
}

TaskSchema LoadTaskSchema(std::filesystem::path const& path)
{
    return ParseTaskSchema(detail::ReadTextFile(path), path.filename().string());
}

std::uint64_t ShapeHash(TaskSchema const& schema)
{
    std::string canonical;
    AppendCanonicalLayout(canonical, "observation", schema.observation);
    if (schema.critic)
    {
        AppendCanonicalLayout(canonical, "critic", *schema.critic);
    }
    AppendCanonical(canonical, "actions");
    for (auto const& group : schema.actions)
    {
        AppendCanonical(canonical, group.name);
        for (auto const& component : group.componentNames)
        {
            AppendCanonical(canonical, component);
        }
        AppendCanonical(canonical, ";");
    }
    AppendCanonical(canonical, "rewards");
    for (auto const& channel : schema.rewards)
    {
        AppendCanonical(canonical, channel.name);
    }
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (char const c : canonical)
    {
        hash = (hash ^ static_cast<unsigned char>(c)) * 0x100000001b3ULL;
    }
    return hash;
}

} // namespace ppokit::schema
