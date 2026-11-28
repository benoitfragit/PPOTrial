/**
 * @file PPOKit/src/artifacts/artifact.cpp
 * @brief The artifact header line and the plain-language refusal of incompatible checkpoints.
 */
#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/artifacts/atomic_file.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>

namespace ppokit::artifacts {
namespace {

using Json = nlohmann::ordered_json;

constexpr char const* ArtifactSchema = "ppokit.artifact/1";

/** @brief Hashes as 16 hex digits: JSON numbers would lose bits above 2^53 in some readers. */
std::string Hex(std::uint64_t value)
{
    std::ostringstream text;
    text << std::hex << std::setw(16) << std::setfill('0') << value;
    return text.str();
}

std::uint64_t FromHex(Json const& value)
{
    return std::stoull(value.get<std::string>(), nullptr, 16);
}

std::string Joined(std::vector<std::string> const& values)
{
    std::string text;
    for (std::string const& value : values)
    {
        text += (text.empty() ? "" : ", ") + value;
    }
    return text.empty() ? std::string("(none)") : text;
}

/** @brief Blocks of `now` missing from `before`, or the reverse. */
std::vector<std::string> Missing(std::vector<std::string> const& from, std::vector<std::string> const& in)
{
    std::vector<std::string> missing;
    for (std::string const& value : from)
    {
        if (std::find(in.begin(), in.end(), value) == in.end())
        {
            missing.push_back(value);
        }
    }
    return missing;
}

/** @brief "the actor had blocks A; graph.json now has B (added: x; removed: y)". */
std::string NetworkDifference(char const* network, std::vector<std::string> const& saved,
    std::vector<std::string> const& current)
{
    if (saved == current)
    {
        return {};
    }
    std::string text = std::string("the ") + network + " had blocks " + Joined(saved) + "; graph.json now has "
        + Joined(current);
    std::vector<std::string> const added = Missing(current, saved);
    std::vector<std::string> const removed = Missing(saved, current);
    if (!added.empty() || !removed.empty())
    {
        text += " (";
        if (!added.empty())
        {
            text += "added: " + Joined(added);
        }
        if (!removed.empty())
        {
            text += std::string(added.empty() ? "" : "; ") + "removed: " + Joined(removed);
        }
        text += ")";
    }
    return text;
}

Json IdentityJson(ArtifactIdentity const& identity)
{
    Json components = Json::array();
    for (auto const& [name, hash] : identity.components)
    {
        components.push_back(Json{{"component", name}, {"hash", Hex(hash)}});
    }
    Json json{{"task", identity.task}, {"task_hash", Hex(identity.taskHash)}, {"graph_hash", Hex(identity.graphHash)},
        {"backend", identity.backend}, {"scalar", identity.scalar}, {"observation_size", identity.observationSize},
        {"action_size", identity.actionSize}, {"actor_blocks", identity.actorBlocks},
        {"critic_blocks", identity.criticBlocks}, {"components", std::move(components)}};
    if (identity.semanticHash)
    {
        json["semantic_v1"] = Hex(*identity.semanticHash);
    }
    // Written only when known, so the headers of tasks without one keep their form.
    if (!identity.contract.empty())
    {
        json["contract"] = identity.contract;
    }
    return json;
}

ArtifactIdentity ReadIdentity(Json const& json)
{
    ArtifactIdentity identity;
    identity.task = json.at("task").get<std::string>();
    identity.taskHash = FromHex(json.at("task_hash"));
    identity.graphHash = FromHex(json.at("graph_hash"));
    if (json.contains("semantic_v1"))
    {
        identity.semanticHash = FromHex(json.at("semantic_v1"));
    }
    identity.backend = json.at("backend").get<std::string>();
    identity.scalar = json.at("scalar").get<std::string>();
    identity.observationSize = json.at("observation_size").get<std::size_t>();
    identity.actionSize = json.at("action_size").get<std::size_t>();
    identity.actorBlocks = json.at("actor_blocks").get<std::vector<std::string>>();
    identity.criticBlocks = json.at("critic_blocks").get<std::vector<std::string>>();
    for (Json const& component : json.at("components"))
    {
        identity.components.emplace_back(component.at("component").get<std::string>(), FromHex(component.at("hash")));
    }
    identity.contract = json.value("contract", std::string());
    return identity;
}

} // namespace

char const* ArtifactKindName(ArtifactKind kind) noexcept
{
    return kind == ArtifactKind::Training ? "training" : "policy";
}

std::string HeaderLine(ArtifactHeader const& header)
{
    Json json = Json::object();
    json["schema"] = ArtifactSchema;
    json["kind"] = ArtifactKindName(header.kind);
    json["identity"] = IdentityJson(header.identity);
    json["updates"] = header.updates;
    json["seed"] = header.seed;
    json["created"] = header.created;
    return json.dump() + "\n";
}

ArtifactHeader ParseHeaderLine(std::string_view line, std::string const& source)
{
    Json const json = Json::parse(line, nullptr, false);
    if (!json.is_object() || json.value("schema", std::string()) != ArtifactSchema)
    {
        throw std::runtime_error(source + ": not a PPOKit artifact (expected a first line with \"schema\": \""
            + ArtifactSchema + "\")");
    }
    try
    {
        ArtifactHeader header;
        std::string const kind = json.at("kind").get<std::string>();
        if (kind != "training" && kind != "policy")
        {
            throw std::runtime_error("unknown kind '" + kind + "'");
        }
        header.kind = kind == "training" ? ArtifactKind::Training : ArtifactKind::Policy;
        header.identity = ReadIdentity(json.at("identity"));
        header.updates = json.at("updates").get<std::uint64_t>();
        header.seed = json.at("seed").get<std::uint32_t>();
        header.created = json.at("created").get<std::string>();
        return header;
    }
    catch (std::exception const& error)
    {
        throw std::runtime_error(source + ": damaged artifact header (" + error.what() + ")");
    }
}

ArtifactHeader ReadCompatibleHeader(std::istream& input, ArtifactIdentity const& expected,
    std::string const& source, bool requireTraining)
{
    std::string line;
    std::getline(input, line);
    ArtifactHeader const header = ParseHeaderLine(line, source);
    RequireCompatible(header.identity, expected, source);
    if (requireTraining && header.kind != ArtifactKind::Training)
    {
        throw std::runtime_error(source + ": a policy checkpoint has no critic or optimizer to resume from; load "
            "it with LoadPolicy, or resume from a training checkpoint");
    }
    return header;
}

void RequirePayloadEnd(std::istream& input, std::string const& source)
{
    std::string trailing;
    if (input >> trailing)
    {
        throw std::runtime_error(source + ": damaged checkpoint: trailing data '" + trailing + "'");
    }
    if (input.bad() || !input.eof())
    {
        throw std::runtime_error(source + ": damaged checkpoint: cannot read the end of the payload");
    }
}

ArtifactHeader ReadArtifactHeader(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error(path.string() + ": cannot open the file");
    }
    std::string line;
    std::getline(stream, line);
    return ParseHeaderLine(line, path.filename().string());
}

void RequireCompatible(ArtifactIdentity const& saved, ArtifactIdentity const& current, std::string const& source)
{
    std::vector<std::string> reasons;
    if (saved.task != current.task)
    {
        reasons.push_back("it was trained for the task '" + saved.task + "', this is '" + current.task + "'");
    }
    if (saved.backend != current.backend || saved.scalar != current.scalar)
    {
        reasons.push_back("it holds " + saved.backend + "/" + saved.scalar + " networks, this program uses "
            + current.backend + "/" + current.scalar);
    }
    if (saved.taskHash != current.taskHash)
    {
        std::string reason = "task.json changed its shapes or names";
        if (saved.observationSize != current.observationSize || saved.actionSize != current.actionSize)
        {
            reason += " (observation " + std::to_string(saved.observationSize) + " -> "
                + std::to_string(current.observationSize) + " values, actions " + std::to_string(saved.actionSize)
                + " -> " + std::to_string(current.actionSize) + ")";
        }
        reasons.push_back(reason);
    }
    if (current.backend == "graph" && current.semanticHash && !saved.semanticHash)
    {
        reasons.emplace_back("missing semantic_v1: migrate the legacy header with its original task and graph");
    }
    if (saved.semanticHash && current.semanticHash && saved.semanticHash != current.semanticHash)
    {
        reasons.emplace_back("policy semantics differ (observation convention, action mapping, bounds or masks)");
    }
    if (saved.graphHash != current.graphHash)
    {
        std::string const actor = NetworkDifference("actor", saved.actorBlocks, current.actorBlocks);
        std::string const critic = NetworkDifference("critic", saved.criticBlocks, current.criticBlocks);
        std::string reason = "trained for another network: ";
        if (!actor.empty() || !critic.empty())
        {
            reason += actor + (actor.empty() || critic.empty() ? "" : "; ") + critic;
        }
        else if (saved.components != current.components)
        {
            reason += "a component library was rebuilt or reconfigured";
        }
        else
        {
            reason += "the same blocks with other widths, settings or wiring";
        }
        reasons.push_back(reason);
    }
    bool const otherEnvironment =
        !saved.contract.empty() && !current.contract.empty() && saved.contract != current.contract;
    if (otherEnvironment)
    {
        reasons.push_back("it was trained in the environment '" + saved.contract + "', this project's is '"
            + current.contract + "' (to start from it here on purpose: ppokit-run transfer)");
    }
    if (reasons.empty())
    {
        return;
    }
    std::string message = source + ": cannot be loaded: ";
    for (std::size_t index = 0; index < reasons.size(); ++index)
    {
        message += (index == 0 ? "" : "; ") + reasons[index];
    }
    throw ArtifactMismatch(message + ". Train a new model, or go back to the files it was trained with.");
}

void RetargetArtifact(std::filesystem::path const& source, std::filesystem::path const& target,
    std::string const& contract)
{
    if (std::filesystem::exists(target))
    {
        throw std::runtime_error(target.string() + " exists: nothing is overwritten (train --fresh archives a run)");
    }
    std::ifstream stream(source, std::ios::binary);
    std::string line;
    if (!stream || !std::getline(stream, line))
    {
        throw std::runtime_error(source.string() + ": cannot read");
    }
    ArtifactHeader header = ParseHeaderLine(line, source.filename().string());
    header.identity.contract = contract;
    std::string const body((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    WriteFileAtomically(
        target, [&](std::ostream& out) { out << HeaderLine(header) << body; }, AtomicWriteOptions::Binary(),
        "transferred artifact");
}

std::string UtcNow()
{
    std::time_t const now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char text[32];
    std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return text;
}

} // namespace ppokit::artifacts
