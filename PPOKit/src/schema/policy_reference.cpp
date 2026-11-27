/** @file policy_reference.cpp
 * @brief Validate reference metadata before presenting or opening a deployed policy.
 */
#include "ppokit/schema/policy_reference.hpp"
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace ppokit::schema {
namespace {
bool HexDigest(std::string const& text, std::size_t length) noexcept
{
    return text.size() == length && text.find_first_not_of("0123456789abcdef") == std::string::npos;
}
} // namespace

std::vector<PolicyReference> LoadPolicyReferences(std::filesystem::path const& project)
{
    auto const path = project / "models.json";
    if (!std::filesystem::exists(path)) return {};
    try
    {
        std::ifstream input(path);
        auto const json = nlohmann::json::parse(input);
        if (json.at("schema") != "ppokit.models/1" || !json.at("policies").is_object())
            throw std::runtime_error("expected ppokit.models/1 with named policies");
        std::vector<PolicyReference> refs;
        for (auto const& [role, item] : json.at("policies").items())
        {
            PolicyReference ref;
            ref.role = role;
            std::filesystem::path const relative = item.at("path").get<std::string>();
            ref.file = (project / relative).lexically_normal();
            ref.sha256 = item.at("sha256").get<std::string>();
            auto const& source = item.at("source");
            ref.sourceCommit = source.at("commit").get<std::string>();
            ref.sourcePath = source.at("path").get<std::string>();
            ref.sourceSha256 = source.at("sha256").get<std::string>();
            if (!source.at("update").is_null())
            {
                if (!source.at("update").is_number_unsigned()) throw std::runtime_error("invalid source update");
                ref.sourceUpdate = source.at("update").get<std::uint64_t>();
            }
            ref.artifactUpdate = ref.sourceUpdate;
            if (item.contains("artifact_update"))
            {
                if (!item.at("artifact_update").is_number_unsigned()) throw std::runtime_error("invalid artifact update");
                ref.artifactUpdate = item.at("artifact_update").get<std::uint64_t>();
            }
            ref.suite = item.value("suite", std::string{});
            if (item.contains("environment_state")) ref.environmentState = item.at("environment_state").dump();
            if (role.empty() || role == "BEST" || role == "LATEST" || role == "CUSTOM" || relative.empty() || relative.is_absolute() || ref.sourcePath.empty()
                || !HexDigest(ref.sha256,64) || !HexDigest(ref.sourceSha256,64) || !HexDigest(ref.sourceCommit,40))
                throw std::runtime_error("invalid reference for role " + role);
            refs.push_back(std::move(ref));
        }
        return refs;
    }
    catch (std::exception const& error)
    {
        throw std::runtime_error(path.string() + ": " + error.what());
    }
}
} // namespace ppokit::schema
