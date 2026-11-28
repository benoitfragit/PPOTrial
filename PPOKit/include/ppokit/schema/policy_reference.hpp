/** @file policy_reference.hpp
 * @brief Immutable, source-attributed policies declared in a project's models.json.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::schema {
struct PolicyReference
{
    std::string role;
    std::filesystem::path file;
    std::string sha256;
    std::string sourceCommit;
    std::string sourcePath;
    std::string sourceSha256;
    std::optional<std::uint64_t> sourceUpdate;
    /** @brief A policy-only converter may serialize zero even when an archive identifies its training update. */
    std::optional<std::uint64_t> artifactUpdate;
    /** @brief Named historical evaluation suite and optional read-only environment context. */
    std::string suite;
    std::optional<std::string> environmentState;
};

/** @brief Load declared roles and their attribution. A project without a catalog has no immutable references. */
std::vector<PolicyReference> LoadPolicyReferences(std::filesystem::path const& project);
} // namespace ppokit::schema
