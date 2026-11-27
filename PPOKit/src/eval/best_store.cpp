/**
 * @file PPOKit/src/eval/best_store.cpp
 * @brief Reading and promoting the BEST policy of an evaluation suite.
 */
#include "ppokit/eval/best_store.hpp"

#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/artifacts/sha256.hpp"

#include "../schema/json_validation.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string_view>
#include <system_error>
#include <utility>

namespace ppokit::eval {
namespace {

using schema::detail::Json;

/** @brief The manifest names its policy by content (`sha256`). */
constexpr char const* BestSchemaVersion = "ppokit.best/2";
/** @brief Former manifests: the policy file was named by suite and update only. */
constexpr char const* FormerBestSchemaVersion = "ppokit.best/1";
/** @brief Hexadecimal digits of the digest kept in a generation's file name. */
constexpr std::size_t GenerationDigestDigits = 16;

std::filesystem::path BestDirectory(std::filesystem::path const& checkpoints)
{
    return checkpoints / "best";
}

/**
 * @brief The generation file of a policy: `<suite>_<update>_<digest prefix><extension>`.
 * @details Two trainings that reach the same update with different weights get different
 * files; the same weights offered twice share one.
 */
std::string GenerationName(std::string const& suite, std::uint64_t update, std::string const& digest,
    std::filesystem::path const& policyFile)
{
    return suite + "_" + std::to_string(update) + "_" + digest.substr(0, GenerationDigestDigits)
        + policyFile.extension().string();
}

/** @brief Check the names written by both BEST formats, including suites containing underscores. */
bool IsSuiteGeneration(std::string_view stem, std::string const& suite) noexcept
{
    if (stem.size() <= suite.size() || stem.compare(0, suite.size(), suite) != 0 || stem[suite.size()] != '_')
        return false;
    stem.remove_prefix(suite.size() + 1);
    auto const separator = stem.find('_');
    auto const update = stem.substr(0, separator);
    if (update.empty() || update.find_first_not_of("0123456789") != std::string_view::npos)
        return false;
    if (separator == std::string_view::npos) return true;  // best/1: <suite>_<update>
    auto const digest = stem.substr(separator + 1);
    return digest.size() == GenerationDigestDigits && digest.find_first_not_of("0123456789abcdef") == std::string_view::npos;
}

/** @brief Read the publisher's policy extension, including for suites removed from eval/. */
std::optional<std::filesystem::path> ManifestPolicyExtension(std::filesystem::path const& file)
{
    std::ifstream input(file, std::ios::binary);
    if (!input) throw std::runtime_error("cannot inspect BEST manifest " + file.string());
    auto const record = Json::parse(input, nullptr, false);
    if (!record.is_object() || !record.contains("schema") || !record.contains("suite")
        || (record.at("schema") != BestSchemaVersion && record.at("schema") != FormerBestSchemaVersion)
        || record.at("suite") != file.stem().string() || !record.contains("policy") || !record.at("policy").is_string())
        return std::nullopt;
    return std::filesystem::path(record.at("policy").get<std::string>()).extension();
}

/** @brief Best-effort deletion on a failure path, where a second error must not hide the first. */
void RemoveQuietly(std::filesystem::path const& path) noexcept
{
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

/**
 * @brief Makes `generation` a copy of `policyFile` whose digest is `digest`.
 * @details An existing generation of the same name must hold the same bytes. A new one is
 * copied to a temporary, checked against the digest (the policy file changed during the copy,
 * or the copy is damaged) and only then atomically installed: a generation is complete or absent.
 * @throws std::runtime_error when an existing generation differs or the copy does not match.
 */
void WriteGeneration(std::filesystem::path const& policyFile, std::string const& digest,
    std::filesystem::path const& generation)
{
    if (std::filesystem::exists(generation))
    {
        if (artifacts::Sha256OfFile(generation) != digest)
        {
            throw std::runtime_error("BEST generation " + generation.string()
                + " exists with other contents than the policy offered");
        }
        return;
    }
    // Clean the fixed temporary name used by best/2 before unique temporaries were introduced.
    RemoveQuietly(artifacts::SuffixedPath(generation, ".partial"));
    std::filesystem::path const temporary = artifacts::detail::UniqueTemporaryPath(generation, ".partial");
    try
    {
        std::filesystem::copy_file(policyFile, temporary);
        if (artifacts::Sha256OfFile(temporary) != digest)
        {
            throw std::runtime_error("the copy of " + policyFile.string() + " differs from the policy evaluated");
        }
        artifacts::CommitTemporaryFile(
            temporary, generation, artifacts::AtomicWriteOptions::Binary(), "BEST policy generation");
    }
    catch (...)
    {
        RemoveQuietly(temporary);
        throw;
    }
}

} // namespace

std::filesystem::path BestManifest(std::filesystem::path const& checkpoints, std::string const& suite)
{
    return BestDirectory(checkpoints) / (suite + ".json");
}

std::vector<std::filesystem::path> BestRunFiles(std::filesystem::path const& checkpoints,
    std::vector<EvaluationSuite> const& configured, std::filesystem::path const& policyFile)
{
    auto const directory = BestDirectory(checkpoints);
    if (!std::filesystem::is_directory(directory)) return {};
    auto const currentExtension = policyFile.extension();
    std::set<std::pair<std::string, std::filesystem::path>> owners;
    for (auto const& suite : configured) owners.emplace(suite.name, currentExtension);
    std::vector<std::filesystem::path> files, candidates;
    for (auto const& entry : std::filesystem::directory_iterator(directory))
    {
        if (!entry.is_regular_file()) continue;  // Native gate champions use separate subdirectories.
        auto const& file = entry.path();
        auto const stem = file.stem().string();
        if (file.extension() == ".json")
        {
            auto const extension = ManifestPolicyExtension(file);
            if (extension) owners.emplace(stem, *extension);
            if (extension || owners.count({stem, currentExtension}) != 0)
            {
                files.push_back(file);
                continue;
            }
        }
        candidates.push_back(file);
    }
    for (auto const& file : candidates)
    {
        auto const stem = file.stem().string();
        auto const extension = file.extension();
        auto const matches = [&](auto const& owner) {
            return extension == owner.second && IsSuiteGeneration(stem, owner.first);
        };
        if (std::any_of(owners.begin(), owners.end(), matches)) files.push_back(file);
    }
    return files;
}

std::optional<BestRecord> ReadBest(std::filesystem::path const& checkpoints, std::string const& suite)
{
    std::filesystem::path const manifest = BestManifest(checkpoints, suite);
    if (!std::filesystem::exists(manifest))
    {
        return std::nullopt;
    }
    std::string const source = manifest.filename().string();
    Json root;
    BestRecord best;
    try
    {
        root = Json::parse(schema::detail::ReadTextFile(manifest));
        bool const current = root.at("schema") == BestSchemaVersion;
        if ((!current && root.at("schema") != FormerBestSchemaVersion) || root.at("suite") != suite)
        {
            throw std::runtime_error("not the BEST manifest of suite '" + suite + "'");
        }
        best.update = root.at("update").get<std::uint64_t>();
        best.policy = BestDirectory(checkpoints) / root.at("policy").get<std::string>();
        best.report = ReportFromJson(root.at("report").dump(), source);
        if (best.report.suite != suite) throw std::runtime_error("report belongs to another evaluation suite");
        if (root.contains("environment_state"))
        {
            best.environmentState = root.at("environment_state").get<std::string>();
        }
        if (root.contains("reward_weights")) best.rewardWeights = root.at("reward_weights").get<std::vector<double>>();
        if (current)
        {
            best.sha256 = root.at("sha256").get<std::string>();
            if (best.sha256.size() != 64)
                throw std::runtime_error("invalid SHA-256 in BEST manifest");
            if (!best.report.policySha256.empty() && best.report.policySha256 != best.sha256)
                throw std::runtime_error("BEST report and manifest identify different policies");
        }
    }
    catch (std::exception const& error)
    {
        throw std::runtime_error(manifest.string() + ": " + error.what());
    }
    if (!std::filesystem::exists(best.policy))
    {
        throw std::runtime_error(manifest.string() + ": its policy " + best.policy.string() + " is missing");
    }
    // A former manifest has no digest: its policy is read as it is.
    if (!best.sha256.empty() && artifacts::Sha256OfFile(best.policy) != best.sha256)
    {
        throw std::runtime_error(manifest.string() + ": its policy " + best.policy.string()
            + " does not hold the weights that were evaluated (SHA-256 differs)");
    }
    return best;
}

bool OfferBest(std::filesystem::path const& checkpoints, EvaluationSuite const& suite, std::uint64_t update,
    std::filesystem::path const& policyFile, EvaluationReport const& report,
    std::optional<std::string> const& environmentState, std::vector<double> const& rewardWeights)
{
    // Serialize the rank decision as well as the manifest commit. Immutable generation
    // files and the atomic manifest keep lock-free readers stable.
    artifacts::detail::ScopedPathLock const lock(checkpoints, "BEST publication");
    if (report.suite != suite.name)
        throw std::invalid_argument("cannot promote a report from another evaluation suite");
    if (!report.Passed())
    {
        return false;
    }
    std::optional<BestRecord> const incumbent = ReadBest(checkpoints, suite.name);
    if (incumbent && !RanksAbove(report, incumbent->report, suite.best))
    {
        return false;
    }
    // The writable alias may have changed since the actor was loaded for the campaign.
    auto const& digest = report.policySha256;
    if (digest.empty() || artifacts::Sha256OfFile(policyFile) != digest)
        throw std::runtime_error("BEST candidate differs from the policy evaluated: " + policyFile.string());
    std::filesystem::path const directory = BestDirectory(checkpoints);
    std::filesystem::create_directories(directory);
    // The generation is named and checked by the digest of the weights evaluated; the
    // manifest is written last, so a failure leaves the previous BEST as it was.
    std::string const name = GenerationName(suite.name, update, digest, policyFile);
    WriteGeneration(policyFile, digest, directory / name);
    Json manifest = Json::object();
    manifest["schema"] = BestSchemaVersion;
    manifest["suite"] = suite.name;
    manifest["update"] = update;
    manifest["policy"] = name;
    manifest["sha256"] = digest;
    if (environmentState) manifest["environment_state"] = *environmentState;
    if (!rewardWeights.empty()) manifest["reward_weights"] = rewardWeights;
    manifest["report"] = Json::parse(report.ToJson());
    std::string const text = manifest.dump(2) + "\n";
    artifacts::WriteFileAtomically(
        BestManifest(checkpoints, suite.name), [&](std::ostream& out) { out << text; },
        artifacts::AtomicWriteOptions{}, "BEST manifest");
    return true;
}

} // namespace ppokit::eval
