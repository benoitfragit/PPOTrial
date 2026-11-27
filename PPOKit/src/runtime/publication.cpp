/**
 * @file PPOKit/src/runtime/publication.cpp
 * @brief A ready directory is the commit decision. Its contents survive partial installation.
 */
#include "ppokit/runtime/publication.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/artifacts/sha256.hpp"
#include "ppokit/runtime/run_state.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <system_error>
#include <chrono>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace ppokit::runtime {
namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;
constexpr char Schema[] = "ppokit.generation/2";
constexpr char PreviousSchema[] = "ppokit.generation/1";
constexpr char Manifest[] = "generation.json";

/** @brief Atomically renames a prepared directory, tolerating short Windows scanner handles. */
void RenameDirectory(fs::path const& source, fs::path const& target)
{
#if defined(_WIN32)
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    for (;;)
    {
        if (MoveFileExW(source.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH)) return;
        DWORD const error = GetLastError();
        if ((error != ERROR_ACCESS_DENIED && error != ERROR_SHARING_VIOLATION && error != ERROR_LOCK_VIOLATION)
            || std::chrono::steady_clock::now() >= deadline)
        {
            throw std::system_error(static_cast<int>(error), std::system_category(),
                "cannot commit publication directory " + source.u8string());
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
#else
    fs::rename(source, target);
#endif
}

/** @brief A generation carries the coefficients effective at its next collection boundary. */
void CheckWeights(std::vector<double> const& weights)
{
    if (weights.empty() || !std::all_of(weights.begin(), weights.end(), [](double weight) { return std::isfinite(weight); }))
        throw std::runtime_error("invalid generation reward weights");
}

Json Read(fs::path const& path)
{
    std::ifstream input(path, std::ios::binary);
    auto result = Json::parse(input);
    if (result.at("schema") == Schema)
        CheckWeights(result.at("reward_weights").get<std::vector<double>>());
    else if (result.at("schema") != PreviousSchema)
        throw std::runtime_error("invalid publication manifest at " + path.string());
    return result;
}

void Write(fs::path const& path, Json const& manifest)
{
    artifacts::WriteFileAtomically(path, [&](std::ostream& out) { out << manifest.dump() << '\n'; },
        artifacts::AtomicWriteOptions{}, "generation manifest");
}

void CheckFile(fs::path const& path, std::string const& digest)
{
    if (artifacts::Sha256OfFile(path) != digest)
        throw std::runtime_error("generation digest mismatch for " + path.string());
}

/** @brief Copy, verify and atomically install one immutable member of a prepared generation. */
void Install(fs::path const& source, fs::path const& target, std::string const& digest)
{
    if (!target.parent_path().empty()) fs::create_directories(target.parent_path());
    auto const next = artifacts::detail::UniqueTemporaryPath(target, ".installing");
    try
    {
        fs::copy_file(source, next);
        CheckFile(next, digest);
        artifacts::CommitTemporaryFile(next, target, artifacts::AtomicWriteOptions{}, "generation member");
    }
    catch (...)
    {
        std::error_code ignored;
        fs::remove(next, ignored);
        throw;
    }
}

Json Destinations(schema::Project const& project)
{
    return {{"training", project.LatestCheckpoint().lexically_relative(project.root).generic_string()},
        {"policy", project.TrainingPolicy().lexically_relative(project.root).generic_string()}};
}

/** @brief Whether two reads identify the same complete generation and context. */
bool SameGeneration(PublishedGeneration const& left, PublishedGeneration const& right)
{
    return left.update == right.update && left.trainingDigest == right.trainingDigest
        && left.policyDigest == right.policyDigest && left.state == right.state
        && left.rewardWeights == right.rewardWeights;
}

/** @brief Replays one prepared transaction. The caller serializes publication for this project. */
void RecoverPublicationUnlocked(schema::Project const& project)
{
    auto const root = project.Checkpoints();
    auto const pending = root / ".publication-ready";
    if (!fs::is_directory(pending)) return;
    auto const record = Read(pending / Manifest);
    if (record.at("destinations") != Destinations(project))
        throw std::runtime_error("project outputs differ from its prepared generation");
    std::string const trainingDigest = record.at("training_sha256").get<std::string>();
    std::string const policyDigest = record.at("policy_sha256").get<std::string>();
    CheckFile(pending / "training", trainingDigest);
    CheckFile(pending / "policy", policyDigest);
    auto const update = record.at("update").get<std::uint64_t>();
    SaveTrainingState(root, update, record.at("state").get<std::string>());
    Install(pending / "training", project.LatestCheckpoint(), trainingDigest);
    Install(pending / "policy", project.TrainingPolicy(), policyDigest);
    Install(pending / Manifest, root / Manifest, artifacts::Sha256OfFile(pending / Manifest));
    KeepOnlyTrainingState(root, update);
    // Retire by rename before deleting. An interruption during removal leaves no incomplete ready directory.
    auto const retired = root / ".publication-retired";
    fs::remove_all(retired);
    RenameDirectory(pending, retired);
    std::error_code ignored;
    fs::remove_all(retired, ignored);
}

} // namespace

void RecoverPublication(schema::Project const& project)
{
    artifacts::detail::ScopedPathLock const lock(project.root, "generation publication");
    RecoverPublicationUnlocked(project);
}

void PublishGeneration(schema::Project const& project, std::uint64_t update, std::string const& state,
    std::vector<double> const& weights, std::function<void(fs::path const&, fs::path const&)> const& save)
{
    CheckWeights(weights);
    artifacts::detail::ScopedPathLock const lock(project.root, "generation publication");
    RecoverPublicationUnlocked(project);
    auto const root = project.Checkpoints();
    auto const stage = root / ".publication-stage";
    fs::remove_all(stage);
    fs::create_directories(stage);
    save(stage / "training", stage / "policy");
    Write(stage / Manifest, {{"schema", Schema}, {"update", update}, {"state", state}, {"reward_weights", weights},
        {"destinations", Destinations(project)}, {"training_sha256", artifacts::Sha256OfFile(stage / "training")},
        {"policy_sha256", artifacts::Sha256OfFile(stage / "policy")}});
    RenameDirectory(stage, root / ".publication-ready");
    RecoverPublicationUnlocked(project);
}

std::optional<PublishedGeneration> ReadPublicationManifest(schema::Project const& project)
{
    auto const root = project.Checkpoints();
    if (fs::exists(root / ".publication-ready")) throw std::runtime_error("training generation publication in progress");
    if (!fs::exists(root / Manifest)) return std::nullopt;
    auto const record = Read(root / Manifest);
    if (fs::exists(root / ".publication-ready")) throw std::runtime_error("training generation publication in progress");
    if (record.at("destinations") != Destinations(project)) throw std::runtime_error("generation outputs have changed");
    std::optional<std::vector<double>> weights;
    if (record.at("schema") == Schema) weights = record.at("reward_weights").get<std::vector<double>>();
    return PublishedGeneration{record.at("update").get<std::uint64_t>(), record.at("training_sha256").get<std::string>(),
        record.at("policy_sha256").get<std::string>(), record.at("state").get<std::string>(), std::move(weights)};
}

std::optional<PublishedGeneration> VerifyPublication(schema::Project const& project)
{
    auto generation = ReadPublicationManifest(project);
    if (!generation) return std::nullopt;
    CheckFile(project.LatestCheckpoint(), generation->trainingDigest);
    CheckFile(project.TrainingPolicy(), generation->policyDigest);
    auto const state = TrainingStateOfUpdate(ReadTrainingStates(project.Checkpoints()), generation->update);
    if (!state) throw std::runtime_error("no environment state of update " + std::to_string(generation->update));
    if (*state != generation->state) throw std::runtime_error("generation environment state mismatch");
    auto const confirmed = ReadPublicationManifest(project);
    if (!confirmed || !SameGeneration(*generation, *confirmed))
        throw std::runtime_error("training generation changed while it was being verified");
    return generation;
}

} // namespace ppokit::runtime
