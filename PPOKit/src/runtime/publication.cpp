/**
 * @file PPOKit/src/runtime/publication.cpp
 * @brief A ready directory is the commit decision. Its contents survive partial installation.
 */
#include "ppokit/runtime/publication.hpp"
#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/artifacts/sha256.hpp"
#include "ppokit/runtime/run_state.hpp"

#include <fstream>
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace ppokit::runtime {
namespace {

namespace fs = std::filesystem;
using Json = nlohmann::json;
constexpr char Schema[] = "ppokit.generation/2";
constexpr char PreviousSchema[] = "ppokit.generation/1";
constexpr char Manifest[] = "generation.json";

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

// Copy beside each destination, including destinations on a different volume from checkpoints/.
void Install(fs::path const& source, fs::path const& target)
{
    if (!target.parent_path().empty()) fs::create_directories(target.parent_path());
    auto const next = artifacts::SuffixedPath(target, ".installing");
    fs::copy_file(source, next, fs::copy_options::overwrite_existing);
    artifacts::CommitTemporaryFile(next, target, artifacts::AtomicWriteOptions{}, "generation member");
}

Json Destinations(schema::Project const& project)
{
    return {{"training", project.LatestCheckpoint().lexically_relative(project.root).generic_string()},
        {"policy", project.TrainingPolicy().lexically_relative(project.root).generic_string()}};
}

} // namespace

void RecoverPublication(schema::Project const& project)
{
    auto const root = project.Checkpoints();
    auto const pending = root / ".publication-ready";
    if (!fs::is_directory(pending)) return;
    auto const record = Read(pending / Manifest);
    if (record.at("destinations") != Destinations(project))
        throw std::runtime_error("project outputs differ from its prepared generation");
    CheckFile(pending / "training", record.at("training_sha256").get<std::string>());
    CheckFile(pending / "policy", record.at("policy_sha256").get<std::string>());
    auto const update = record.at("update").get<std::uint64_t>();
    SaveTrainingState(root, update, record.at("state").get<std::string>());
    Install(pending / "training", project.LatestCheckpoint());
    Install(pending / "policy", project.TrainingPolicy());
    Install(pending / Manifest, root / Manifest);
    KeepOnlyTrainingState(root, update);
    // Retire by rename before deleting. An interruption during removal leaves no incomplete ready directory.
    auto const retired = root / ".publication-retired";
    fs::remove_all(retired);
    fs::rename(pending, retired);
    std::error_code ignored;
    fs::remove_all(retired, ignored);
}

void PublishGeneration(schema::Project const& project, std::uint64_t update, std::string const& state,
    std::vector<double> const& weights, std::function<void(fs::path const&, fs::path const&)> const& save)
{
    CheckWeights(weights);
    RecoverPublication(project);
    auto const root = project.Checkpoints();
    auto const stage = root / ".publication-stage";
    fs::remove_all(stage);
    fs::create_directories(stage);
    save(stage / "training", stage / "policy");
    Write(stage / Manifest, {{"schema", Schema}, {"update", update}, {"state", state}, {"reward_weights", weights},
        {"destinations", Destinations(project)}, {"training_sha256", artifacts::Sha256OfFile(stage / "training")},
        {"policy_sha256", artifacts::Sha256OfFile(stage / "policy")}});
    fs::rename(stage, root / ".publication-ready");
    RecoverPublication(project);
}

std::optional<PublishedGeneration> ReadPublicationManifest(schema::Project const& project)
{
    auto const root = project.Checkpoints();
    if (fs::exists(root / ".publication-ready")) throw std::runtime_error("training generation publication in progress");
    if (!fs::exists(root / Manifest)) return std::nullopt;
    auto const record = Read(root / Manifest);
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
    return generation;
}

} // namespace ppokit::runtime
