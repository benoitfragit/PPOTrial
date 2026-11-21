/**
 * @file flight_model_store.cpp
 * @brief BEST manifest reading, checksum verification, ranking and atomic promotion.
 *
 * Promotion sequence: save the candidate as best/candidate.pending, checksum it (FNV-1a
 * 64), rename it to an immutable best/flight_<update>_<checksum>.txt (or verify an
 * identical existing generation), then atomically replace best/current.json. A crash at
 * any point leaves the previous manifest and generation readable.
 *
 * Read next: flight_model_store.hpp, flight_atomic_file.hpp.
 */
#include "flight_model_store.hpp"
#include "flight_atomic_file.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ppo::examples::flight {
namespace {
using Json = nlohmann::json;
// Version 1 used one continuous RNG stream; version 2 seeds each episode
// independently; version 3 stratifies rolling and airborne starts up to the
// curriculum difficulty recorded in the manifest.
constexpr unsigned EvaluationProtocolVersion = 3;
/** @brief Version of the current.json manifest layout. */
constexpr int ManifestFormatVersion = 1;
/** @brief Difficulty of manifests written before the curriculum: the full airborne task. */
constexpr double LegacyManifestDifficulty = 1.0;
/** @brief Protocol version assumed for manifests without the field. */
constexpr unsigned LegacyEvaluationProtocolVersion = 1U;
/** @brief FNV-1a 64-bit offset basis and prime. */
constexpr std::uint64_t FnvOffsetBasis = 14695981039346656037ULL;
constexpr std::uint64_t FnvPrime = 1099511628211ULL;
/** @brief Hexadecimal digits of a 64-bit checksum. */
constexpr int ChecksumHexDigits = 16;
/** @brief Indentation of the JSON manifest. */
constexpr int ManifestIndent = 2;

/** @brief FNV-1a 64 of a file, as 16 lowercase hexadecimal digits. */
std::string Checksum(std::filesystem::path const& path)
{
    // FNV-1a is an integrity checksum, not a security/authentication mechanism.
    std::uint64_t hash = FnvOffsetBasis;
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("FlightControl3D: missing BEST checkpoint");
    }
    char byte{};
    while (stream.get(byte))
    {
        hash ^= static_cast<unsigned char>(byte);
        hash *= FnvPrime;
    }
    if (!stream.eof())
    {
        throw std::runtime_error("FlightControl3D: BEST checksum read failure");
    }
    std::ostringstream text;
    text << std::hex << std::setw(ChecksumHexDigits) << std::setfill('0') << hash;
    return text.str();
}

/** @brief Parsed manifest: metrics and the checkpoint it points to. */
struct BestRecord final
{
    ValidationMetrics metrics{};
    std::filesystem::path checkpoint;
};

/** @brief Reads the campaign counts of a manifest. */
ValidationMetrics ReadManifestMetrics(Json const& json)
{
    ValidationMetrics metrics{};
    metrics.update = json.at("update").get<std::uint64_t>();
    // Earlier protocols always evaluated the full airborne task.
    metrics.difficulty = json.value("difficulty", LegacyManifestDifficulty);
    metrics.episodes = json.at("episodes").get<std::size_t>();
    metrics.touchdowns = json.at("touchdowns").get<std::size_t>();
    metrics.stops = json.at("stops").get<std::size_t>();
    metrics.crashes = json.at("crashes").get<std::size_t>();
    metrics.timeouts = json.at("timeouts").get<std::size_t>();
    metrics.touchdownSumMeters = json.at("touchdownSumMeters").get<double>();
    metrics.stopSumMeters = json.at("stopSumMeters").get<double>();
    return metrics;
}

/**
 * @brief Reads and verifies best/current.json.
 * @param config Current contract and evaluation settings.
 * @param directory Model directory.
 * @param allowTrainingChanges Accept a warm-startable contract change (the old score is then not reused).
 * @return The record for the exact current contract and protocol, or nothing.
 * @throws std::runtime_error for an incompatible contract, an unvalidated campaign or a checksum mismatch.
 */
std::optional<BestRecord> ReadBest(Config const& config, std::filesystem::path const& directory,
    bool allowTrainingChanges = false)
{
    auto const pointer = directory / "best/current.json";
    if (!std::filesystem::exists(pointer))
    {
        return std::nullopt;
    }
    std::ifstream stream(pointer);
    Json json;
    stream >> json;
    auto const signature = json.at("signature").get<std::string>();
    bool const changedContract = signature != config.checkpointSignature;
    bool const acceptedContract = !changedContract
        || (allowTrainingChanges && CompatibleTrainingSignature(signature, config.checkpointSignature));
    if (json.at("formatVersion") != ManifestFormatVersion || !acceptedContract
        || json.at("seed").get<std::uint32_t>() != config.evaluation.seed
        || json.at("requiredEpisodes").get<std::size_t>() != config.evaluation.episodes)
    {
        throw std::runtime_error("FlightControl3D: incompatible BEST validation contract");
    }
    BestRecord record{};
    record.metrics = ReadManifestMetrics(json);
    auto const filename = std::filesystem::path(json.at("checkpoint").get<std::string>());
    // The checkpoint must be a plain file name inside best/, never a path.
    if (filename.empty() || filename.is_absolute() || filename != filename.filename()
        || !Validated(config, record.metrics))
    {
        throw std::runtime_error("FlightControl3D: unvalidated BEST manifest");
    }
    record.checkpoint = directory / "best" / filename;
    if (Checksum(record.checkpoint) != json.at("fnv1a64").get<std::string>())
    {
        throw std::runtime_error("FlightControl3D: BEST checkpoint/manifest mismatch");
    }
    // Preserve the old generation; its score requires the exact current contract.
    if (changedContract || json.value("evaluationProtocolVersion", LegacyEvaluationProtocolVersion)
        != EvaluationProtocolVersion)
    {
        return std::nullopt;
    }
    return record;
}

/**
 * @brief Removes a pending candidate and its backup left by an interrupted promotion.
 *
 * The pending file is scratch space, never a generation: clearing it first means
 * FlightLearner::Save() has no previous file to back up as "<pending>.previous".
 * @param pending Path of the pending candidate checkpoint.
 */
void RemovePendingCandidate(std::filesystem::path const& pending)
{
    auto backup = pending;
    backup += ".previous";
    std::filesystem::remove(pending);
    std::filesystem::remove(backup);
}

/** @brief Writes the manifest to a temporary file, then atomically replaces the pointer. */
void CommitPointer(std::filesystem::path const& path, Json const& json)
{
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream stream(temporary, std::ios::trunc);
        stream << json.dump(ManifestIndent) << '\n';
        stream.flush();
        if (!stream)
        {
            throw std::runtime_error("FlightControl3D: BEST manifest write failed");
        }
    }
    detail::CommitTemporaryFile(temporary, path);
}

/** @brief Manifest of a promoted generation; the key set is a compatibility contract. */
Json Manifest(Config const& config, ValidationMetrics const& metrics, std::string const& filename,
    std::string const& checksum)
{
    return Json{{"formatVersion", ManifestFormatVersion}, {"signature", config.checkpointSignature},
        {"evaluationProtocolVersion", EvaluationProtocolVersion},
        {"seed", config.evaluation.seed},
        {"requiredEpisodes", config.evaluation.episodes}, {"update", metrics.update},
        {"difficulty", metrics.difficulty}, {"episodes", metrics.episodes},
        {"touchdowns", metrics.touchdowns}, {"stops", metrics.stops}, {"crashes", metrics.crashes},
        {"timeouts", metrics.timeouts},
        {"touchdownSumMeters", metrics.touchdownSumMeters}, {"stopSumMeters", metrics.stopSumMeters},
        {"score", metrics.Score()}, {"checkpoint", filename}, {"fnv1a64", checksum}};
}
} // namespace

void ValidationMetrics::Record(MissionState const& mission)
{
    if (mission.outcome == Outcome::Flying)
    {
        throw std::logic_error("FlightControl3D: incomplete validation episode");
    }
    ++episodes;
    if (mission.touchedDown)
    {
        ++touchdowns;
        touchdownSumMeters += mission.touchdownFromThreshold;
    }
    if (mission.outcome == Outcome::Stopped)
    {
        ++stops;
        stopSumMeters += mission.stopFromThreshold;
    }
    if (mission.outcome == Outcome::Crash)
    {
        ++crashes;
    }
    if (mission.outcome == Outcome::Timeout)
    {
        ++timeouts;
    }
}

bool Validated(Config const& config, ValidationMetrics const& metrics) noexcept
{
    double const runwayLength = config.runway.lengthMeters;
    // Complete campaign of the configured size and finite difficulty.
    bool const complete = metrics.episodes == config.evaluation.episodes && metrics.episodes > 0
        && std::isfinite(metrics.difficulty) && metrics.difficulty >= 0 && metrics.difficulty <= 1;
    // Counts consistent with one outcome per episode.
    bool const consistent = metrics.touchdowns <= metrics.episodes && metrics.stops <= metrics.touchdowns
        && metrics.stops <= metrics.episodes && metrics.crashes <= metrics.episodes
        && metrics.timeouts <= metrics.episodes
        && metrics.stops + metrics.crashes + metrics.timeouts == metrics.episodes;
    // Distances finite and on the runway.
    bool const plausible = std::isfinite(metrics.stopSumMeters) && std::isfinite(metrics.touchdownSumMeters)
        && metrics.stopSumMeters >= 0 && metrics.touchdownSumMeters >= 0
        && metrics.stopSumMeters <= static_cast<double>(metrics.stops) * runwayLength
        && metrics.touchdownSumMeters <= static_cast<double>(metrics.touchdowns) * runwayLength;
    bool const successful = metrics.SuccessRate() >= config.evaluation.minimumSuccessRate;
    return complete && consistent && successful && plausible;
}

bool Better(ValidationMetrics const& candidate, ValidationMetrics const& incumbent) noexcept
{
    // Campaigns follow the curriculum: a policy validated on a harder mix
    // supersedes one validated on an easier mix, whatever their stop rates.
    if (candidate.difficulty != incumbent.difficulty)
    {
        return candidate.difficulty > incumbent.difficulty;
    }
    if (candidate.SuccessRate() != incumbent.SuccessRate())
    {
        return candidate.SuccessRate() > incumbent.SuccessRate();
    }
    return candidate.MeanStop() < incumbent.MeanStop();
}

FlightModelStore::FlightModelStore(Config const& config, std::filesystem::path directory)
    : config_(config), directory_(std::move(directory))
{
}

void FlightModelStore::SetRewardContract(RewardSettings const& settings, std::string const& signature)
{
    config_.reward = settings;
    config_.checkpointSignature = signature;
}

std::filesystem::path FlightModelStore::Latest() const
{
    return directory_ / "latest/flight_policy.txt";
}

std::filesystem::path FlightModelStore::CurriculumLevel() const
{
    return directory_ / "latest/curriculum_level.txt";
}

std::optional<ValidationMetrics> FlightModelStore::BestMetrics(bool allowTrainingChanges) const
{
    auto const record = ReadBest(config_, directory_, allowTrainingChanges);
    if (!record)
    {
        return std::nullopt;
    }
    return record->metrics;
}

void FlightModelStore::LoadBest(FlightLearner& learner) const
{
    auto const record = ReadBest(config_, directory_);
    if (!record)
    {
        throw std::runtime_error("FlightControl3D: no validated BEST; train and complete validation first");
    }
    learner.Load(record->checkpoint);
    if (learner.Stats().updates != record->metrics.update)
    {
        throw std::runtime_error("FlightControl3D: BEST update/validation mismatch");
    }
}

bool FlightModelStore::Promote(FlightLearner const& learner, ValidationMetrics const& metrics)
{
    if (!Validated(config_, metrics))
    {
        return false;
    }
    if (metrics.update != learner.Stats().updates)
    {
        throw std::logic_error("FlightControl3D: validation must describe the frozen candidate");
    }
    auto const best = ReadBest(config_, directory_, true);
    if (best && !Better(metrics, best->metrics))
    {
        return false;
    }
    auto const bestDirectory = directory_ / "best";
    std::filesystem::create_directories(bestDirectory);
    auto const pending = bestDirectory / "candidate.pending";
    RemovePendingCandidate(pending);
    learner.Save(pending);
    auto const checksum = Checksum(pending);
    auto const filename = "flight_" + std::to_string(metrics.update) + "_" + checksum + ".txt";
    auto const destination = bestDirectory / filename;
    // Generations are immutable: an existing file must be byte-identical.
    if (std::filesystem::exists(destination))
    {
        bool const identical = Checksum(destination) == checksum;
        // The candidate is discarded whether or not the existing generation matches.
        std::filesystem::remove(pending);
        if (!identical)
        {
            throw std::runtime_error("FlightControl3D: existing BEST generation mismatch");
        }
    }
    else
    {
        std::filesystem::rename(pending, destination);
    }
    CommitPointer(bestDirectory / "current.json", Manifest(config_, metrics, filename, checksum));
    return true;
}
} // namespace ppo::examples::flight
