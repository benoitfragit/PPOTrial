/**
 * @file PPOKit/src/runtime/run_state.cpp
 * @brief The environment's training state beside the run's checkpoint, and the archive of a
 *        replaced run.
 */
#include "ppokit/runtime/run_state.hpp"

#include "ppokit/artifacts/atomic_file.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <set>
#include <stdexcept>
#include <system_error>

namespace ppokit::runtime {
namespace {

using Json = nlohmann::json;

constexpr char const* StateSchema = "ppokit.environment_state/1";
constexpr char const* StateExtension = ".json";

std::filesystem::path StateFile(std::filesystem::path const& checkpoints, std::uint64_t update)
{
    return TrainingStateFolder(checkpoints) / (std::to_string(update) + StateExtension);
}

/** @brief One saved state; its file name must be its update. */
SavedTrainingState ReadStateFile(std::filesystem::path const& file)
{
    try
    {
        std::ifstream stream(file, std::ios::binary);
        Json const root = Json::parse(stream);
        if (root.at("schema") != StateSchema)
        {
            throw std::runtime_error("not an environment state (schema " + root.at("schema").dump() + ")");
        }
        SavedTrainingState saved{root.at("update").get<std::uint64_t>(), root.at("state").get<std::string>()};
        if (file.stem().string() != std::to_string(saved.update))
        {
            throw std::runtime_error("holds the state of update " + std::to_string(saved.update));
        }
        return saved;
    }
    catch (std::exception const& error)
    {
        throw std::runtime_error(file.string() + ": damaged environment state: " + error.what());
    }
}

/** @brief "run-YYYYmmdd-HHMMSS" in UTC, the name of an archive folder. */
std::string ArchiveStamp()
{
    std::time_t const now = std::time(nullptr);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    char text[32] = {};
    std::strftime(text, sizeof(text), "run-%Y%m%d-%H%M%S", &utc);
    return text;
}

/** @brief A folder of checkpoints/archive that does not exist yet. */
std::filesystem::path NewArchiveFolder(std::filesystem::path const& checkpoints)
{
    std::filesystem::path const archive = checkpoints / "archive";
    std::string const stamp = ArchiveStamp();
    std::filesystem::path folder = archive / stamp;
    for (int index = 2; std::filesystem::exists(folder); ++index)
    {
        folder = archive / (stamp + "-" + std::to_string(index));
    }
    return folder;
}

/** @brief Where `file` goes in the archive: its path under `checkpoints`, else its name. */
std::filesystem::path ArchivedPath(std::filesystem::path const& checkpoints, std::filesystem::path const& file)
{
    std::filesystem::path const relative = file.lexically_normal().lexically_relative(checkpoints.lexically_normal());
    if (relative.empty() || *relative.begin() == "..")
    {
        return file.filename();
    }
    return relative;
}

} // namespace

std::filesystem::path TrainingStateFolder(std::filesystem::path const& checkpoints)
{
    return checkpoints / "environment_state";
}

void SaveTrainingState(std::filesystem::path const& checkpoints, std::uint64_t update, std::string const& state)
{
    Json root;
    root["schema"] = StateSchema;
    root["update"] = update;
    root["state"] = state;
    std::string const text = root.dump(2) + "\n";
    artifacts::WriteFileAtomically(
        StateFile(checkpoints, update), [&text](std::ostream& out) { out << text; }, artifacts::AtomicWriteOptions{},
        "environment state");
}

void KeepOnlyTrainingState(std::filesystem::path const& checkpoints, std::uint64_t update)
{
    std::filesystem::path const folder = TrainingStateFolder(checkpoints);
    if (!std::filesystem::exists(folder))
    {
        return;
    }
    std::filesystem::path const kept = StateFile(checkpoints, update);
    for (std::filesystem::directory_entry const& entry : std::filesystem::directory_iterator(folder))
    {
        if (entry.path() != kept && entry.path().extension() == StateExtension)
        {
            std::filesystem::remove(entry.path());
        }
    }
}

std::vector<SavedTrainingState> ReadTrainingStates(std::filesystem::path const& checkpoints)
{
    std::vector<SavedTrainingState> saved;
    std::filesystem::path const folder = TrainingStateFolder(checkpoints);
    if (!std::filesystem::exists(folder))
    {
        return saved;
    }
    for (std::filesystem::directory_entry const& entry : std::filesystem::directory_iterator(folder))
    {
        if (entry.path().extension() == StateExtension)
        {
            saved.push_back(ReadStateFile(entry.path()));
        }
    }
    std::sort(saved.begin(), saved.end(),
        [](SavedTrainingState const& left, SavedTrainingState const& right) { return left.update < right.update; });
    return saved;
}

std::optional<std::string> TrainingStateOfUpdate(std::vector<SavedTrainingState> const& saved, std::uint64_t update)
{
    for (SavedTrainingState const& generation : saved)
    {
        if (generation.update == update)
        {
            return generation.state;
        }
    }
    return std::nullopt;
}

std::optional<std::filesystem::path> ArchiveRun(std::filesystem::path const& checkpoints,
    std::vector<std::filesystem::path> const& files)
{
    namespace fs = std::filesystem;
    std::vector<std::pair<fs::path, fs::path>> moves;
    std::set<fs::path> targets;
    fs::path const folder = NewArchiveFolder(checkpoints);
    for (auto const& file : files)
    {
        if (!fs::exists(file)) continue;
        fs::path const target = folder / ArchivedPath(checkpoints, file);
        if (!targets.insert(target).second)
        {
            throw std::runtime_error("two run files would occupy the same archive path: " + target.string());
        }
        moves.emplace_back(file, target);
    }
    if (moves.empty()) return std::nullopt;
    std::size_t moved = 0;
    try
    {
        for (auto const& move : moves)
        {
            fs::create_directories(move.second.parent_path());
            fs::rename(move.first, move.second);
            ++moved;
        }
    }
    catch (...)
    {
        while (moved > 0)
        {
            auto const& move = moves[--moved];
            std::error_code error;
            fs::rename(move.second, move.first, error);
            if (error)
            {
                throw std::runtime_error("archive rollback failed: preserved " + move.second.string()
                    + " for recovery to " + move.first.string() + ": " + error.message());
            }
        }
        std::error_code ignored;
        fs::remove_all(folder, ignored);
        throw;
    }
    return folder;
}

} // namespace ppokit::runtime
