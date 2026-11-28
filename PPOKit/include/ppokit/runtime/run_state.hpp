/**
 * @file PPOKit/include/ppokit/runtime/run_state.hpp
 * @brief What a training run keeps beside its checkpoint besides the networks: the
 *        environment's training state (a curriculum, its windows, its random stream), and the
 *        archive of a run that a fresh run replaces.
 * @details checkpoints/environment_state/<update>.json holds the environment's state at an
 * update the run saved. ppokit-run writes the state of an update, then the checkpoint of that
 * update, then removes the other states: whatever interrupts it, the checkpoint on disk finds
 * the state of its own update.
 *
 * @code{.json}
 * { "schema": "ppokit.environment_state/1", "update": 120, "state": "<the environment's text>" }
 * @endcode
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::runtime {

/** @brief The environment's training state at one update of the run. */
struct SavedTrainingState final
{
    std::uint64_t update = 0;
    std::string state;
};

/** @brief checkpoints/environment_state: one file per saved update. */
std::filesystem::path TrainingStateFolder(std::filesystem::path const& checkpoints);

/**
 * @brief Writes the state of `update` atomically.
 * @throws std::runtime_error when the file cannot be written.
 */
void SaveTrainingState(std::filesystem::path const& checkpoints, std::uint64_t update, std::string const& state);

/** @brief Removes the saved states of every update but `update` (once its checkpoint is saved). */
void KeepOnlyTrainingState(std::filesystem::path const& checkpoints, std::uint64_t update);

/**
 * @brief Every saved state, by increasing update.
 * @throws std::runtime_error for a damaged file.
 */
std::vector<SavedTrainingState> ReadTrainingStates(std::filesystem::path const& checkpoints);

/** @brief The state saved at `update`, if there is one. */
std::optional<std::string> TrainingStateOfUpdate(std::vector<SavedTrainingState> const& saved, std::uint64_t update);

/**
 * @brief Moves the files of a finished run that exist out of the way, into
 *        checkpoints/archive/run-<UTC date and time>[-n]/, keeping their paths relative to
 *        `checkpoints` (the file name for a file elsewhere), so that a fresh run starts
 *        without them and nothing is lost.
 * @param files Files or folders (absent ones are skipped).
 * @return The archive folder, or nothing when none of them existed.
 * @throws std::runtime_error when one cannot be moved; completed moves are rolled back in reverse order.
 */
std::optional<std::filesystem::path> ArchiveRun(std::filesystem::path const& checkpoints,
    std::vector<std::filesystem::path> const& files);

} // namespace ppokit::runtime
