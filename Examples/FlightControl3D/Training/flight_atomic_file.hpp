/**
 * @file flight_atomic_file.hpp
 * @brief Replacement of a file by a fully written temporary, keeping the previous version.
 *
 * Role in the chain: persistence helper of the learner checkpoints (flight_learner.cpp)
 * and of the BEST manifest (flight_model_store.cpp). The caller writes and flushes
 * "<path>.tmp"; this helper then moves the current file to "<path>.previous" and the
 * temporary into place, restoring the previous file if the final rename fails. A reader
 * therefore always finds either the old or the new complete file.
 *
 * Read next: flight_model_store.cpp (BEST generations and checksums).
 */
#pragma once

#include <filesystem>

namespace ppo::examples::flight::detail {

/**
 * @brief Moves a complete temporary file onto its destination, keeping one backup.
 * @param temporary Fully written and flushed temporary file.
 * @param path Destination; an existing file becomes path + ".previous".
 * @throws std::filesystem::filesystem_error when a rename fails (the old file is restored first).
 */
inline void CommitTemporaryFile(std::filesystem::path const& temporary, std::filesystem::path const& path)
{
    auto backup = path;
    backup += ".previous";
    if (std::filesystem::exists(backup))
    {
        std::filesystem::remove(backup);
    }
    bool const hadPrevious = std::filesystem::exists(path);
    if (hadPrevious)
    {
        std::filesystem::rename(path, backup);
    }
    try
    {
        std::filesystem::rename(temporary, path);
    }
    catch (...)
    {
        if (hadPrevious)
        {
            std::filesystem::rename(backup, path);
        }
        throw;
    }
}

} // namespace ppo::examples::flight::detail
