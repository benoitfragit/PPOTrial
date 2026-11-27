/**
 * @file PPOKit/src/artifacts/atomic_file.cpp
 * @brief Commit step of the transactional file replacement.
 */
#include "ppokit/artifacts/atomic_file.hpp"

#include <system_error>

namespace ppokit::artifacts {
namespace {

/** @brief Builds the message of a failed commit, keeping the filesystem cause. */
std::string CommitFailure(char const* context, std::filesystem::path const& path, char const* cause)
{
    return std::string(context) + ": cannot replace '" + path.string() + "' (" + cause + ")";
}

/** @brief Best-effort deletion used on cleanup paths, where a second failure must not hide the first. */
void RemoveQuietly(std::filesystem::path const& path) noexcept
{
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

} // namespace

std::filesystem::path SuffixedPath(std::filesystem::path const& path, char const* suffix)
{
    auto suffixed = path;
    suffixed += suffix;
    return suffixed;
}

void CommitTemporaryFile(std::filesystem::path const& temporary, std::filesystem::path const& path,
    AtomicWriteOptions const& options, char const* context)
{
    auto const backup = SuffixedPath(path, options.backupSuffix);
    RemoveQuietly(backup);
    bool const hadPrevious = std::filesystem::exists(path);
    if (hadPrevious)
    {
        try
        {
            std::filesystem::rename(path, backup);
        }
        catch (std::filesystem::filesystem_error const& error)
        {
            RemoveQuietly(temporary);
            throw ArtifactWriteError(CommitFailure(context, path, error.what()));
        }
    }
    try
    {
        std::filesystem::rename(temporary, path);
    }
    catch (std::filesystem::filesystem_error const& error)
    {
        if (hadPrevious)
        {
            // Restore the previous generation so that readers keep a valid artifact.
            std::error_code restoreError;
            std::filesystem::rename(backup, path, restoreError);
        }
        RemoveQuietly(temporary);
        throw ArtifactWriteError(CommitFailure(context, path, error.what()));
    }
    if (hadPrevious && options.backupPolicy == BackupPolicy::RemoveAfterCommit)
    {
        RemoveQuietly(backup);
    }
}

} // namespace ppokit::artifacts
