/**
 * @file PPOKit/include/ppokit/artifacts/atomic_file.hpp
 * @brief Transactional replacement of a file: write a temporary, then commit it.
 * @details A reader never observes a missing or partially written artifact.
 * The replacement uses the native atomic-replace primitive of the host OS.
 * Callers choose whether the previous generation is copied to a backup path,
 * because some workflows archive it (FlightControl3D CI uploads
 * `*.previous`). Writers targeting the same file are serialized across
 * processes and always use distinct temporary paths.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ppokit::artifacts {

namespace detail {

/**
 * @brief Bounded cross-process exclusion associated with one normalized path.
 * @details This internal primitive lets multi-file transactions serialize
 * their preparation while each individual member still uses atomic replace.
 */
class ScopedPathLock final
{
public:
    /** @brief Acquires the path lock or throws after the bounded wait. */
    ScopedPathLock(std::filesystem::path const& path, char const* context);
    ~ScopedPathLock();
    ScopedPathLock(ScopedPathLock const&) = delete;
    ScopedPathLock& operator=(ScopedPathLock const&) = delete;

private:
    struct Handle;
    std::unique_ptr<Handle> handle_;
};

/**
 * @brief Returns a process-unique temporary path beside `path`.
 * @param path Final artifact path.
 * @param suffix Human-readable temporary suffix.
 * @return A path on the same filesystem as the final artifact.
 */
std::filesystem::path UniqueTemporaryPath(std::filesystem::path const& path, char const* suffix);

/**
 * @brief Flushes a closed file through the operating-system cache.
 * @param path File whose bytes must reach stable storage.
 * @param context Short owner name used in error messages.
 * @throws ArtifactWriteError when the file cannot be opened or synchronized.
 */
void SynchronizeFile(std::filesystem::path const& path, char const* context);

} // namespace detail

/** @brief What happens to the previous file once the new one is committed. */
enum class BackupPolicy : std::uint8_t
{
    /** No previous generation is kept; a stale backup is removed before commit. */
    RemoveAfterCommit,
    /** The backup stays on disk as the previous generation of the artifact. */
    KeepPrevious,
};

/** @brief Options of WriteFileAtomically; defaults suit most checkpoints. */
struct AtomicWriteOptions final
{
    /** Suffix appended to the final path for the temporary file. */
    char const* temporarySuffix = ".tmp";
    /** Suffix appended to the final path for the previous generation. */
    char const* backupSuffix = ".bak";
    /** Whether the previous generation survives a successful commit. */
    BackupPolicy backupPolicy = BackupPolicy::RemoveAfterCommit;
    /** Create missing parent directories before writing. */
    bool createParentDirectories = true;
    /** Open mode of the temporary stream; `trunc` is always added. */
    std::ios::openmode mode = std::ios::out;

    /** @brief Copy an existing payload byte for byte, including its original line endings. */
    static AtomicWriteOptions Binary() noexcept
    {
        AtomicWriteOptions options;
        options.mode = std::ios::binary;
        return options;
    }

    /** @brief Options that keep the previous generation under `path + backupSuffix`. */
    static AtomicWriteOptions KeepingPrevious(char const* previousSuffix) noexcept
    {
        AtomicWriteOptions options;
        options.backupSuffix = previousSuffix;
        options.backupPolicy = BackupPolicy::KeepPrevious;
        return options;
    }
};

/** @brief Raised when an artifact cannot be written or committed. */
class ArtifactWriteError final : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/** @brief Returns `path` with `suffix` appended to its last component. */
std::filesystem::path SuffixedPath(std::filesystem::path const& path, char const* suffix);

/**
 * @brief Replaces `path` by an already written `temporary` file.
 * @param context Short owner name used in error messages, e.g. "FlightControl3D checkpoint".
 * @throws ArtifactWriteError when synchronization or the atomic replacement fails.
 */
void CommitTemporaryFile(std::filesystem::path const& temporary, std::filesystem::path const& path,
    AtomicWriteOptions const& options, char const* context);

/**
 * @brief Serializes into a temporary file and atomically replaces `path` with it.
 * @tparam Writer Callable invoked as `writer(std::ostream&)`.
 * @param path Final artifact path.
 * @param writer Serialization step; it may throw, in which case nothing is replaced.
 * @param options Suffixes, backup policy and open mode.
 * @param context Short owner name used in error messages.
 * @throws ArtifactWriteError on I/O failure; exceptions of `writer` propagate unchanged.
 */
template<typename Writer>
void WriteFileAtomically(std::filesystem::path const& path, Writer&& writer,
    AtomicWriteOptions const& options, char const* context)
{
    if (options.createParentDirectories && path.has_parent_path())
    {
        try
        {
            std::filesystem::create_directories(path.parent_path());
        }
        catch (std::filesystem::filesystem_error const& error)
        {
            throw ArtifactWriteError(std::string(context) + ": cannot create parent directories for '"
                + path.u8string() + "' (" + error.what() + ")");
        }
    }
    auto const temporary = detail::UniqueTemporaryPath(path, options.temporarySuffix);
    try
    {
        std::ofstream stream(temporary, options.mode | std::ios::out | std::ios::trunc);
        if (!stream)
        {
            throw ArtifactWriteError(std::string(context) + ": cannot create '" + temporary.u8string() + "'");
        }
        std::forward<Writer>(writer)(static_cast<std::ostream&>(stream));
        stream.flush();
        if (!stream)
        {
            throw ArtifactWriteError(std::string(context) + ": cannot write '" + temporary.u8string() + "'");
        }
        stream.close();
        if (!stream)
        {
            throw ArtifactWriteError(std::string(context) + ": cannot close '" + temporary.u8string() + "'");
        }
    }
    catch (...)
    {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
    CommitTemporaryFile(temporary, path, options, context);
}

} // namespace ppokit::artifacts
