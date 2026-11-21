/**
 * @file PPOKit/include/ppokit/artifacts/atomic_file.hpp
 * @brief Transactional replacement of a file: write a temporary, then commit it.
 * @details A reader never observes a partially written artifact. While the
 * replacement is committed, the previous file is moved to a backup path and
 * restored if the final rename fails. Callers choose whether that backup is
 * kept afterwards, because some workflows archive it (FlightControl3D CI
 * uploads `*.previous`) while others only need the rollback.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>
#include <utility>

namespace ppokit::artifacts {

/** @brief What happens to the previous file once the new one is committed. */
enum class BackupPolicy : std::uint8_t
{
    /** The backup only protects the commit and is deleted after success. */
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
 * @throws ArtifactWriteError when the commit fails; the previous file is then restored.
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
        std::filesystem::create_directories(path.parent_path());
    }
    auto const temporary = SuffixedPath(path, options.temporarySuffix);
    try
    {
        std::ofstream stream(temporary, options.mode | std::ios::out | std::ios::trunc);
        if (!stream)
        {
            throw ArtifactWriteError(std::string(context) + ": cannot create '" + temporary.string() + "'");
        }
        std::forward<Writer>(writer)(static_cast<std::ostream&>(stream));
        stream.flush();
        if (!stream)
        {
            throw ArtifactWriteError(std::string(context) + ": cannot write '" + temporary.string() + "'");
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
