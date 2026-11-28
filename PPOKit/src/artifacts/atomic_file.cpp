/**
 * @file PPOKit/src/artifacts/atomic_file.cpp
 * @brief Durable, concurrent and atomic file replacement on Windows and POSIX.
 */
#include "ppokit/artifacts/atomic_file.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cwctype>
#if defined(_WIN32)
#include <iomanip>
#include <sstream>
#endif
#include <system_error>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ppokit::artifacts {
namespace {

namespace fs = std::filesystem;
constexpr auto LockTimeout = std::chrono::seconds(30);

/** @brief Builds a UTF-8 error message while preserving the native failure. */
std::string FileFailure(char const* context, char const* action, fs::path const& path, std::string const& cause)
{
    return std::string(context) + ": cannot " + action + " '" + path.u8string() + "' (" + cause + ")";
}

/** @brief Best-effort deletion used only while another exception is already active. */
void RemoveQuietly(fs::path const& path) noexcept
{
    std::error_code ignored;
    fs::remove(path, ignored);
}

/** @brief Removes one stale regular file and reports every cleanup failure. */
void RemoveChecked(fs::path const& path, char const* context)
{
    std::error_code error;
    bool const removed = fs::remove(path, error);
    if (error)
    {
        throw ArtifactWriteError(FileFailure(context, "remove stale backup", path, error.message()));
    }
    (void)removed;
}

/** @brief Returns a stable, case-insensitive 64-bit identifier for a normalized Windows target path. */
#if defined(_WIN32)
std::uint64_t PathIdentifier(fs::path const& path)
{
    std::error_code error;
    fs::path normalized = fs::absolute(path, error);
    if (error) normalized = path;
    normalized = normalized.lexically_normal();
    std::uint64_t hash = 14695981039346656037ull;
    for (wchar_t value : normalized.native())
    {
        std::uint32_t const code = static_cast<std::uint32_t>(std::towlower(value));
        for (unsigned int shift = 0; shift < sizeof(wchar_t) * 8; shift += 8)
        {
            hash ^= static_cast<std::uint8_t>(code >> shift);
            hash *= 1099511628211ull;
        }
    }
    return hash;
}
#endif

/** @brief Cross-process lock held only during the commit of one target. */
class TargetLock final
{
public:
    TargetLock(fs::path const& target, char const* context)
    {
#if defined(_WIN32)
        std::wostringstream name;
        name << L"Local\\PPOKitAtomicFileV2_" << std::hex << std::setw(16) << std::setfill(L'0')
             << PathIdentifier(target);
        handle_ = CreateMutexW(nullptr, FALSE, name.str().c_str());
        if (handle_ == nullptr)
        {
            throw ArtifactWriteError(FileFailure(context, "create replacement lock for", target,
                std::error_code(static_cast<int>(GetLastError()), std::system_category()).message()));
        }
        DWORD const result = WaitForSingleObject(handle_, static_cast<DWORD>(LockTimeout.count() * 1000));
        if (result == WAIT_TIMEOUT)
        {
            CloseHandle(handle_);
            handle_ = nullptr;
            throw ArtifactWriteError(FileFailure(context, "acquire replacement lock for", target,
                "timed out after 30 seconds"));
        }
        if (result != WAIT_OBJECT_0 && result != WAIT_ABANDONED)
        {
            DWORD const code = GetLastError();
            CloseHandle(handle_);
            handle_ = nullptr;
            throw ArtifactWriteError(FileFailure(context, "acquire replacement lock for", target,
                std::error_code(static_cast<int>(code), std::system_category()).message()));
        }
#else
        fs::path const directory = target.has_parent_path() ? target.parent_path() : fs::path(".");
        descriptor_ = ::open(directory.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
        if (descriptor_ < 0)
        {
            throw ArtifactWriteError(FileFailure(context, "open replacement directory", directory,
                std::error_code(errno, std::generic_category()).message()));
        }
        auto const deadline = std::chrono::steady_clock::now() + LockTimeout;
        while (::flock(descriptor_, LOCK_EX | LOCK_NB) != 0)
        {
            int const code = errno;
            if (code != EWOULDBLOCK && code != EAGAIN)
            {
                ::close(descriptor_);
                descriptor_ = -1;
                throw ArtifactWriteError(FileFailure(context, "lock replacement directory", directory,
                    std::error_code(code, std::generic_category()).message()));
            }
            if (std::chrono::steady_clock::now() >= deadline)
            {
                ::close(descriptor_);
                descriptor_ = -1;
                throw ArtifactWriteError(FileFailure(context, "lock replacement directory", directory,
                    "timed out after 30 seconds"));
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
#endif
    }

    ~TargetLock() noexcept
    {
#if defined(_WIN32)
        if (handle_ != nullptr)
        {
            ReleaseMutex(handle_);
            CloseHandle(handle_);
        }
#else
        if (descriptor_ >= 0)
        {
            ::flock(descriptor_, LOCK_UN);
            ::close(descriptor_);
        }
#endif
    }

    TargetLock(TargetLock const&) = delete;
    TargetLock& operator=(TargetLock const&) = delete;

private:
#if defined(_WIN32)
    HANDLE handle_ = nullptr;
#else
    int descriptor_ = -1;
#endif
};

/** @brief Flushes the containing directory after a rename on POSIX. */
void SynchronizeDirectory(fs::path const& path, char const* context)
{
#if defined(_WIN32)
    (void)path;
    (void)context;
#else
    fs::path const directory = path.has_parent_path() ? path.parent_path() : fs::path(".");
    int const descriptor = ::open(directory.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY);
    if (descriptor < 0)
    {
        throw ArtifactWriteError(FileFailure(context, "open replacement directory", directory,
            std::error_code(errno, std::generic_category()).message()));
    }
    if (::fsync(descriptor) != 0)
    {
        int const code = errno;
        ::close(descriptor);
        throw ArtifactWriteError(FileFailure(context, "synchronize replacement directory", directory,
            std::error_code(code, std::generic_category()).message()));
    }
    if (::close(descriptor) != 0)
    {
        throw ArtifactWriteError(FileFailure(context, "close replacement directory", directory,
            std::error_code(errno, std::generic_category()).message()));
    }
#endif
}

/** @brief Atomically installs `source` over `target`; both paths are siblings. */
void ReplacePath(fs::path const& source, fs::path const& target, char const* context)
{
#if defined(_WIN32)
    bool const targetExists = fs::exists(target);
    if (targetExists)
    {
        if (ReplaceFileW(target.c_str(), source.c_str(), nullptr, REPLACEFILE_WRITE_THROUGH, nullptr, nullptr) != FALSE)
            return;
        DWORD const code = GetLastError();
        throw ArtifactWriteError(FileFailure(context, "atomically replace", target,
            std::error_code(static_cast<int>(code), std::system_category()).message()));
    }
    if (MoveFileExW(source.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH) != FALSE) return;
    DWORD const code = GetLastError();
    throw ArtifactWriteError(FileFailure(context, "atomically install", target,
        std::error_code(static_cast<int>(code), std::system_category()).message()));
#else
    if (::rename(source.c_str(), target.c_str()) != 0)
    {
        throw ArtifactWriteError(FileFailure(context, "atomically replace", target,
            std::error_code(errno, std::generic_category()).message()));
    }
#endif
}

/** @brief Copies the current target to its durable previous-generation path. */
void PrepareBackup(fs::path const& target, fs::path const& backup, char const* context)
{
    fs::path const staged = detail::UniqueTemporaryPath(backup, ".preparing");
    try
    {
        std::error_code error;
        bool const copied = fs::copy_file(target, staged, fs::copy_options::none, error);
        if (!copied || error)
        {
            throw ArtifactWriteError(FileFailure(context, "prepare previous generation", backup,
                error ? error.message() : std::string("copy did not create a file")));
        }
        detail::SynchronizeFile(staged, context);
        ReplacePath(staged, backup, context);
        SynchronizeDirectory(backup, context);
    }
    catch (...)
    {
        RemoveQuietly(staged);
        throw;
    }
}

} // namespace

namespace detail {

struct ScopedPathLock::Handle final
{
    explicit Handle(fs::path const& path, char const* context) : lock(path, context) {}
    TargetLock lock;
};

ScopedPathLock::ScopedPathLock(fs::path const& path, char const* context)
    : handle_(std::make_unique<Handle>(path, context))
{
}

ScopedPathLock::~ScopedPathLock() = default;

fs::path UniqueTemporaryPath(fs::path const& path, char const* suffix)
{
    static std::atomic<std::uint64_t> sequence{0};
    if (suffix == nullptr || *suffix == '\0')
    {
        throw ArtifactWriteError("atomic file: temporary suffix must not be empty");
    }
#if defined(_WIN32)
    std::uint64_t const process = static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    std::uint64_t const process = static_cast<std::uint64_t>(::getpid());
#endif
    for (;;)
    {
        auto temporary = SuffixedPath(path, suffix);
        temporary += "." + std::to_string(process) + "."
            + std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
        std::error_code error;
        bool const exists = fs::exists(temporary, error);
        if (error)
        {
            throw ArtifactWriteError(FileFailure("atomic file", "inspect temporary", temporary, error.message()));
        }
        if (!exists) return temporary;
    }
}

void SynchronizeFile(fs::path const& path, char const* context)
{
#if defined(_WIN32)
    HANDLE const handle = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle == INVALID_HANDLE_VALUE)
    {
        throw ArtifactWriteError(FileFailure(context, "open for synchronization", path,
            std::error_code(static_cast<int>(GetLastError()), std::system_category()).message()));
    }
    if (FlushFileBuffers(handle) == FALSE)
    {
        DWORD const code = GetLastError();
        CloseHandle(handle);
        throw ArtifactWriteError(FileFailure(context, "synchronize", path,
            std::error_code(static_cast<int>(code), std::system_category()).message()));
    }
    if (CloseHandle(handle) == FALSE)
    {
        throw ArtifactWriteError(FileFailure(context, "close synchronized file", path,
            std::error_code(static_cast<int>(GetLastError()), std::system_category()).message()));
    }
#else
    int const descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor < 0)
    {
        throw ArtifactWriteError(FileFailure(context, "open for synchronization", path,
            std::error_code(errno, std::generic_category()).message()));
    }
    if (::fsync(descriptor) != 0)
    {
        int const code = errno;
        ::close(descriptor);
        throw ArtifactWriteError(FileFailure(context, "synchronize", path,
            std::error_code(code, std::generic_category()).message()));
    }
    if (::close(descriptor) != 0)
    {
        throw ArtifactWriteError(FileFailure(context, "close synchronized file", path,
            std::error_code(errno, std::generic_category()).message()));
    }
#endif
}

} // namespace detail

fs::path SuffixedPath(fs::path const& path, char const* suffix)
{
    auto suffixed = path;
    suffixed += suffix;
    return suffixed;
}

void CommitTemporaryFile(fs::path const& temporary, fs::path const& path,
    AtomicWriteOptions const& options, char const* context)
{
    if (temporary.parent_path() != path.parent_path())
    {
        RemoveQuietly(temporary);
        throw ArtifactWriteError(FileFailure(context, "commit temporary from another directory to", path,
            "atomic replacement requires sibling paths"));
    }

    try
    {
        detail::SynchronizeFile(temporary, context);
        detail::ScopedPathLock const lock(path, context);
        auto const backup = SuffixedPath(path, options.backupSuffix);
        std::error_code existenceError;
        bool const hadPrevious = fs::exists(path, existenceError);
        if (existenceError)
        {
            throw ArtifactWriteError(FileFailure(context, "inspect current generation", path,
                existenceError.message()));
        }
        if (options.backupPolicy == BackupPolicy::KeepPrevious && hadPrevious)
        {
            PrepareBackup(path, backup, context);
        }
        else if (options.backupPolicy == BackupPolicy::RemoveAfterCommit)
        {
            RemoveChecked(backup, context);
        }

        ReplacePath(temporary, path, context);
        SynchronizeDirectory(path, context);
    }
    catch (...)
    {
        RemoveQuietly(temporary);
        throw;
    }
}

} // namespace ppokit::artifacts
