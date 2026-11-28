#include "ppokit/runtime/writer_lock.hpp"

#include <set>
#include <stdexcept>
#include <system_error>
#if defined(_WIN32)
#include <Windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace ppokit::runtime {

struct WriterLock::Handle final
{
#if defined(_WIN32)
    HANDLE file = INVALID_HANDLE_VALUE;
    ~Handle() { if (file != INVALID_HANDLE_VALUE) CloseHandle(file); }
#else
    int file = -1;
    ~Handle() { if (file >= 0) ::close(file); }
#endif
};

WriterLock::WriterLock(std::vector<std::filesystem::path> const& outputs)
{
    namespace fs = std::filesystem;
    std::set<fs::path> directories;
    for (auto const& output : outputs)
    {
        auto const directory = fs::weakly_canonical(fs::absolute(output)).parent_path();
        fs::create_directories(directory);
        directories.insert(directory);
    }
    for (auto const& directory : directories)
    {
        auto const lock = directory / ".ppokit-writer.lock";
        auto handle = std::make_unique<Handle>();
#if defined(_WIN32)
        handle->file = CreateFileW(lock.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
            nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        OVERLAPPED position{};
        bool const acquired = handle->file != INVALID_HANDLE_VALUE && LockFileEx(handle->file,
            LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &position);
#else
        handle->file = ::open(lock.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
        int result = -1;
        if (handle->file >= 0)
        {
            do { result = ::flock(handle->file, LOCK_EX | LOCK_NB); } while (result < 0 && errno == EINTR);
        }
        bool const acquired = result == 0;
#endif
        if (!acquired)
        {
            throw std::runtime_error("cannot acquire training writer lock in " + directory.string());
        }
        held_.push_back(std::move(handle));
    }
}

WriterLock::~WriterLock() = default;

} // namespace ppokit::runtime
