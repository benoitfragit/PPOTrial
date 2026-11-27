#pragma once

#include <filesystem>
#include <memory>
#include <vector>

namespace ppokit::runtime {

/** @brief An exclusive, nonblocking OS lock on each canonical output directory.
 * @details The files outlive a process. The kernel releases their locks on crash or exit.
 * Readers can open published artifacts without taking a writer lock.
 */
class WriterLock final
{
public:
    explicit WriterLock(std::vector<std::filesystem::path> const& outputs);
    ~WriterLock();
    WriterLock(WriterLock const&) = delete;
    WriterLock& operator=(WriterLock const&) = delete;

private:
    struct Handle;
    std::vector<std::unique_ptr<Handle>> held_;
};

} // namespace ppokit::runtime
