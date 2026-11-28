/**
 * @file PPOKit/tests/atomic_file_tests.cpp
 * @brief Regressions of the transactional file replacement shared by the examples.
 */
#include "ppokit/artifacts/atomic_file.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using ppokit::artifacts::AtomicWriteOptions;
using ppokit::artifacts::ArtifactWriteError;
using ppokit::artifacts::WriteFileAtomically;

int failures = 0;

void Check(bool condition, char const* description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

std::string ReadAll(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

void WriteText(std::filesystem::path const& path, std::string const& text, AtomicWriteOptions const& options)
{
    WriteFileAtomically(path, [&text](std::ostream& stream) { stream << text; }, options, "test artifact");
}

/** @brief A fresh scratch directory per test, removed on destruction. */
class ScratchDirectory final
{
public:
    explicit ScratchDirectory(char const* name)
        : path_(std::filesystem::temp_directory_path() / "ppokit_atomic_file_tests" / name)
    {
        std::filesystem::remove_all(path_);
        std::filesystem::create_directories(path_);
    }
    ~ScratchDirectory()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    ScratchDirectory(ScratchDirectory const&) = delete;
    ScratchDirectory& operator=(ScratchDirectory const&) = delete;
    std::filesystem::path const& Path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

void CreatesMissingParentDirectories()
{
    ScratchDirectory scratch("parents");
    auto const path = scratch.Path() / "a" / "b" / "model.txt";
    WriteText(path, "first", AtomicWriteOptions{});
    Check(ReadAll(path) == "first", "new file written through created parents");
    Check(!std::filesystem::exists(path.string() + ".tmp"), "no temporary left after a commit");
    Check(!std::filesystem::exists(path.string() + ".bak"), "no backup for a first generation");
}

void RemovesBackupByDefault()
{
    ScratchDirectory scratch("remove_backup");
    auto const path = scratch.Path() / "model.txt";
    WriteText(path, "first", AtomicWriteOptions{});
    WriteText(path, "second", AtomicWriteOptions{});
    Check(ReadAll(path) == "second", "second generation replaces the first");
    Check(!std::filesystem::exists(path.string() + ".bak"), "RemoveAfterCommit deletes the backup");
}

void KeepsPreviousGenerationWhenRequested()
{
    ScratchDirectory scratch("keep_previous");
    auto const path = scratch.Path() / "model.txt";
    auto const options = AtomicWriteOptions::KeepingPrevious(".previous");
    WriteText(path, "first", options);
    Check(!std::filesystem::exists(path.string() + ".previous"), "first generation has no previous");
    WriteText(path, "second", options);
    WriteText(path, "third", options);
    Check(ReadAll(path) == "third", "latest generation committed");
    Check(ReadAll(path.string() + ".previous") == "second", "only the immediately previous generation is kept");
}

void FailingWriterLeavesArtifactUntouched()
{
    ScratchDirectory scratch("failing_writer");
    auto const path = scratch.Path() / "model.txt";
    WriteText(path, "valid", AtomicWriteOptions{});
    bool thrown = false;
    try
    {
        WriteFileAtomically(path, [](std::ostream& stream) {
            stream << "partial";
            throw std::invalid_argument("serialization refused");
        }, AtomicWriteOptions{}, "test artifact");
    }
    catch (std::invalid_argument const&)
    {
        thrown = true;
    }
    Check(thrown, "writer exception propagates unchanged");
    Check(ReadAll(path) == "valid", "a failed write keeps the previous artifact");
    Check(!std::filesystem::exists(path.string() + ".tmp"), "a failed write removes its temporary");
}

void NestedWritesUseDistinctTemporaries()
{
    ScratchDirectory scratch("nested_writes");
    auto const path = scratch.Path() / "model.txt";
    WriteFileAtomically(path, [&](std::ostream& outer) {
        outer << "outer generation";
        WriteText(path, "inner generation", AtomicWriteOptions{});
    }, AtomicWriteOptions{}, "outer test artifact");
    Check(ReadAll(path) == "outer generation", "nested writes never share or truncate a temporary file");
}

void ConcurrentWritesCommitOnlyCompleteGenerations()
{
    ScratchDirectory scratch("concurrent_writes");
    auto const path = scratch.Path() / "model.txt";
    constexpr std::size_t WriterCount = 6;
    constexpr std::size_t WritesPerThread = 4;
    std::vector<std::string> payloads;
    for (std::size_t index = 0; index < WriterCount; ++index)
    {
        payloads.push_back("generation-" + std::to_string(index) + "\n"
            + std::string(64 * 1024, static_cast<char>('A' + index)));
    }
    std::atomic<bool> start{false};
    std::atomic<int> writeFailures{0};
    std::vector<std::thread> writers;
    for (std::size_t index = 0; index < WriterCount; ++index)
    {
        writers.emplace_back([&, index] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            try
            {
                for (std::size_t write = 0; write < WritesPerThread; ++write)
                    WriteText(path, payloads[index], AtomicWriteOptions::Binary());
            }
            catch (std::exception const&)
            {
                writeFailures.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& writer : writers) writer.join();
    std::string const committed = ReadAll(path);
    Check(writeFailures.load(std::memory_order_relaxed) == 0, "concurrent writers all complete successfully");
    Check(std::find(payloads.begin(), payloads.end(), committed) != payloads.end(),
        "the final file is one complete concurrent generation");
}

void UnicodePathRoundTrips()
{
    ScratchDirectory scratch("unicode");
    auto const path = scratch.Path() / std::filesystem::path(u8"données_日本")
        / std::filesystem::path(u8"modèle é.txt");
    WriteText(path, "unicode path payload", AtomicWriteOptions{});
    Check(ReadAll(path) == "unicode path payload", "UTF-8 and UTF-16 path components round-trip");
}

void InvalidTemporarySuffixKeepsArtifact()
{
    ScratchDirectory scratch("invalid_suffix");
    auto const path = scratch.Path() / "model.txt";
    WriteText(path, "valid", AtomicWriteOptions{});
    AtomicWriteOptions options;
    options.temporarySuffix = "";
    bool thrown = false;
    try
    {
        WriteText(path, "lost", options);
    }
    catch (ArtifactWriteError const& error)
    {
        thrown = std::string(error.what()).find("suffix") != std::string::npos;
    }
    Check(thrown, "an invalid temporary suffix is reported");
    Check(ReadAll(path) == "valid", "invalid temporary configuration keeps the previous artifact");
}

void BlockedBackupKeepsArtifact()
{
    ScratchDirectory scratch("blocked_backup");
    auto const path = scratch.Path() / "model.txt";
    WriteText(path, "valid", AtomicWriteOptions{});
    // A non-empty directory squatting the backup path makes checked cleanup fail.
    std::filesystem::create_directories(scratch.Path() / "model.txt.bak" / "occupied");
    bool thrown = false;
    try
    {
        WriteText(path, "lost", AtomicWriteOptions{});
    }
    catch (ArtifactWriteError const& error)
    {
        thrown = std::string(error.what()).find("remove stale backup") != std::string::npos;
    }
    Check(thrown, "a failed commit raises ArtifactWriteError");
    Check(ReadAll(path) == "valid", "a failed commit keeps the previous artifact");
    Check(!std::filesystem::exists(path.string() + ".tmp"), "a failed commit removes its temporary");
}

void HonoursBinaryMode()
{
    ScratchDirectory scratch("binary");
    auto const path = scratch.Path() / "model.bin";
    WriteText(path, std::string("a\nb\0c", 5), AtomicWriteOptions::Binary());
    Check(ReadAll(path) == std::string("a\nb\0c", 5), "binary mode writes bytes unchanged");
}

} // namespace

int main()
{
    CreatesMissingParentDirectories();
    RemovesBackupByDefault();
    KeepsPreviousGenerationWhenRequested();
    FailingWriterLeavesArtifactUntouched();
    NestedWritesUseDistinctTemporaries();
    ConcurrentWritesCommitOnlyCompleteGenerations();
    UnicodePathRoundTrips();
    InvalidTemporarySuffixKeepsArtifact();
    BlockedBackupKeepsArtifact();
    HonoursBinaryMode();
    if (failures != 0)
    {
        std::cerr << failures << " atomic file check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit atomic file tests passed\n";
    return EXIT_SUCCESS;
}
