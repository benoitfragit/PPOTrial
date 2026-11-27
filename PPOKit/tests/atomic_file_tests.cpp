/**
 * @file PPOKit/tests/atomic_file_tests.cpp
 * @brief Regressions of the transactional file replacement shared by the examples.
 */
#include "ppokit/artifacts/atomic_file.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>

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

void UnwritableTemporaryKeepsArtifact()
{
    ScratchDirectory scratch("failing_commit");
    auto const path = scratch.Path() / "model.txt";
    WriteText(path, "valid", AtomicWriteOptions{});
    // A non-empty directory squatting the temporary path makes the write fail
    // before any rename, on every platform.
    std::filesystem::create_directories(scratch.Path() / "model.txt.tmp" / "occupied");
    bool thrown = false;
    try
    {
        WriteText(path, "lost", AtomicWriteOptions{});
    }
    catch (ArtifactWriteError const& error)
    {
        thrown = std::string(error.what()).find("test artifact") != std::string::npos;
    }
    Check(thrown, "I/O failure raises ArtifactWriteError naming its owner");
    Check(ReadAll(path) == "valid", "the artifact survives a failed replacement");
}

void BlockedBackupKeepsArtifact()
{
    ScratchDirectory scratch("blocked_backup");
    auto const path = scratch.Path() / "model.txt";
    WriteText(path, "valid", AtomicWriteOptions{});
    // A non-empty directory squatting the backup path makes the commit fail
    // when the previous generation is moved aside.
    std::filesystem::create_directories(scratch.Path() / "model.txt.bak" / "occupied");
    bool thrown = false;
    try
    {
        WriteText(path, "lost", AtomicWriteOptions{});
    }
    catch (ArtifactWriteError const& error)
    {
        thrown = std::string(error.what()).find("cannot replace") != std::string::npos;
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
    UnwritableTemporaryKeepsArtifact();
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
