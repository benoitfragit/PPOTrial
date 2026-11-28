/**
 * @file PPOKit/tests/process_tests.cpp
 * @brief Programs started by the studio: output collected line by line, arguments with
 *        spaces and quotes passed intact, exit codes, kill, a missing program.
 */
#include "ppokit/editor/process.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ppokit;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/** @brief Waits (at most 20 s) until the program ends, collecting its lines. */
std::vector<std::string> WaitForExit(editor::Process& process)
{
    std::vector<std::string> lines;
    for (int attempt = 0; attempt < 2000 && process.Running(); ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    for (std::string& line : process.TakeLines())
    {
        lines.push_back(std::move(line));
    }
    return lines;
}

bool Has(std::vector<std::string> const& lines, std::string const& line)
{
    for (std::string const& candidate : lines)
    {
        if (candidate == line)
        {
            return true;
        }
    }
    return false;
}

void OutputAndExitCode()
{
    editor::Process process;
    Check(!process.Running() && !process.ExitCode(), "nothing runs before Start");
    process.Start(PPOKIT_ECHO_PROGRAM, {"plain", "with space", "with \"quotes\"", "trailing\\", "--exit", "3"},
        std::filesystem::temp_directory_path());
    std::vector<std::string> const lines = WaitForExit(process);
    Check(process.ExitCode() == 3, "the exit code is reported");
    Check(Has(lines, "arg: plain") && Has(lines, "arg: with space") && Has(lines, "arg: with \"quotes\"")
            && Has(lines, "arg: trailing\\"),
        "arguments with spaces, quotes and backslashes arrive intact");
    Check(Has(lines, "to stderr"), "errors are collected with the output");
    Check(process.TakeLines().empty(), "lines are handed over once");
}

void KillAndRestart()
{
    editor::Process process;
    process.Start(PPOKIT_ECHO_PROGRAM, {"--wait"}, {});
    bool waiting = false;
    for (int attempt = 0; attempt < 2000 && !waiting; ++attempt)
    {
        for (std::string const& line : process.TakeLines())
        {
            waiting = waiting || line == "waiting";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    Check(waiting && process.Running(), "a waiting program runs, and its lines come while it runs");
    bool refused = false;
    try
    {
        process.Start(PPOKIT_ECHO_PROGRAM, {}, {});
    }
    catch (std::exception const&)
    {
        refused = true;
    }
    Check(refused, "a second Start while one runs is refused");
    process.Kill();
    WaitForExit(process);
    Check(!process.Running() && process.ExitCode().has_value(), "Kill ends the program");
    process.Start(PPOKIT_ECHO_PROGRAM, {"again"}, {});
    Check(Has(WaitForExit(process), "arg: again") && process.ExitCode() == 0, "the process can start again");
}

void MissingProgram()
{
    editor::Process process;
    bool failed = false;
    try
    {
        process.Start(std::filesystem::temp_directory_path() / "no_such_program_ppokit", {}, {});
        WaitForExit(process);
        failed = process.ExitCode().value_or(0) != 0;
    }
    catch (std::exception const&)
    {
        failed = true;
    }
    Check(failed, "a missing program fails, at start or with a non-zero exit code");
}

/** @brief Executable and working-directory paths with spaces and Unicode survive launch. */
void UnicodePaths()
{
    std::filesystem::path const root = std::filesystem::temp_directory_path()
        / std::filesystem::u8path(u8"ppokit processus été") / std::filesystem::u8path(u8"répertoire 测试");
    std::error_code error;
    std::filesystem::remove_all(root.parent_path(), error);
    error.clear();
    std::filesystem::create_directories(root, error);
    Check(!error, "the Unicode process test folder is created");
    std::filesystem::path const source(PPOKIT_ECHO_PROGRAM);
    std::filesystem::path copyName = std::filesystem::u8path(u8"écho 测试");
    copyName += source.extension();
    std::filesystem::path const copy = root / copyName;
    std::filesystem::copy_file(source, copy, std::filesystem::copy_options::overwrite_existing, error);
    Check(!error, "the executable is copied to a Unicode path");
    if (!error)
    {
        editor::Process process;
        process.Start(copy, {"unicode-path"}, root);
        Check(Has(WaitForExit(process), "arg: unicode-path") && process.ExitCode() == 0,
            "a program starts from Unicode executable and working-directory paths");
    }
    std::filesystem::remove_all(root.parent_path(), error);
}

/** @brief A descendant holding stdout cannot delay completion or the next launch. */
void InheritedPipe()
{
    editor::Process process;
    auto const begin = std::chrono::steady_clock::now();
    process.Start(PPOKIT_ECHO_PROGRAM, {"--hold-pipe"}, {});
    auto const lines = WaitForExit(process);
    Check(std::chrono::steady_clock::now() - begin < std::chrono::seconds(5), "inherited pipe completes promptly");
    Check(process.ExitCode() == 7 && Has(lines, "inherited"), "parent exit code and final output survive");
    process.Start(PPOKIT_ECHO_PROGRAM, {"again"}, {});
    Check(Has(WaitForExit(process), "arg: again"), "restart after a descendant inherited the pipe");
}

void Quoting()
{
    Check(editor::QuoteCommandLine("/bin/tool", {"a", "b c", ""}) == "/bin/tool a \"b c\" \"\"",
        "the shown command line quotes arguments with spaces");
}

} // namespace

int main()
{
    OutputAndExitCode();
    KillAndRestart();
    InheritedPipe();
    MissingProgram();
    UnicodePaths();
    Quoting();
    if (failures != 0)
    {
        std::cerr << failures << " process check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "process checks passed\n";
    return EXIT_SUCCESS;
}
