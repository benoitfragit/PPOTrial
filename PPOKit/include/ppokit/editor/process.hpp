/**
 * @file PPOKit/include/ppokit/editor/process.hpp
 * @brief A program started by the studio (training, evaluation, inference), with its
 *        console output collected line by line while it runs.
 * @details The output (stdout and stderr together) is read by a background thread into a
 * bounded list of lines; the studio takes the new lines each frame. Kill ends the program
 * at once; a clean stop goes through the project (RequestStop in command_line.hpp terms:
 * a stop file the program checks between updates). No GUI.
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::editor {

class Process final
{
public:
    /** @brief Lines kept; older ones are dropped (the journal keeps the training history). */
    static constexpr std::size_t MaximumLines = 5000;

    Process();
    /** @brief Kills the program if it still runs. */
    ~Process();
    Process(Process const&) = delete;
    Process& operator=(Process const&) = delete;

    /**
     * @brief Starts `program` with `arguments` in `folder`.
     * @throws std::runtime_error when a program already runs or it cannot be started.
     */
    void Start(std::filesystem::path const& program, std::vector<std::string> const& arguments,
        std::filesystem::path const& folder);

    /** @brief Whether the program runs (false before Start and once it exited). */
    bool Running();

    /** @brief Exit code once the program exited. */
    std::optional<int> ExitCode();

    /** @brief Ends the program at once (its last checkpoint stays). */
    void Kill();

    /** @brief Lines written since the last call. */
    std::vector<std::string> TakeLines();

    /** @brief Command line of the last Start, for display. */
    std::string const& CommandLine() const noexcept { return commandLine_; }

private:
    struct State;
    std::unique_ptr<State> state_;
    std::string commandLine_;
};

/** @brief A command line as a shell would show it: arguments with spaces quoted. */
std::string QuoteCommandLine(std::filesystem::path const& program, std::vector<std::string> const& arguments);

} // namespace ppokit::editor
