/**
 * @file PPOKit/src/editor/recent_projects.cpp
 * @brief The recent project list, one folder per line.
 */
#include "ppokit/editor/recent_projects.hpp"
#include "ppokit/artifacts/atomic_file.hpp"

#include <cstdlib>
#include <fstream>
#include <string>
#include <system_error>

namespace ppokit::editor {
namespace {

std::vector<std::filesystem::path> ReadLines(std::filesystem::path const& file)
{
    std::vector<std::filesystem::path> folders;
    std::ifstream stream(file);
    std::string line;
    while (std::getline(stream, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (!line.empty())
        {
            folders.push_back(std::filesystem::u8path(line));
        }
    }
    return folders;
}

void WriteLines(std::filesystem::path const& file, std::vector<std::filesystem::path> const& folders)
{
    artifacts::WriteFileAtomically(file, [&folders](std::ostream& stream) {
        for (std::filesystem::path const& folder : folders) stream << folder.u8string() << '\n';
    }, artifacts::AtomicWriteOptions{}, "PPOKit recent projects");
}

/** @brief `folder` made absolute and normal, so that one folder is remembered once. */
std::filesystem::path Normal(std::filesystem::path const& folder)
{
    std::error_code error;
    std::filesystem::path const absolute = std::filesystem::absolute(folder, error);
    std::filesystem::path normal = (error ? folder : absolute).lexically_normal();
    // "a/." normalizes to "a/": drop the trailing separator so that both name the same folder.
    if (normal.filename().empty() && normal.has_relative_path())
    {
        normal = normal.parent_path();
    }
    return normal;
}

} // namespace

std::filesystem::path DefaultRecentProjectsFile()
{
#if defined(_WIN32)
    wchar_t const* const appData = _wgetenv(L"APPDATA");
    std::filesystem::path const base = appData != nullptr ? std::filesystem::path(appData) : std::filesystem::path(".");
    return base / "PPOKit" / "recent_projects.txt";
#else
    char const* const config = std::getenv("XDG_CONFIG_HOME");
    char const* const home = std::getenv("HOME");
    std::filesystem::path base = ".";
    if (config != nullptr && config[0] != '\0')
    {
        base = config;
    }
    else if (home != nullptr)
    {
        base = std::filesystem::path(home) / ".config";
    }
    return base / "ppokit" / "recent_projects.txt";
#endif
}

std::vector<std::filesystem::path> LoadRecentProjects(std::filesystem::path const& file)
{
    std::vector<std::filesystem::path> existing;
    for (std::filesystem::path const& folder : ReadLines(file))
    {
        std::error_code error;
        if (std::filesystem::is_directory(folder, error) && existing.size() < MaximumRecentProjects)
        {
            existing.push_back(folder);
        }
    }
    return existing;
}

void RememberProject(std::filesystem::path const& file, std::filesystem::path const& folder)
{
    try
    {
        std::filesystem::path const remembered = Normal(folder);
        std::vector<std::filesystem::path> folders{remembered};
        for (std::filesystem::path const& other : ReadLines(file))
        {
            if (Normal(other) != remembered && folders.size() < MaximumRecentProjects)
            {
                folders.push_back(other);
            }
        }
        WriteLines(file, folders);
    }
    catch (std::exception const&) {}
}

void ForgetProject(std::filesystem::path const& file, std::filesystem::path const& folder)
{
    try
    {
        std::filesystem::path const forgotten = Normal(folder);
        std::vector<std::filesystem::path> kept;
        for (std::filesystem::path const& other : ReadLines(file))
        {
            if (Normal(other) != forgotten)
            {
                kept.push_back(other);
            }
        }
        WriteLines(file, kept);
    }
    catch (std::exception const&) {}
}

} // namespace ppokit::editor
