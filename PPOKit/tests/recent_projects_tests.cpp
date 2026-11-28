/**
 * @file PPOKit/tests/recent_projects_tests.cpp
 * @brief The studio's recent projects: most recent first, once each, bounded, vanished
 *        folders skipped, forgetting one.
 */
#include "ppokit/editor/recent_projects.hpp"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

} // namespace

int main()
{
    namespace editor = ppokit::editor;
    std::filesystem::path const root = std::filesystem::temp_directory_path() / "ppokit_recent_projects_tests";
    std::error_code removed;
    std::filesystem::remove_all(root, removed);
    std::filesystem::path const file = root / "config" / "recent_projects.txt";
    Check(editor::LoadRecentProjects(file).empty(), "no file, no recent project");

    for (char const* name : {"a", "b", "c"})
    {
        std::filesystem::create_directories(root / name);
        editor::RememberProject(file, root / name);
    }
    editor::RememberProject(file, root / "a" / ".");
    auto recent = editor::LoadRecentProjects(file);
    Check(recent.size() == 3 && recent[0].filename() == "a" && recent[1].filename() == "c",
        "the last opened comes first, and a folder is listed once");

    std::filesystem::remove_all(root / "c");
    recent = editor::LoadRecentProjects(file);
    Check(recent.size() == 2 && recent[1].filename() == "b", "a folder that no longer exists is skipped");

    editor::ForgetProject(file, root / "a");
    recent = editor::LoadRecentProjects(file);
    Check(recent.size() == 1 && recent[0].filename() == "b", "a forgotten folder is gone");

    for (std::size_t index = 0; index < editor::MaximumRecentProjects + 3; ++index)
    {
        std::filesystem::path const folder = root / ("p" + std::to_string(index));
        std::filesystem::create_directories(folder);
        editor::RememberProject(file, folder);
    }
    Check(editor::LoadRecentProjects(file).size() == editor::MaximumRecentProjects, "the list is bounded");
    Check(!editor::DefaultRecentProjectsFile().empty(), "there is a default place for the list");

    std::filesystem::remove_all(root, removed);
    if (failures != 0)
    {
        std::cerr << failures << " recent project check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "recent project checks passed\n";
    return EXIT_SUCCESS;
}
