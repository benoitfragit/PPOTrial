/**
 * @file PPOKit/include/ppokit/editor/recent_projects.hpp
 * @brief The project folders the user opened in the studio, most recent first (and only
 *        those: the studio never searches the disk for projects).
 */
#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace ppokit::editor {

/** @brief Folders remembered at most. */
inline constexpr std::size_t MaximumRecentProjects = 8;

/** @brief %APPDATA%/PPOKit/recent_projects.txt on Windows, ~/.config/ppokit/recent_projects.txt elsewhere. */
std::filesystem::path DefaultRecentProjectsFile();

/** @brief The remembered folders that still exist, most recent first (none when the file is missing). */
std::vector<std::filesystem::path> LoadRecentProjects(std::filesystem::path const& file);

/** @brief Puts `folder` first (once), keeps MaximumRecentProjects, writes the file. Best effort: errors are ignored. */
void RememberProject(std::filesystem::path const& file, std::filesystem::path const& folder);

/** @brief Forgets `folder`. Best effort. */
void ForgetProject(std::filesystem::path const& file, std::filesystem::path const& folder);

} // namespace ppokit::editor
