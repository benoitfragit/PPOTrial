/**
 * @file PPOKit/include/ppokit/tools/scaffold.hpp
 * @brief Writes a new project that already trains: what ppokit-new and the studio's
 *        "New project" create (project.json, task.json, network.json, training.json,
 *        eval/, components/, task.hpp, main.cpp, CMakeLists.txt, README.md).
 */
#pragma once

#include <filesystem>
#include <string>

namespace ppokit::tools {

/** @brief Lowercase letters, digits and '_', starting with a letter. */
bool ValidProjectName(std::string const& name) noexcept;

/** @brief "my_task" -> "MyTask", the C++ class name of the task. */
std::string ProjectTypeName(std::string const& name);

/**
 * @brief Writes `into`/`name`/ and returns that folder.
 * @throws std::invalid_argument for an invalid name; std::runtime_error when the folder
 *         exists and is not empty (nothing is written then) or a file cannot be written.
 */
std::filesystem::path WriteNewProject(std::string const& name, std::filesystem::path const& into);

} // namespace ppokit::tools
