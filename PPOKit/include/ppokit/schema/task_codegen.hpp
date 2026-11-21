/**
 * @file PPOKit/include/ppokit/schema/task_codegen.hpp
 * @brief Generation of the C++ layout header of a task from its validated schema.
 * @details The generated header only contains constants: offsets and sizes of every
 * observation, critic and action group, named slot enums, reward channel names,
 * labels and default weights, and the shape hash. Example code includes it instead
 * of hard-coding offsets, and can `static_assert` that its own constants agree.
 *
 * CMake: `ppokit_generate_task_layout(TARGET <consumer> TASK task.json HEADER <name>.hpp)`
 * (PPOKit/cmake/PPOKitTask.cmake) regenerates the header whenever the task file changes.
 */
#pragma once

#include "ppokit/schema/task_schema.hpp"

#include <string>
#include <string_view>

namespace ppokit::schema {

/**
 * @brief Converts a snake_case or CamelCase identifier to PascalCase ("glide_path_error" -> "GlidePathError").
 */
std::string ToPascalCase(std::string_view identifier);

/**
 * @brief Generates the layout header of `schema`.
 * @param sourceName Task file name quoted in the "generated from" comment.
 * @throws SchemaError when two names map to the same C++ identifier.
 */
std::string GenerateTaskLayoutHeader(TaskSchema const& schema, std::string_view sourceName);

} // namespace ppokit::schema
