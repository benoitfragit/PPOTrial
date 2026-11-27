/**
 * @file PPOKit/src/schema/json_validation.hpp
 * @brief Private helpers shared by the task and network schema readers.
 * @details Each helper checks one JSON value and, on failure, raises a SchemaError
 * that names the file and the dotted path of the key, e.g.
 * "graph.json: actor.hidden.width: expected a positive integer".
 */
#pragma once

#include "ppokit/schema/task_schema.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <set>
#include <string>
#include <string_view>

namespace ppokit::schema::detail {

/** @brief ordered_json keeps groups and channels in the order the author wrote them. */
using Json = nlohmann::ordered_json;

/** @brief Reports validation errors with the source file and the dotted key path. */
class ErrorReporter final
{
public:
    explicit ErrorReporter(std::string_view sourceName) : source_(sourceName) {}

    [[noreturn]] void Fail(std::string const& path, std::string const& message) const
    {
        throw SchemaError(source_, path, message);
    }

private:
    std::string source_;
};

/** @brief True for a C/C++ identifier: a letter or '_' followed by letters, digits or '_'. */
bool IsIdentifier(std::string const& text) noexcept;

/** @brief True for identifiers separated by "::", such as "ppo::examples::flight". */
bool IsQualifiedName(std::string const& text);

/** @brief Dotted path of `key` below `path`; the root path is empty. */
std::string ChildPath(std::string const& path, std::string const& key);

/** @brief Rejects any key of `object` that is not in `allowed`, listing the accepted keys. */
void RequireKnownKeys(Json const& object, std::set<std::string> const& allowed, std::string const& path,
    ErrorReporter const& errors);

/** @brief Fails unless `value` is an object. */
void RequireObject(Json const& value, std::string const& path, ErrorReporter const& errors);

/** @brief Returns `value` when it is a string holding an identifier, otherwise fails. */
std::string RequireIdentifier(Json const& value, std::string const& path, ErrorReporter const& errors);

/** @brief Returns `value` when it is an integer greater than zero, otherwise fails. */
std::size_t RequirePositiveInteger(Json const& value, std::string const& path, ErrorReporter const& errors);

/** @brief Returns `value` when it is a finite number, otherwise fails. */
double RequireFiniteNumber(Json const& value, std::string const& path, ErrorReporter const& errors);

/** @brief Parses JSON text that must hold an object at its root. */
Json ParseRootObject(std::string_view text, ErrorReporter const& errors);

/** @brief Reads a whole text file. @throws SchemaError when it cannot be opened. */
std::string ReadTextFile(std::filesystem::path const& path);

} // namespace ppokit::schema::detail
