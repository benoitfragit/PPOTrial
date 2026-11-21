/**
 * @file PPOKit/src/schema/json_validation.cpp
 * @brief Implementation of the private JSON validation helpers (documented in the header).
 */
#include "json_validation.hpp"

#include <cmath>
#include <fstream>
#include <iterator>

namespace ppokit::schema::detail {
namespace {

/** @brief True for an ASCII letter or '_', the characters that may start an identifier. */
bool IsIdentifierStart(char character) noexcept
{
    return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || character == '_';
}

/** @brief True for a character allowed after the first one of an identifier. */
bool IsIdentifierCharacter(char character) noexcept
{
    return IsIdentifierStart(character) || (character >= '0' && character <= '9');
}

} // namespace

bool IsIdentifier(std::string const& text) noexcept
{
    if (text.empty() || !IsIdentifierStart(text.front()))
    {
        return false;
    }
    for (char const character : text)
    {
        if (!IsIdentifierCharacter(character))
        {
            return false;
        }
    }
    return true;
}

std::string ChildPath(std::string const& path, std::string const& key)
{
    if (path.empty())
    {
        return key;
    }
    return path + "." + key;
}

bool IsQualifiedName(std::string const& text)
{
    std::size_t begin = 0;
    while (true)
    {
        std::size_t const separator = text.find("::", begin);
        std::string const part = text.substr(begin, separator - begin);
        if (!IsIdentifier(part))
        {
            return false;
        }
        if (separator == std::string::npos)
        {
            return true;
        }
        begin = separator + 2;
    }
}

void RequireKnownKeys(Json const& object, std::set<std::string> const& allowed, std::string const& path,
    ErrorReporter const& errors)
{
    for (auto const& item : object.items())
    {
        if (allowed.count(item.key()) == 0)
        {
            std::string accepted;
            for (auto const& key : allowed)
            {
                accepted += accepted.empty() ? key : ", " + key;
            }
            errors.Fail(ChildPath(path, item.key()), "unknown key (accepted: " + accepted + ")");
        }
    }
}

Json const& RequireObject(Json const& value, std::string const& path, ErrorReporter const& errors)
{
    if (!value.is_object())
    {
        errors.Fail(path, "expected an object { ... }");
    }
    return value;
}

std::string RequireIdentifier(Json const& value, std::string const& path, ErrorReporter const& errors)
{
    if (!value.is_string() || !IsIdentifier(value.get<std::string>()))
    {
        errors.Fail(path, "expected an identifier (letters, digits and '_', not starting with a digit)");
    }
    return value.get<std::string>();
}

std::size_t RequirePositiveInteger(Json const& value, std::string const& path, ErrorReporter const& errors)
{
    if (!value.is_number_unsigned() || value.get<std::size_t>() == 0)
    {
        errors.Fail(path, "expected a positive integer");
    }
    return value.get<std::size_t>();
}

double RequireFiniteNumber(Json const& value, std::string const& path, ErrorReporter const& errors)
{
    if (!value.is_number() || !std::isfinite(value.get<double>()))
    {
        errors.Fail(path, "expected a finite number");
    }
    return value.get<double>();
}

Json ParseRootObject(std::string_view text, ErrorReporter const& errors)
{
    Json root;
    try
    {
        root = Json::parse(text);
    }
    catch (Json::parse_error const& error)
    {
        errors.Fail("", std::string("invalid JSON: ") + error.what());
    }
    RequireObject(root, "", errors);
    return root;
}

std::string ReadTextFile(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw SchemaError(path.string() + ": cannot open the file");
    }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

} // namespace ppokit::schema::detail
