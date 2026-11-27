/**
 * @file PPOKit/src/schema/code_writer.hpp
 * @brief Private helpers that format generated C++ code.
 */
#pragma once

#include <cstddef>
#include <sstream>
#include <string>

namespace ppokit::schema::detail {

/** @brief Accumulates indented lines of generated code. */
class CodeWriter final
{
public:
    void Line(std::string const& text = {})
    {
        if (!text.empty())
        {
            text_.append(static_cast<std::size_t>(indent_) * 4, ' ');
            text_ += text;
        }
        text_ += '\n';
    }
    void Open(std::string const& text)
    {
        Line(text);
        Line("{");
        ++indent_;
    }
    void Close(std::string const& suffix = {})
    {
        --indent_;
        Line("}" + suffix);
    }
    std::string const& Text() const noexcept { return text_; }

private:
    std::string text_;
    int indent_ = 0;
};

/** @brief Shortest decimal text that reads back as the same double. */
inline std::string DoubleLiteral(double value)
{
    std::ostringstream text;
    text.precision(17);
    text << value;
    std::string literal = text.str();
    if (literal.find_first_of(".eE") == std::string::npos)
    {
        literal += ".0";
    }
    return literal;
}

/** @brief C++ string literal of `text`, with quotes and backslashes escaped. */
inline std::string Quoted(std::string const& text)
{
    std::string quoted = "\"";
    for (char const character : text)
    {
        if (character == '"' || character == '\\')
        {
            quoted += '\\';
        }
        quoted += character;
    }
    return quoted + "\"";
}

} // namespace ppokit::schema::detail
