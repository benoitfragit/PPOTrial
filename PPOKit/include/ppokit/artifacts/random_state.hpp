/**
 * @file PPOKit/include/ppokit/artifacts/random_state.hpp
 * @brief Portable text form of a std::mt19937 state, so a checkpoint resumes on Windows and
 *        on Linux alike.
 * @details The C++ standard defines the text of a Mersenne Twister as its last n = 624
 * generated values X(i-n) ... X(i-1). MSVC and libc++ write that form; libstdc++ writes its
 * buffer of 624 words and then its position p (625 numbers), which the other libraries cannot
 * read, and it cannot read theirs. Checkpoints always hold the standard form; reading accepts
 * both forms and converts to what the local library expects.
 *
 * libstdc++ buffer to standard form: after a twist the buffer holds the block X(k) ... X(k+623)
 * and p values of it are consumed, so the last 624 values are X(k+p-624) ... X(k+p-1): the end
 * of the previous block, then the first p values of the buffer. The previous block is
 * recovered exactly by undoing the twist X(t+624) = X(t+397) ^ T(upper(X(t)) | lower(X(t+1))),
 * whose transform T(y) = (y >> 1) ^ (odd(y) ? A : 0) is invertible (A has its top bit set).
 * Standard form to libstdc++: the 624 values with p = 624 (the next call twists first).
 */
#pragma once

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppokit::artifacts {

namespace random_state_detail {

constexpr std::size_t StateSize = 624;
constexpr std::size_t ShiftSize = 397;
constexpr std::uint32_t MatrixA = 0x9908b0dfU;
constexpr std::uint32_t UpperMask = 0x80000000U;
constexpr std::uint32_t LowerMask = 0x7fffffffU;

using Words = std::vector<std::uint32_t>;

/** @brief The numbers of a line of text. */
inline Words ReadWords(std::string const& text)
{
    std::istringstream stream(text);
    Words words;
    std::string token;
    while (stream >> token)
    {
        std::uint32_t value = 0;
        auto const parsed = std::from_chars(token.data(), token.data() + token.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size())
        {
            throw std::runtime_error("PPOKit random state: invalid unsigned 32-bit word");
        }
        words.push_back(value);
    }
    return words;
}

/** @brief Inverse of the twist transform T: the word y whose transform is `z`. */
constexpr std::uint32_t UntwistWord(std::uint32_t z) noexcept
{
    // T(y) has its top bit set exactly when y was odd (y >> 1 never has it, A does).
    std::uint32_t const odd = z >> 31U;
    std::uint32_t const shifted = odd != 0U ? z ^ MatrixA : z;
    return (shifted << 1U) | odd;
}

/**
 * @brief The standard form (last 624 values) from a libstdc++ buffer and position.
 * @param buffer X(k) ... X(k+623), the block of the last twist (or the seeded state).
 * @param position Values of the block already consumed, 1..624.
 */
inline Words StandardFromBuffer(Words const& buffer, std::size_t position)
{
    if (position == StateSize)
    {
        return buffer; // Nothing of a previous block is needed.
    }
    // previous[j] = X(k-624+j): rebuilt from the highest index down, so that the X(t+397)
    // term of each step (a later previous value, or a buffer value) is already known.
    Words previous(StateSize, 0U);
    for (std::size_t j = StateSize; j-- > position - 1U;)
    {
        std::size_t const later = j + ShiftSize;
        std::uint32_t const addend = later < StateSize ? previous[later] : buffer[later - StateSize];
        std::uint32_t const y = UntwistWord(buffer[j] ^ addend);
        previous[j] = (previous[j] & LowerMask) | (y & UpperMask);
        if (j + 1U < StateSize)
        {
            previous[j + 1U] = (previous[j + 1U] & UpperMask) | (y & LowerMask);
        }
    }
    Words standard(previous.begin() + static_cast<std::ptrdiff_t>(position), previous.end());
    standard.insert(standard.end(), buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(position));
    return standard;
}

/** @brief Number of values the local library writes for an engine: 624 (standard) or 625 (libstdc++). */
inline std::size_t LocalWordCount()
{
    std::ostringstream text;
    text << std::mt19937{};
    return ReadWords(text.str()).size();
}

inline std::string Join(Words const& words)
{
    std::ostringstream text;
    for (std::size_t index = 0; index < words.size(); ++index)
    {
        text << (index == 0U ? "" : " ") << words[index];
    }
    return text.str();
}

} // namespace random_state_detail

/** @brief The standard text of an engine's state: its last 624 values, separated by spaces. */
inline std::string StandardRandomText(std::mt19937 const& engine)
{
    namespace detail = random_state_detail;
    std::ostringstream local;
    local << engine;
    detail::Words const words = detail::ReadWords(local.str());
    if (words.size() == detail::StateSize)
    {
        return detail::Join(words);
    }
    if (words.size() != detail::StateSize + 1U || words.back() == 0U || words.back() > detail::StateSize)
    {
        throw std::logic_error("PPOKit random state: unexpected text of std::mt19937 in this library");
    }
    detail::Words const buffer(words.begin(), words.end() - 1);
    return detail::Join(detail::StandardFromBuffer(buffer, words.back()));
}

/**
 * @brief An engine from the text of its state, in the standard form or the libstdc++ form.
 * @param source Named in the error of a damaged state.
 * @throws std::runtime_error for anything else than 624 or 625 numbers.
 */
inline std::mt19937 ParseRandomText(std::string const& text, std::string const& source)
{
    namespace detail = random_state_detail;
    detail::Words words;
    try
    {
        words = detail::ReadWords(text);
    }
    catch (std::runtime_error const& error)
    {
        throw std::runtime_error(source + ": damaged checkpoint: " + error.what());
    }
    bool const libstdcxx = words.size() == detail::StateSize + 1U;
    if (libstdcxx && (words.back() == 0U || words.back() > detail::StateSize))
    {
        throw std::runtime_error(source + ": damaged checkpoint: random state position out of range");
    }
    if (libstdcxx)
    {
        detail::Words const buffer(words.begin(), words.end() - 1);
        words = detail::StandardFromBuffer(buffer, words.back());
    }
    if (words.size() != detail::StateSize)
    {
        throw std::runtime_error(source + ": damaged checkpoint: the random state holds "
            + std::to_string(words.size()) + " numbers, expected 624");
    }
    std::string local = detail::Join(words);
    if (detail::LocalWordCount() == detail::StateSize + 1U)
    {
        local += " 624"; // libstdc++: the next value starts a new block from these 624.
    }
    std::istringstream stream(local);
    std::mt19937 engine;
    if (!(stream >> engine))
    {
        throw std::runtime_error(source + ": damaged checkpoint: unreadable random state");
    }
    return engine;
}

} // namespace ppokit::artifacts
