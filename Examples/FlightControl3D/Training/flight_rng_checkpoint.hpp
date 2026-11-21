/**
 * @file flight_rng_checkpoint.hpp
 * @brief Portable restoration of the std::mt19937 state stored in learner checkpoints.
 *
 * Role in the chain: checkpoint reading (flight_learner.cpp). The learner RNG drives
 * action sampling and minibatch shuffles, so a resumed run must continue the exact same
 * random sequence. The standard streams operator<< of std::mt19937 differs between
 * standard libraries:
 * - MSVC writes the 624 words that precede the next output (the recurrence window);
 * - libstdc++ writes the current 624-word block followed by the index of its next word.
 * ReadCheckpointRng accepts both and converts them to the native representation.
 *
 * Not here: writing (the native operator<< is used by FlightLearner::Write).
 *
 * Read next: flight_learner.cpp (Read), tests/learner_tests.cpp (PortableRngCheckpoint).
 */
#pragma once

#include <array>
#include <cstdint>
#include <istream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ppo::examples::flight::detail {

/**
 * @brief Reverses the in-place twist of a partly consumed libstdc++ block.
 *
 * The Mersenne Twister computes word i of the next block from words i, i + 1 and i + m
 * (m = 397) of the recurrence. Walking the block backward recovers the 624 preceding words,
 * so that the window starting at the libstdc++ position can be rebuilt.
 * @param state Current block, as written by libstdc++.
 * @param position Index of the next word of the block, in [0, 624).
 * @return The 624 words preceding the next output, in MSVC window order.
 */
inline std::array<std::uint32_t, std::mt19937::state_size> UntwistedWindow(
    std::array<std::uint32_t, std::mt19937::state_size> const& state, std::size_t position)
{
    constexpr std::size_t count = std::mt19937::state_size;
    // Twist offset m of mt19937.
    constexpr std::size_t shift = std::mt19937::shift_size;
    // count - shift: words whose twist partner wraps into the new block.
    constexpr std::size_t wrapped = count - shift;
    constexpr std::uint32_t upper = 0x80000000U;
    constexpr std::uint32_t lower = 0x7fffffffU;
    constexpr std::uint32_t twist = 0x9908b0dfU;
    static_assert(shift == 397 && wrapped == 227, "mt19937 twist parameters");
    std::array<std::uint32_t, count> previous{};
    for (std::size_t index = count; index-- > 0;)
    {
        auto const reference = index < wrapped ? previous[index + shift] : state[index - wrapped];
        auto const mixed = state[index] ^ reference;
        auto const bit = mixed >> 31U;
        auto const joined = ((mixed ^ (bit ? twist : 0U)) << 1U) | bit;
        previous[index] = (previous[index] & lower) | (joined & upper);
        if (index + 1 < count)
        {
            previous[index + 1] = (previous[index + 1] & upper) | (joined & lower);
        }
        else if ((joined & lower) != (state[0] & lower))
        {
            throw std::runtime_error("FlightControl3D: inconsistent RNG state");
        }
    }
    std::array<std::uint32_t, count> window{};
    std::size_t out = 0;
    for (std::size_t index = position; index < count; ++index)
    {
        window[out++] = previous[index];
    }
    for (std::size_t index = 0; index < position; ++index)
    {
        window[out++] = state[index];
    }
    return window;
}

/**
 * @brief Read either libstdc++ or MSVC's mt19937 stream while preserving its sequence.
 * @param input Checkpoint stream positioned at the beginning of the RNG line.
 * @return Restored engine with the same future integer outputs.
 * @throws std::runtime_error for a missing, truncated, inconsistent or trailing state.
 */
inline std::mt19937 ReadCheckpointRng(std::istream& input)
{
    constexpr std::size_t count = std::mt19937::state_size;
    std::string line;
    if (!std::getline(input >> std::ws, line))
    {
        throw std::runtime_error("FlightControl3D: missing RNG state");
    }
    std::istringstream source(line);
    std::array<std::uint32_t, count> state{};
    for (auto& value : state)
    {
        std::uint64_t word{};
        if (!(source >> word) || word > UINT32_MAX)
        {
            throw std::runtime_error("FlightControl3D: invalid RNG state");
        }
        value = static_cast<std::uint32_t>(word);
    }
    source >> std::ws;
    // A 625th number is the libstdc++ position; MSVC writes none.
    bool const hasPosition = source.peek() != std::char_traits<char>::eof();
    std::size_t position = count;
    if (hasPosition && (!(source >> position) || position > count))
    {
        throw std::runtime_error("FlightControl3D: invalid RNG position");
    }
    source >> std::ws;
    if (!source.eof())
    {
        throw std::runtime_error("FlightControl3D: trailing RNG state");
    }

    // MSVC serializes the 624 preceding recurrence words, while libstdc++
    // serializes a block and the index of its next word. Reverse the in-place
    // twist to recover the preceding words when the block is partly consumed.
    if (hasPosition && position < count)
    {
        state = UntwistedWindow(state, position);
    }

    // Re-encode the window in the native format of this standard library: a
    // probe of the default engine tells whether it writes 624 or 625 numbers.
    std::ostringstream probe;
    probe << std::mt19937{};
    std::istringstream probeWords(probe.str());
    std::uint64_t word{};
    std::size_t nativeCount = 0;
    while (probeWords >> word)
    {
        ++nativeCount;
    }
    if (nativeCount != count && nativeCount != count + 1)
    {
        throw std::runtime_error("FlightControl3D: unsupported native RNG format");
    }
    std::ostringstream converted;
    for (auto const value : state)
    {
        converted << value << ' ';
    }
    if (nativeCount == count + 1)
    {
        converted << count;
    }
    std::istringstream native(converted.str());
    std::mt19937 result;
    if (!(native >> result))
    {
        throw std::runtime_error("FlightControl3D: invalid RNG state");
    }
    return result;
}

} // namespace ppo::examples::flight::detail
