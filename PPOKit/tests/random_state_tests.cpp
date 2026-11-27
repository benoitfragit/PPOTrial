/**
 * @file PPOKit/tests/random_state_tests.cpp
 * @brief Portable random state: the standard text of std::mt19937 (its last 624 values) is
 *        what the C++ standard defines, and an engine read back from it, or from the
 *        libstdc++ text, continues with exactly the same values, at every position of a block.
 */
#include "ppokit/artifacts/random_state.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

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

/** @brief Inverse of the Mersenne Twister tempering: the state value behind an output. */
std::uint32_t Untemper(std::uint32_t value) noexcept
{
    value ^= value >> 18U;
    value ^= (value << 15U) & 0xefc60000U;
    std::uint32_t result = value;
    for (int round = 0; round < 5; ++round)
    {
        result = value ^ ((result << 7U) & 0x9d2c5680U);
    }
    value = result;
    result = value;
    for (int round = 0; round < 3; ++round)
    {
        result = value ^ (result >> 11U);
    }
    return result;
}

std::vector<std::uint32_t> Words(std::string const& text)
{
    std::istringstream stream(text);
    std::vector<std::uint32_t> words;
    unsigned long long value = 0;
    while (stream >> value)
    {
        words.push_back(static_cast<std::uint32_t>(value));
    }
    return words;
}

/** @brief The next `count` outputs of an engine (a copy: the engine is unchanged). */
std::vector<std::uint32_t> Next(std::mt19937 engine, std::size_t count)
{
    std::vector<std::uint32_t> values;
    for (std::size_t index = 0; index < count; ++index)
    {
        values.push_back(static_cast<std::uint32_t>(engine()));
    }
    return values;
}

/** @brief After `draws` outputs: standard text by definition, and an exact continuation from both forms. */
void CheckAfter(std::size_t draws)
{
    std::mt19937 engine(20261124U);
    std::vector<std::uint32_t> outputs;
    for (std::size_t index = 0; index < draws; ++index)
    {
        outputs.push_back(static_cast<std::uint32_t>(engine()));
    }
    std::string const standard = ppokit::artifacts::StandardRandomText(engine);
    std::vector<std::uint32_t> const words = Words(standard);
    std::string const label = "after " + std::to_string(draws) + " draws";
    Check(words.size() == 624U, label + ": the standard text holds 624 values");
    if (draws >= 624U)
    {
        bool definition = words.size() == 624U;
        for (std::size_t index = 0; definition && index < 624U; ++index)
        {
            definition = words[index] == Untemper(outputs[draws - 624U + index]);
        }
        Check(definition, label + ": the standard text is the last 624 values (C++ standard)");
    }
    std::vector<std::uint32_t> const expected = Next(engine, 2000);
    Check(Next(ppokit::artifacts::ParseRandomText(standard, "test"), 2000) == expected,
        label + ": an engine read from the standard text continues exactly");
    std::ostringstream local;
    local << engine;
    Check(Next(ppokit::artifacts::ParseRandomText(local.str(), "test"), 2000) == expected,
        label + ": an engine read from this library's own text continues exactly");
}

void DamagedStatesAreRefused()
{
    bool refused = false;
    try
    {
        (void)ppokit::artifacts::ParseRandomText("1 2 3", "latest.ppokit");
    }
    catch (std::runtime_error const& error)
    {
        refused = std::string(error.what()).find("latest.ppokit") != std::string::npos;
    }
    Check(refused, "a state of 3 numbers is refused, naming the file");

    std::string const valid = ppokit::artifacts::StandardRandomText(std::mt19937{7});
    auto const firstSeparator = valid.find(' ');
    for (auto const& damaged : {valid + " garbage", std::string("4294967296") + valid.substr(firstSeparator),
             std::string("-1") + valid.substr(firstSeparator)})
    {
        refused = false;
        try { (void)ppokit::artifacts::ParseRandomText(damaged, "latest.ppokit"); }
        catch (std::runtime_error const& error)
        {
            refused = std::string(error.what()).find("latest.ppokit") != std::string::npos;
        }
        Check(refused, "invalid RNG word or trailing token is refused, naming the file");
    }
}

} // namespace

int main()
{
    for (std::size_t draws : {0U, 1U, 2U, 396U, 397U, 623U, 624U, 625U, 1000U, 1247U, 1248U, 1249U, 5000U})
    {
        CheckAfter(draws);
    }
    DamagedStatesAreRefused();
    if (failures != 0)
    {
        std::cerr << failures << " random state check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit random state tests passed\n";
    return EXIT_SUCCESS;
}
