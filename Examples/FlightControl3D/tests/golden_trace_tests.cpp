/**
 * @file Examples/FlightControl3D/tests/golden_trace_tests.cpp
 * @brief Bit-exact fingerprint of a short deterministic training run.
 * @details The PPOKit migration moves GAE, PPO and checkpoint code out of this
 * example. Each move must leave training numerically unchanged, so this test
 * runs JSBSim episodes and three PPO updates from a fixed seed and hashes the
 * complete learner checkpoint (weights, Adam state, RNG) with the update
 * statistics. Libm and compiler differences make the value platform specific:
 * it is enforced where a reference was recorded and reported elsewhere.
 * Run with `--print` to display the fingerprint of the current code.
 */
#include "test_support.hpp"
#include "flight_learner.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

using namespace ppo::examples::flight;

namespace {

/** @brief CTest skip code, registered with SKIP_RETURN_CODE. */
constexpr int SkippedReturnCode = 77;

/** @brief FNV-1a 64-bit accumulator over raw bytes. */
class Fingerprint final
{
public:
    void AddBytes(void const* data, std::size_t size) noexcept
    {
        auto const* bytes = static_cast<unsigned char const*>(data);
        for (std::size_t index = 0; index < size; ++index)
        {
            value_ = (value_ ^ bytes[index]) * 0x100000001b3ULL;
        }
    }
    void AddDouble(double value) noexcept { AddBytes(&value, sizeof(value)); }
    std::uint64_t Value() const noexcept { return value_; }

private:
    std::uint64_t value_ = 0xcbf29ce484222325ULL;
};

/** @brief Small rollouts and short episodes keep the run under a few seconds. */
Config GoldenConfig()
{
    auto config = LoadConfig(PPO_FLIGHT_CONFIG);
    config.training.rolloutSteps = 256;
    config.training.minibatch = 64;
    config.training.epochs = 2;
    config.mission.maxSeconds = 20;
    return config;
}

/** @brief Same collection loop as FlightSession::Tick in training mode. */
std::uint64_t RunGoldenTrace()
{
    constexpr std::uint64_t Updates = 3;
    auto const config = GoldenConfig();
    FlightEnvironment environment(config);
    FlightLearner learner(config);
    Fingerprint fingerprint;
    environment.Reset();
    while (learner.Stats().updates < Updates)
    {
        auto const observation = environment.ObservationNow();
        auto const action = learner.Choose(observation);
        auto const step = environment.Step(ToControls(action, config.initial.throttle));
        learner.Record(observation, action, step, environment.ObservationNow());
        if (step.terminated || step.truncated)
        {
            environment.Reset();
        }
        if (learner.Ready())
        {
            learner.Optimize();
            auto const& stats = learner.Stats();
            fingerprint.AddDouble(stats.policyLoss);
            fingerprint.AddDouble(stats.valueLoss);
            fingerprint.AddDouble(stats.entropy);
            fingerprint.AddDouble(stats.meanReward);
            for (double const reward : stats.meanRewards)
            {
                fingerprint.AddDouble(reward);
            }
        }
    }
    TemporaryDirectory directory("flight-golden-trace-");
    auto const checkpoint = directory.path / "golden.txt";
    learner.Save(checkpoint);
    std::ifstream stream(checkpoint, std::ios::binary);
    std::string const bytes{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    fingerprint.AddBytes(bytes.data(), bytes.size());
    return fingerprint.Value();
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        auto const value = RunGoldenTrace();
        std::printf("[FLIGHT GOLDEN] 0x%016llx\n", static_cast<unsigned long long>(value));
        if (argc > 1 && std::strcmp(argv[1], "--print") == 0)
        {
            return 0;
        }
#if defined(__linux__) && defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
        // Recorded on master before the PPOKit migration (GCC 13, x86-64, Debug and Release).
        constexpr std::uint64_t Reference = 0x0f0414a118b47b70ULL;
        Check(value == Reference, "FlightControl3D training trace changed");
        return 0;
#else
        std::puts("[FLIGHT GOLDEN] no reference recorded for this platform: skipped");
        return SkippedReturnCode;
#endif
    }
    catch (std::exception const& error)
    {
        std::fprintf(stderr, "FAILED: %s\n", error.what());
        return 1;
    }
}
