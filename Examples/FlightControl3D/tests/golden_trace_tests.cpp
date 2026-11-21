/**
 * @file golden_trace_tests.cpp
 * @brief Bit-exact behavioural fingerprint of the whole FlightControl3D chain.
 *
 * This test is the safety net of readability refactorings: it plays seeded
 * episodes through JSBSim -> mission -> rewards -> observations -> PPO actor,
 * runs two PPO updates on a small contract, saves the checkpoint, and hashes
 * every produced value (observations, the seven reward channels, episode
 * boundaries, outcomes, actions, learner statistics, the contract signature
 * and the checkpoint bytes) with FNV-1a 64.
 *
 * The reference fingerprint was frozen on the code *before* the readability
 * refactoring. It must never be updated to make a refactoring pass: a mismatch
 * means the refactoring changed the model, the physics coupling, a random draw
 * order or a floating-point operation order.
 *
 * Floating-point results of JSBSim, libm and the RNG stream representation are
 * platform specific, so the reference is only enforced on the platform where it
 * was frozen (Linux x86_64, GCC 13, libstdc++). Elsewhere the test prints its
 * fingerprint and reports a CTest skip (return code 77).
 *
 * Related files: tests/session_tests.cpp (session semantics),
 * tests/learner_tests.cpp (PPO derivatives), docs/FlightControl3D_Architecture.md.
 */
#include "test_support.hpp"
#include "flight_learner.hpp"
#include <iomanip>
#include <iterator>
#include <sstream>
#include <type_traits>

using namespace ppo::examples::flight;

namespace {

/** @brief CTest return code declared as SKIP_RETURN_CODE for unsupported platforms. */
constexpr int SkipReturnCode = 77;

/** @brief Upper bound of actions per episode; the shortened maxSeconds ends episodes far earlier. */
constexpr std::size_t MaximumActionsPerEpisode = 100000;

/** @brief Incremental FNV-1a 64-bit hash of raw value representations. */
class Fingerprint final
{
public:
    void Bytes(void const* data, std::size_t size) noexcept
    {
        auto const* bytes = static_cast<unsigned char const*>(data);
        for (std::size_t index = 0; index < size; ++index)
        {
            hash_ ^= bytes[index];
            hash_ *= Prime;
        }
    }

    template<typename Value>
    void Add(Value const& value) noexcept
    {
        static_assert(std::is_trivially_copyable_v<Value>, "hash only plain values");
        Bytes(&value, sizeof(value));
    }

    void Text(std::string const& text) noexcept
    {
        std::uint64_t const size = text.size();
        Add(size);
        Bytes(text.data(), text.size());
    }

    std::uint64_t Value() const noexcept { return hash_; }

private:
    static constexpr std::uint64_t OffsetBasis = 14695981039346656037ULL;
    static constexpr std::uint64_t Prime = 1099511628211ULL;
    std::uint64_t hash_{OffsetBasis};
};

/** @brief Small, fast training contract derived from the production scenario. */
Config GoldenConfig()
{
    auto config = LoadConfig(PPO_FLIGHT_CONFIG);
    config.training.rolloutSteps = 64;
    config.training.minibatch = 16;
    config.training.epochs = 2;
    config.training.hiddenWidth = 32;
    config.training.randomizeEnvironmentSeed = false;
    // Short episodes keep the test fast while still exercising touchdown,
    // crash, stop and the partial-action timeout correction.
    config.mission.maxSeconds = 120.0;
    Validate(config);
    return config;
}

/** @brief Side drawn by a seeded airborne reset, replaying FlightEnvironment::Reset(seed) draws. */
ApproachSide SeededSide(Config const& config, std::uint32_t seed)
{
    std::mt19937 rng(seed);
    SampleRunway(config, rng);
    return std::bernoulli_distribution(0.5)(rng) ? ApproachSide::Right : ApproachSide::Left;
}

/** @brief First seed from a base that draws the requested approach side. */
std::uint32_t SeedForSide(Config const& config, std::uint32_t base, ApproachSide side)
{
    for (std::uint32_t seed = base; seed < base + 1000U; ++seed)
        if (SeededSide(config, seed) == side) return seed;
    throw std::runtime_error("no seed draws the requested approach side");
}

void HashMission(Fingerprint& fingerprint, MissionState const& mission)
{
    fingerprint.Add(mission.phase);
    fingerprint.Add(mission.outcome);
    fingerprint.Add(mission.steps);
    fingerprint.Add(mission.stoppedSeconds);
    fingerprint.Add(mission.touchedDown);
    fingerprint.Add(mission.contactAttempted);
    fingerprint.Add(mission.firstWheelContact);
    fingerprint.Add(mission.touchdownCrossMeters);
    fingerprint.Add(mission.touchdownSinkMps);
    fingerprint.Add(mission.touchdownSpeedMps);
    fingerprint.Add(mission.touchdownGearPosition);
    fingerprint.Add(mission.touchdownQuality);
    fingerprint.Add(mission.landingAreaFactor);
    fingerprint.Add(mission.touchdownFromThreshold);
    fingerprint.Add(mission.stopFromThreshold);
    fingerprint.Add(mission.groundRoll);
}

/** @brief Hashes crash conditions field by field, so struct padding never enters the fingerprint. */
void HashCrash(Fingerprint& fingerprint, CrashDiagnostic const& diagnostic)
{
    for (auto const& condition : diagnostic.conditions)
    {
        fingerprint.Add(condition.failed);
        fingerprint.Add(condition.value);
        fingerprint.Add(condition.limit);
    }
}

void HashStep(Fingerprint& fingerprint, Observation const& observation, PolicyAction const& action,
    Controls const& controls, StepResult const& step)
{
    fingerprint.Add(observation);
    fingerprint.Add(action.raw);
    fingerprint.Add(action.logProbability);
    fingerprint.Add(action.brakeActive);
    fingerprint.Add(action.policy.output);
    fingerprint.Add(action.policy.logStandardDeviations);
    fingerprint.Add(controls.elevator);
    fingerprint.Add(controls.aileron);
    fingerprint.Add(controls.rudder);
    fingerprint.Add(controls.throttle);
    fingerprint.Add(controls.brake);
    fingerprint.Add(step.rewards);
    fingerprint.Add(step.reward);
    fingerprint.Add(step.terminated);
    fingerprint.Add(step.truncated);
}

/** @brief Plays one seeded episode to its boundary and hashes every action. */
void PlaySeededEpisode(Fingerprint& fingerprint, FlightEnvironment& environment, FlightLearner& learner,
    Config const& config, std::uint32_t seed, EpisodeStart start, bool deterministic)
{
    auto observation = environment.Reset(seed, start);
    fingerprint.Add(seed);
    fingerprint.Add(observation);
    std::size_t actions = 0;
    for (std::size_t action = 0; action < MaximumActionsPerEpisode; ++action)
    {
        ++actions;
        auto const chosen = learner.Choose(observation, deterministic);
        auto const controls = ToControls(chosen, config.initial);
        auto const step = environment.Step(controls);
        HashStep(fingerprint, observation, chosen, controls, step);
        observation = environment.ObservationNow();
        if (step.terminated || step.truncated) break;
    }
    fingerprint.Add(observation);
    HashMission(fingerprint, environment.Mission());
    HashCrash(fingerprint, environment.CrashDetails());
    std::cout << "[FLIGHT_GOLDEN_EPISODE] seed=" << seed << " start=" << (start.rolling ? "rolling"
        : "airborne")
        << " difficulty=" << start.difficulty << " deterministic=" << deterministic << " actions=" << actions
        << " outcome=" << static_cast<int>(environment.Mission().outcome)
        << " touched_down=" << environment.Mission().touchedDown << '\n';
}

/**
 * @brief Plays one seeded episode with a fixed raw action, to reach outcomes the initial policy misses.
 * @param raw Raw actor action held for the whole episode.
 */
void PlayScriptedEpisode(Fingerprint& fingerprint, FlightEnvironment& environment, Config const& config,
    std::uint32_t seed, EpisodeStart start, std::array<double, ContinuousActions> const& raw)
{
    auto observation = environment.Reset(seed, start);
    fingerprint.Add(seed);
    fingerprint.Add(observation);
    std::size_t actions = 0;
    for (std::size_t action = 0; action < MaximumActionsPerEpisode; ++action)
    {
        ++actions;
        PolicyAction scripted{};
        scripted.raw = raw;
        scripted.brakeActive = observation[RolloutObservation] > 0.5F;
        auto const controls = ToControls(scripted, config.initial);
        auto const step = environment.Step(controls);
        HashStep(fingerprint, observation, scripted, controls, step);
        observation = environment.ObservationNow();
        if (step.terminated || step.truncated) break;
    }
    fingerprint.Add(observation);
    HashMission(fingerprint, environment.Mission());
    std::cout << "[FLIGHT_GOLDEN_SCRIPTED] seed=" << seed << " actions=" << actions
        << " outcome=" << static_cast<int>(environment.Mission().outcome) << '\n';
}

void HashStats(Fingerprint& fingerprint, LearningStats const& stats)
{
    fingerprint.Add(stats.updates);
    fingerprint.Add(stats.policyLoss);
    fingerprint.Add(stats.valueLoss);
    fingerprint.Add(stats.entropy);
    fingerprint.Add(stats.meanReward);
    fingerprint.Add(stats.maximumApproximateKl);
    fingerprint.Add(stats.maximumSampledKl);
    fingerprint.Add(stats.actorSteps);
    fingerprint.Add(stats.criticSteps);
    fingerprint.Add(stats.actorKlSkips);
    fingerprint.Add(stats.meanRewards);
    fingerprint.Add(stats.hasRewardMeans);
}

/** @brief Collects training-stream rollouts and runs the requested PPO updates. */
void TrainUpdates(Fingerprint& fingerprint, FlightLearner& learner, Config const& config, std::size_t updates)
{
    FlightEnvironment environment(config, DefaultDataRoot(), config.training.seed);
    environment.SetDifficulty(0.4);
    auto observation = environment.Reset();
    fingerprint.Add(observation);
    while (learner.Stats().updates < updates)
    {
        auto const chosen = learner.Choose(observation, false);
        auto const controls = ToControls(chosen, config.initial);
        auto const step = environment.Step(controls);
        auto const next = environment.ObservationNow();
        HashStep(fingerprint, observation, chosen, controls, step);
        learner.Record(observation, chosen, step, next);
        observation = next;
        if (step.terminated || step.truncated)
        {
            HashMission(fingerprint, environment.Mission());
            fingerprint.Add(environment.StartedAirborne());
            observation = environment.Reset();
            fingerprint.Add(observation);
        }
        if (learner.Ready())
        {
            learner.Optimize();
            HashStats(fingerprint, learner.Stats());
        }
    }
}

std::string ReadBytes(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

/** @brief Computes the complete behavioural fingerprint of the current build. */
std::uint64_t ComputeFingerprint()
{
    auto const config = GoldenConfig();
    Fingerprint fingerprint;
    fingerprint.Text(config.checkpointSignature);
    FlightLearner learner(config);
    FlightEnvironment environment(config, DefaultDataRoot(), config.evaluation.seed);

    // Seeded episodes: two rolling starts, then approaches at difficulties
    // 0, 0.4 and 1 on both sides, each flown by the deterministic and by the
    // stochastic initial policy.
    for (bool const deterministic : {true, false})
    {
        std::uint32_t const rollingSeeds[] = {1001U, 1002U};
        for (auto const seed : rollingSeeds)
            PlaySeededEpisode(fingerprint, environment, learner, config, seed, {true, 0.0}, deterministic);
        std::uint32_t base = 2000U;
        for (double const difficulty : {0.0, 0.4, 1.0})
        {
            for (auto const side : {ApproachSide::Left, ApproachSide::Right})
            {
                auto const seed = SeedForSide(config, base, side);
                PlaySeededEpisode(fingerprint, environment, learner, config, seed, {false, difficulty},
                    deterministic);
                base = seed + 1U;
            }
        }
    }

    // Scripted episodes: a full-brake rolling start reaches the Stop outcome,
    // and a time limit that is not a multiple of the action period truncates
    // an action midway, which exercises the partial-timeout potential correction.
    constexpr std::array<double, ContinuousActions> FullBrakeIdle{0.0, 0.0, 0.0, -1.0, 1.0};
    constexpr std::array<double, ContinuousActions> TrimmedApproach{};
    for (std::uint32_t const seed : {3001U, 3002U})
        PlayScriptedEpisode(fingerprint, environment, config, seed, {true, 0.0}, FullBrakeIdle);
    auto shortConfig = config;
    shortConfig.mission.maxSeconds = 3.05;
    FlightEnvironment shortEnvironment(shortConfig, DefaultDataRoot(), config.evaluation.seed);
    PlayScriptedEpisode(fingerprint, shortEnvironment, shortConfig, 3003U, {false, 0.0}, TrimmedApproach);

    // Two PPO updates on the training stream, then a checkpoint round trip.
    TrainUpdates(fingerprint, learner, config, 2);
    TemporaryDirectory directory("flight-golden-");
    auto const checkpoint = directory.path / "golden_policy.txt";
    learner.Save(checkpoint);
    fingerprint.Text(ReadBytes(checkpoint));

    FlightLearner restored(config);
    restored.Load(checkpoint);
    HashStats(fingerprint, restored.Stats());
    for (auto const& probe : {Observation{}, environment.ObservationNow()})
    {
        auto const policy = restored.Evaluate(probe);
        Check(policy == learner.Evaluate(probe), "a reloaded checkpoint reproduces the actor");
        fingerprint.Add(policy.output);
        fingerprint.Add(policy.logStandardDeviations);
    }
    return fingerprint.Value();
}

std::string Hex(std::uint64_t value)
{
    std::ostringstream text;
    text << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
    return text.str();
}

} // namespace

int main()
{
    try
    {
        std::uint64_t const fingerprint = ComputeFingerprint();
        std::cout << "[FLIGHT_GOLDEN] fingerprint=" << Hex(fingerprint) << '\n';
#if defined(__linux__) && defined(__x86_64__) && defined(__GLIBCXX__) && defined(__GNUC__) \
    && !defined(__clang__) && __GNUC__ == 13
        // PPOKit step 3 (user decision B): one shared Gaussian/PPO formula, which moves
        // log-probabilities and weights by about one ulp. Episodes, actions and outcomes are
        // unchanged. Previous reference, master after #87: 0x471aaf859adb0883.
        // Identical in Debug and Release.
        constexpr std::uint64_t Reference = 0x67f683c02a76946cULL;
        Check(fingerprint == Reference, "golden trace changed: the refactoring altered model behaviour");
        std::cout << "Flight golden trace matches the pre-refactoring reference\n";
        return 0;
#else
        std::cout << "Flight golden trace reference is not frozen for this platform\n";
        return SkipReturnCode;
#endif
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
