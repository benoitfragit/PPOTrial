/**
 * @file flight_headless.cpp
 * @brief Command-line runner: unthrottled training or evaluation, telemetry CSV and CI log lines.
 *
 * Role in the chain: last stage of the headless path, JSBSim -> ... -> session -> **logs**.
 * It runs a FlightSession until it finishes and writes:
 * - <output>/training.csv or evaluation.csv: one row per finished episode;
 * - <output>/diagnostics.csv: one row per logged update when diagnostics are enabled;
 * - [FLIGHT], [FLIGHT_VALIDATION], [FLIGHT_GATE] and [FLIGHT_RUNTIME] lines on stdout.
 * The CSV columns and the log lines are parsed by tools/flight/curriculum_ci.py and the
 * CI workflows: their text and order are a compatibility contract.
 *
 * Read-only modes: --inspect-contract prints the network contract and signature as JSON;
 * --check-checkpoint FILE loads a checkpoint against the scenario; --trim prints the JSBSim
 * trim of the nominal approach and of level flight.
 *
 * Read next: flight_session.hpp, tools/flight/curriculum_ci.py,
 * docs/PPOTrial_FlightControl3D_T6_Approach_CI.md.
 */
#include "flight_session.hpp"
#include "flight_trim.hpp"
#include <nlohmann/json.hpp>
#include <charconv>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <optional>
#include <string_view>

namespace {
using namespace ppo::examples::flight;

/** @brief Largest accepted --updates or --episodes value. */
constexpr std::size_t MaximumCountArgument = 100000000;
/** @brief Difficulty of a gate-eligible validation: the full task. */
constexpr double FullTaskDifficulty = 1.0;
/** @brief BEST score printed when there is no BEST yet. */
constexpr double MissingBestScore = -1;

/** @brief Parses a strictly positive decimal integer in [1, 100000000]. */
std::size_t Positive(char const* text)
{
    std::string_view value(text);
    std::size_t result{};
    auto const parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || result == 0
        || result > MaximumCountArgument)
    {
        throw std::invalid_argument("Expected an integer in [1,100000000]");
    }
    return result;
}

template<typename Values>
double Maximum(Values const& values) noexcept
{
    return *std::max_element(values.begin(), values.end());
}

double ActorMeanMaximum(FlightDiagnostics const& diagnostics) noexcept
{
    return *std::max_element(diagnostics.policyMaximumAbsolute.begin(),
        diagnostics.policyMaximumAbsolute.begin() + ContinuousActions);
}

template<typename Values>
double MeanAbsolute(Values const& sums, double samples) noexcept
{
    return std::accumulate(sums.begin(), sums.end(), 0.0) / (samples * static_cast<double>(sums.size()));
}

template<typename Counts>
std::uint64_t Sum(Counts const& counts) noexcept
{
    return std::accumulate(counts.begin(), counts.end(), std::uint64_t{});
}

/** @brief Command-line options. */
struct Options final
{
    std::filesystem::path configPath{PPO_FLIGHT_CONFIG};
    std::filesystem::path output;
    std::filesystem::path checkpointToCheck;
    bool training{true};
    bool inspectContract{};
    bool trim{};
    bool help{};
    std::optional<std::size_t> updates;
    std::optional<std::size_t> episodes;
};

Options ParseOptions(int argc, char** argv)
{
    Options options;
    for (int index = 1; index < argc; ++index)
    {
        std::string_view option(argv[index]);
        if (option == "--evaluate")
        {
            options.training = false;
            continue;
        }
        if (option == "--inspect-contract")
        {
            options.inspectContract = true;
            continue;
        }
        if (option == "--trim")
        {
            options.trim = true;
            continue;
        }
        if (option == "--help")
        {
            options.help = true;
            return options;
        }
        if (option != "--config" && option != "--output" && option != "--updates" && option != "--episodes"
            && option != "--check-checkpoint")
        {
            throw std::invalid_argument("Unknown option");
        }
        if (++index == argc)
        {
            throw std::invalid_argument("Missing option value");
        }
        if (option == "--config")
        {
            options.configPath = argv[index];
        }
        else if (option == "--check-checkpoint")
        {
            options.checkpointToCheck = argv[index];
        }
        else if (option == "--output")
        {
            options.output = argv[index];
        }
        else if (option == "--updates")
        {
            options.updates = Positive(argv[index]);
        }
        else
        {
            options.episodes = Positive(argv[index]);
        }
    }
    return options;
}

/** @brief Read-only contract or checkpoint inspection. */
void Inspect(Options const& options, Config const& config)
{
    if ((options.inspectContract && !options.checkpointToCheck.empty()) || options.updates || options.episodes
        || !options.output.empty() || !options.training)
    {
        throw std::invalid_argument(
            "Checkpoint/contract inspection is read-only and cannot be combined with execution options");
    }
    if (options.inspectContract)
    {
        std::cout << nlohmann::json{{"observations", ObservationSize}, {"actions", ContinuousActions},
            {"outputs", PolicyOutputs}, {"width", config.training.hiddenWidth},
            {"build", PPO_FLIGHT_BUILD_CONFIGURATION},
            {"signature", nlohmann::json::parse(config.checkpointSignature)}}.dump() << '\n';
        return;
    }
    FlightLearner candidate(config);
    candidate.Load(options.checkpointToCheck);
    std::cout << "[FLIGHT_CHECKPOINT] compatible=true updates=" << candidate.Stats().updates << '\n';
}

/** @brief Prints one trim result as a [FLIGHT_TRIM] line. */
void PrintTrimLine(Config const& config, char const* method, double pathDegrees, double speed,
    TrimResult const& trim)
{
    std::cout << "[FLIGHT_TRIM] aircraft=" << config.aircraft << " method=" << method << " path_deg="
        << pathDegrees << " speed_mps=" << speed << " elevator=" << trim.elevator << " throttle="
        << trim.throttle << " pitch_deg=" << trim.pitch / DegreesToRadians << " converged="
        << std::boolalpha << trim.converged << '\n';
}

/**
 * @brief Prints the trim of the nominal 3-degree approach and of level flight, with JSBSim's
 *        FGTrim and by simulated flight.
 *
 * The approach trim is the action reference of the scenario ("elevatorTrim" and
 * "throttle"): use it when configuring a new aircraft. FGTrim cannot trim a turboprop such
 * as the PC-7; the simulated flight can.
 * @param config Validated scenario.
 */
void PrintTrim(Config const& config)
{
    constexpr double TrimHeightMeters = 300.0;
    JsbsimAircraft aircraft(config, DefaultDataRoot());
    TrimCondition condition{};
    condition.calibratedSpeed = config.mission.approachSpeed;
    condition.flap = config.initial.approachFlap;
    condition.heightMeters = TrimHeightMeters;
    for (double const pathDegrees : {-config.mission.glideDegrees, 0.0})
    {
        condition.flightPathRadians = pathDegrees * DegreesToRadians;
        PrintTrimLine(config, "jsbsim", pathDegrees, condition.calibratedSpeed, aircraft.Trim(condition));
        PrintTrimLine(config, "flight", pathDegrees, condition.calibratedSpeed, FlyTrim(aircraft, config,
            condition));
    }
}

void WriteTelemetryHeader(std::ofstream& telemetry)
{
    telemetry << "episode,updates,outcome,simulated_seconds,touchdown_m,stop_m,ground_roll_m,"
        "touchdowns,stops,crashes,timeouts,first_wheel_contact,valid_touchdown,touchdown_cross_m,"
        "touchdown_sink_mps,touchdown_speed_mps,gear_position,touchdown_quality,curriculum_level\n";
}

void WriteDiagnosticsHeader(std::ofstream& diagnostics)
{
    diagnostics << "updates,samples,observation_saturation_ratio,observation_max_abs,actor_mean_max_abs,"
        "raw_action_mean_abs,applied_action_mean_abs,raw_action_saturation_ratio,"
        "applied_action_saturation_ratio,reward_mean,reward_min,reward_max,actor_steps,"
        "maximum_minibatch_approx_kl,maximum_minibatch_sampled_kl,critic_steps,actor_kl_skips\n";
}

/** @brief One telemetry row per finished episode; empty touchdown/stop cells when they did not happen. */
void WriteEpisodeRow(std::ofstream& telemetry, FlightSnapshot const& state)
{
    auto const& mission = state.lastEpisode;
    telemetry << state.episodes << ',' << state.learning.updates << ','
        << static_cast<int>(mission.outcome) << ',' << state.lastEpisodeSeconds << ',';
    if (mission.contactAttempted)
    {
        telemetry << mission.touchdownFromThreshold;
    }
    telemetry << ',';
    if (mission.outcome == Outcome::Stopped)
    {
        telemetry << mission.stopFromThreshold;
    }
    double touchdownQuality = 0.0;
    if (mission.touchedDown)
    {
        touchdownQuality = mission.touchdownQuality;
    }
    telemetry << ',' << mission.groundRoll << ',' << state.touchdowns << ',' << state.stops << ','
        << state.crashes
        << ',' << state.timeouts
        << ',' << mission.firstWheelContact << ',' << mission.touchedDown << ','
            << mission.touchdownCrossMeters
        << ',' << mission.touchdownSinkMps
        << ',' << mission.touchdownSpeedMps << ',' << mission.touchdownGearPosition
        << ',' << touchdownQuality << ',' << state.curriculumLevel << '\n';
    telemetry.flush();
}

/** @brief One diagnostics row: saturation ratios, magnitudes and KL guard counters of the last window. */
void WriteDiagnosticsRow(std::ofstream& diagnostics, FlightSnapshot const& state)
{
    auto const& window = state.lastDiagnostics;
    double const samples = static_cast<double>(std::max<std::uint64_t>(window.samples, 1));
    diagnostics << state.learning.updates << ',' << window.samples << ','
        << static_cast<double>(Sum(window.observationSaturations)) / (samples * ObservationSize) << ','
        << Maximum(window.observationMaximumAbsolute) << ',' << ActorMeanMaximum(window) << ','
        << MeanAbsolute(window.rawAbsoluteSums, samples) << ','
        << MeanAbsolute(window.appliedAbsoluteSums, samples) << ','
        << static_cast<double>(Sum(window.rawSaturations)) / (samples * ContinuousActions) << ','
        << static_cast<double>(Sum(window.appliedSaturations)) / (samples * ContinuousActions) << ','
        << window.rewardSum / samples << ',' << window.rewardMinimum << ',' << window.rewardMaximum << ','
        << state.learning.actorSteps << ','
        << state.learning.maximumApproximateKl << ','
        << state.learning.maximumSampledKl << ',' << state.learning.criticSteps << ','
        << state.learning.actorKlSkips << '\n';
    diagnostics.flush();
}

void LogValidation(FlightSnapshot const& state)
{
    double success = 0;
    if (state.gateEvaluated > 0)
    {
        success = static_cast<double>(state.gateSuccesses) / static_cast<double>(state.gateEvaluated);
    }
    double best = MissingBestScore;
    if (state.hasBest)
    {
        best = state.bestScore;
    }
    std::cout << "[FLIGHT_VALIDATION] update=" << state.lastValidation.update
        << " episodes=" << state.lastValidation.episodes
        << " difficulty=" << state.lastValidation.difficulty << " success=" << success << " best=" << best
        << std::endl;
}

/**
 * @brief Prints the final gate verdict of a training run.
 *
 * Validation follows the curriculum; only a campaign on the full task can pass the gate,
 * however easy levels score. The gate needs ceil(minimumSuccessRate x episodes) stops.
 */
void LogGate(Config const& config, FlightSnapshot const& state)
{
    double const difficulty = state.lastValidation.difficulty;
    auto const requiredStops = static_cast<std::size_t>(
        std::ceil(config.evaluation.minimumSuccessRate * static_cast<double>(state.gateEvaluated)));
    bool const passed = difficulty >= FullTaskDifficulty && state.gateSuccesses >= requiredStops;
    std::cout << "[FLIGHT_GATE] successes=" << state.gateSuccesses << " episodes=" << state.gateEvaluated
        << " difficulty=" << difficulty << " passed=" << (passed ? "true" : "false") << std::endl;
}

/** @brief Runs the session to completion while writing telemetry and progress logs. */
void Run(Options const& options, Config const& config)
{
    std::cout << "[FLIGHT_RUNTIME] build=" << PPO_FLIGHT_BUILD_CONFIGURATION << " aircraft="
        << config.aircraft
        << " mode=headless autorun=true unthrottled=true config=" << options.configPath.string() << std::endl;
    auto output = options.output;
    if (output.empty())
    {
        // Default: <repository>/<checkpointDirectory>, the scenario living in resources/settings/Scenarios.
        output = std::filesystem::absolute(options.configPath).parent_path().parent_path().parent_path()
            / config.training.checkpointDirectory;
    }
    FlightSession session(config, options.training, output, options.episodes, TrainingBudget::TotalUpdates);
    std::filesystem::create_directories(output);
    std::ofstream telemetry(output / (options.training ? "training.csv" : "evaluation.csv"));
    if (!telemetry)
    {
        throw std::runtime_error("Cannot write FlightControl3D telemetry");
    }
    WriteTelemetryHeader(telemetry);
    std::ofstream diagnostics;
    if (config.diagnostics.enabled)
    {
        diagnostics.open(output / "diagnostics.csv");
        if (!diagnostics)
        {
            throw std::runtime_error("Cannot write FlightControl3D diagnostics");
        }
        WriteDiagnosticsHeader(diagnostics);
    }
    std::uint64_t previousEpisodes = 0;
    std::uint64_t previousUpdates = session.Snapshot().learning.updates;
    std::uint64_t previousValidations = 0;
    while (!session.Finished())
    {
        session.Tick();
        auto const state = session.Snapshot();
        if (state.episodes != previousEpisodes)
        {
            WriteEpisodeRow(telemetry, state);
            previousEpisodes = state.episodes;
        }
        if (state.learning.updates != previousUpdates)
        {
            std::cout << "[FLIGHT] updates=" << state.learning.updates << " episodes=" << state.episodes
                << " touchdowns=" << state.touchdowns << " stops=" << state.stops
                << " loss=" << state.learning.policyLoss << std::endl;
            if (diagnostics && state.learning.updates % config.diagnostics.logEveryUpdates == 0)
            {
                WriteDiagnosticsRow(diagnostics, state);
            }
            previousUpdates = state.learning.updates;
        }
        if (state.validationCampaigns != previousValidations)
        {
            LogValidation(state);
            previousValidations = state.validationCampaigns;
        }
    }
    auto const state = session.Snapshot();
    if (options.training && state.gateEvaluated > 0)
    {
        LogGate(config, state);
    }
    std::cout << "[FLIGHT] completed episodes=" << state.episodes << " touchdowns=" << state.touchdowns
        << " stops=" << state.stops << " crashes=" << state.crashes << " timeouts=" << state.timeouts << '\n';
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        auto const options = ParseOptions(argc, argv);
        if (options.help)
        {
            std::cout << "FlightControlHeadless [--config FILE] [--output DIRECTORY] [--updates N] "
                << "[--evaluate --episodes N] [--inspect-contract | --check-checkpoint FILE | --trim]\n";
            return 0;
        }
        auto config = LoadConfig(options.configPath);
        if (options.updates)
        {
            config.training.maxUpdates = *options.updates;
        }
        Validate(config);
        if ((options.training && options.episodes) || (!options.training && options.updates))
        {
            throw std::invalid_argument("--updates is for training; --episodes is for evaluation");
        }
        if (options.trim)
        {
            PrintTrim(config);
            return 0;
        }
        if (options.inspectContract || !options.checkpointToCheck.empty())
        {
            Inspect(options, config);
            return 0;
        }
        Run(options, config);
    }
    catch (std::exception const& error)
    {
        std::cerr << "[FLIGHT_ERROR] " << error.what() << '\n';
        return 1;
    }
}
