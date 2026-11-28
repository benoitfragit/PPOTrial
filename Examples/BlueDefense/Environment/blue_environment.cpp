/**
 * @file Examples/BlueDefense/Environment/blue_environment.cpp
 * @brief The BlueDefense world as an environment DLL for ppokit-run: P blues against N
 *        RLMission reds, the blue team driven from outside the world.
 * @details The DLL holds the ECS world of the BlueDefense example, loaded from the project's
 * scenario.json, with the blue team's controller driven from outside: the simulation, red
 * controllers, rewards and outcome in the world, the learner in the network DLL. The world is
 * stepped from ppokit-run's thread in two halves (Environment::BeginStep / EndStep):
 *
 * - reset / begin_step: the world begins its step (rules before the step, the
 *   re-initialization after an ended episode); the blue controller reads the team's state
 *   and publishes the observations the blues act on;
 * - step: the blues' actions are handed to the controller; the world ends its step (every
 *   agent chooses, entities advance, rules resolve, agents read the new state); the
 *   controller publishes each blue's reward and channels, the next observations and the
 *   outcome.
 *
 * Rows of agents are the blue slots blue_1..blue_9; the first P are the episode's blues.
 * The episodes of training follow the scenario's own seed (blue_patrol_training.json); an
 * evaluation episode (reset_evaluation) is the case of its seed and its place in the suite
 * (patrol side, racetrack phase, GeneralizedEvaluationCaseOf), with the evaluation step
 * limit. Its metrics are the episode's results (score included).
 *
 * The world (task.json "world", for the observers of the Visualiseur): one row per blue slot,
 * that blue (present, alive, leader, position, heading, speed, its patrol reference), then
 * the scene every row repeats so that a row stands alone: the reds (north, east, heading,
 * alive, present), the objective and its defended area, the racetrack's waypoints and the
 * active one (-1 when the team does not follow them).
 */
#include "bluedefenseexample.hpp"
#include "generalized_evaluation_case.hpp"
#include "generalized_patrol_team_agent.hpp"

#include "airspacemeta.hpp"
#include "environment.hpp"

#include "ppokit/environment/sdk.hpp"
#include "ppokit/schema/project.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <nlohmann/json.hpp>

namespace {

namespace defenders = ppo::examples::bluedefense::defenders;

constexpr std::size_t Capacity = defenders::TeamCapacity::MaximumBlueMobiles;
constexpr std::size_t ObservationSize = defenders::GeneralizedObservationLayout::Size;
constexpr std::size_t ActionSize = defenders::GeneralizedObservationLayout::ActionSize;
constexpr std::size_t Channels = defenders::kGeneralizedRewardChannelCount;
constexpr std::size_t Reds = defenders::TeamCapacity::MaximumRedMobiles;
constexpr std::size_t Waypoints = defenders::GeneralizedPatrolWaypointSequencer::WaypointCount;

/** @brief One row of task.json "world", in its order: values per item, offsets of the groups. */
namespace world_row {
constexpr std::size_t BlueValues = 9;
constexpr std::size_t RedValues = 5;
constexpr std::size_t ObjectiveValues = 7;
constexpr std::size_t RedsAt = BlueValues;
constexpr std::size_t ObjectiveAt = RedsAt + Reds * RedValues;
constexpr std::size_t WaypointsAt = ObjectiveAt + ObjectiveValues;
/** @brief Per blue: role, cone radius and half angle, projected reference, guidance heading, shown. */
constexpr std::size_t BlueViewValues = 7;
constexpr std::size_t BlueViewAt = WaypointsAt + Waypoints * 2 + 1;
/** @brief Per red: cone radius and half angle. */
constexpr std::size_t RedConeValues = 2;
constexpr std::size_t RedConesAt = BlueViewAt + BlueViewValues;
/** @brief The blue's lidar: ranges, distances, span (0: none). */
constexpr std::size_t LidarBins = defenders::SceneLidar::Bins;
constexpr std::size_t LidarAt = RedConesAt + Reds * RedConeValues;
constexpr std::size_t LidarValues = 2 * LidarBins + 1;
/** @brief Per zone: north, east, radius, hostile; then how many zones the world has. */
constexpr std::size_t Zones = defenders::GeneralizedTeamScene::ZoneCapacity;
constexpr std::size_t ZoneValues = 4;
constexpr std::size_t ZonesAt = LidarAt + LidarValues;
/** @brief The outer failure rectangle's half extents. */
constexpr std::size_t BoundsAt = ZonesAt + Zones * ZoneValues + 1;
/** @brief Track samples (north, east), their count; waypoint headings; validated waypoints. */
constexpr std::size_t TrackSamples = ppo::model::sPatrolTrackOverlay::SampleCapacity;
constexpr std::size_t PatrolAt = BoundsAt + 2;
constexpr std::size_t PatrolValues = 2 * TrackSamples + 1 + Waypoints + 1;
constexpr std::size_t Size = PatrolAt + PatrolValues;
} // namespace world_row

// task.json "world" declares these sizes.
static_assert(Reds * world_row::RedConeValues == 20, "task.json world.red_cones");
static_assert(world_row::LidarValues == 123, "task.json world.lidar");
static_assert(world_row::Zones * world_row::ZoneValues + 1 == 65, "task.json world.zones");
static_assert(world_row::PatrolValues == 222, "task.json world.patrol");
static_assert(world_row::Size == 558, "task.json world");

/** @brief The scenario of a BluePatrol project: its settings file (project.json "settings"). */
std::filesystem::path ScenarioOf(std::filesystem::path const& project)
{
    std::optional<std::filesystem::path> const settings = ppokit::schema::LoadProject(project).SettingsFile();
    if (!settings)
    {
        throw std::runtime_error(project.string() + ": project.json names no scenario (\"settings\")");
    }
    // Absolute: the scenario's loader resolves relative paths against the shared settings.
    return std::filesystem::absolute(*settings);
}

/** @brief Own a private folder for validation inputs without touching the project or candidate. */
class ScenarioValidationFolder final
{
public:
    ScenarioValidationFolder()
    {
        std::random_device random;
        for (int attempt = 0; attempt < 16; ++attempt)
        {
            auto const candidate = std::filesystem::temp_directory_path()
                / ("blue-scenario-" + std::to_string(random()));
            if (std::filesystem::create_directory(candidate))
            {
                path_ = candidate;
                return;
            }
        }
        throw std::runtime_error("Cannot allocate a Blue scenario validation folder");
    }
    ~ScenarioValidationFolder()
    {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }
    ScenarioValidationFolder(ScenarioValidationFolder const&) = delete;
    ScenarioValidationFolder& operator=(ScenarioValidationFolder const&) = delete;
    std::filesystem::path const& Path() const noexcept { return path_; }
private:
    std::filesystem::path path_;
};

/** @brief The scene every row repeats: reds, objective and defended area, waypoints. */
void WriteSceneRow(defenders::GeneralizedTeamScene const& scene, float* row) noexcept
{
    for (std::size_t red = 0; red < Reds; ++red)
    {
        float* const values = row + world_row::RedsAt + red * world_row::RedValues;
        defenders::Mobile const& mobile = scene.state.reds[red];
        bool const present = red < scene.state.counts.redMobiles;
        values[0] = mobile.north;
        values[1] = mobile.east;
        values[2] = mobile.heading;
        values[3] = present && mobile.available ? 1.F : 0.F;
        values[4] = present ? 1.F : 0.F;
    }
    float* const objective = row + world_row::ObjectiveAt;
    objective[0] = scene.state.objectiveNorth;
    objective[1] = scene.state.objectiveEast;
    objective[2] = scene.state.objectiveAvailable ? 1.F : 0.F;
    objective[3] = scene.defendedArea.m_north;
    objective[4] = scene.defendedArea.m_east;
    objective[5] = scene.defendedArea.m_halfNorth;
    objective[6] = scene.defendedArea.m_halfEast;
    float* const waypoints = row + world_row::WaypointsAt;
    for (std::size_t index = 0; index < Waypoints; ++index)
    {
        waypoints[2 * index] = scene.waypoints[index].north;
        waypoints[2 * index + 1] = scene.waypoints[index].east;
    }
    waypoints[2 * Waypoints] = scene.followingWaypoints ? static_cast<float>(scene.activeWaypoint) : -1.F;
    for (std::size_t red = 0; red < Reds; ++red)
    {
        float* const cone = row + world_row::RedConesAt + red * world_row::RedConeValues;
        cone[0] = scene.redCones[red].radius;
        cone[1] = scene.redCones[red].halfAngle;
    }
    float* const zones = row + world_row::ZonesAt;
    for (std::size_t zone = 0; zone < std::min(scene.zoneCount, world_row::Zones); ++zone)
    {
        float* const values = zones + zone * world_row::ZoneValues;
        values[0] = scene.zones[zone].north;
        values[1] = scene.zones[zone].east;
        values[2] = scene.zones[zone].radius;
        values[3] = scene.zones[zone].hostile ? 1.F : 0.F;
    }
    zones[world_row::Zones * world_row::ZoneValues] = static_cast<float>(scene.zoneCount);
    row[world_row::BoundsAt] = scene.hardHalfNorth;
    row[world_row::BoundsAt + 1] = scene.hardHalfEast;
    float* const patrol = row + world_row::PatrolAt;
    std::size_t const samples = std::min(scene.track.m_sampleCount, world_row::TrackSamples);
    for (std::size_t sample = 0; sample < samples; ++sample)
    {
        patrol[2 * sample] = scene.track.m_north[sample];
        patrol[2 * sample + 1] = scene.track.m_east[sample];
    }
    patrol[2 * world_row::TrackSamples] = static_cast<float>(samples);
    float* const headings = patrol + 2 * world_row::TrackSamples + 1;
    for (std::size_t index = 0; index < Waypoints; ++index)
    {
        headings[index] = scene.waypoints[index].heading;
    }
    headings[Waypoints] = static_cast<float>(scene.validatedWaypoints);
}

/** @brief A blue slot's role, cone, patrol reference and lidar. */
void WriteBlueView(defenders::GeneralizedTeamScene const& scene, std::size_t slot, float* row) noexcept
{
    float* const view = row + world_row::BlueViewAt;
    int const role = scene.roles[slot];
    view[0] = static_cast<float>(role);
    view[1] = scene.blueCones[slot].radius;
    view[2] = scene.blueCones[slot].halfAngle;
    // The references are role-ordered: the leader first, then teammate slots W1, W2...
    std::size_t const entry = static_cast<std::size_t>(role < 0 ? 0 : role);
    bool const shown = role >= 0 && entry < scene.references.m_activeReferenceCount
        && scene.references.Visible() && scene.references.m_references[entry].m_valid;
    if (shown)
    {
        auto const& reference = scene.references.m_references[entry];
        view[3] = reference.m_projectedNorth;
        view[4] = reference.m_projectedEast;
        view[5] = reference.m_guidanceHeading;
        view[6] = 1.F;
    }
    defenders::SceneLidar const& lidar = scene.lidars[slot];
    if (lidar.valid)
    {
        float* const values = row + world_row::LidarAt;
        std::copy(lidar.ranges.begin(), lidar.ranges.end(), values);
        std::copy(lidar.distances.begin(), lidar.distances.end(), values + world_row::LidarBins);
        values[2 * world_row::LidarBins] = lidar.span;
    }
}

/** @brief The world's rows: each blue slot, then the scene (all zero for a slot beyond P but the scene). */
void WriteWorld(defenders::GeneralizedTeamScene const& scene, std::size_t blues, float* world) noexcept
{
    std::fill(world, world + Capacity * world_row::Size, 0.F);
    for (std::size_t slot = 0; slot < Capacity; ++slot)
    {
        float* const row = world + slot * world_row::Size;
        if (slot < blues)
        {
            defenders::Mobile const& blue = scene.state.blues[slot];
            defenders::GeneralizedPatrolTarget const& target = scene.targets[slot];
            row[0] = 1.F;
            row[1] = blue.available ? 1.F : 0.F;
            row[2] = slot == scene.leader ? 1.F : 0.F;
            row[3] = blue.north;
            row[4] = blue.east;
            row[5] = blue.heading;
            row[6] = blue.speed;
            row[7] = target.valid ? target.guidancePose.north : blue.north;
            row[8] = target.valid ? target.guidancePose.east : blue.east;
            WriteBlueView(scene, slot, row);
        }
        WriteSceneRow(scene, row);
    }
}

/** @brief Writes `count` rows of team observations and availability (absent slots zero). */
void WriteTeam(defenders::GeneralizedTeamExchange::Observations const& observations,
    defenders::GeneralizedTeamExchange::Availability const& availability, std::size_t count, float* rows,
    std::uint8_t* available) noexcept
{
    std::fill(rows, rows + Capacity * ObservationSize, 0.F);
    std::fill(available, available + Capacity, std::uint8_t{0});
    for (std::size_t slot = 0; slot < count; ++slot)
    {
        std::copy(observations[slot].values.begin(), observations[slot].values.end(), rows + slot * ObservationSize);
        available[slot] = availability[slot] ? 1 : 0;
    }
}

class BlueEnvironment final
{
public:
    static constexpr std::size_t Agents = Capacity;
    /** @brief task.json "metrics": the campaign's results of one episode. */
    static constexpr std::size_t MetricCount = 8;

    /**
     * @param project BluePatrol project folder (scenario.json, blue_patrol_training.json).
     * @param seed Not used: the scenario's seed drives the training episodes;
     *        an evaluation episode is set by reset_evaluation.
     */
    BlueEnvironment(std::filesystem::path const& project, std::uint64_t /*seed*/)
        : project_(project)
    {
        ppo::examples::airspace::RegisterAirspaceMeta();
        LoadScenario(ScenarioOf(project_));
    }

    void Reset(float* observations, std::uint8_t* available, float* world)
    {
        controller_->EndExternalEvaluation();
        StartEpisode(observations, available, world);
    }

    void ResetEvaluation(std::uint32_t seed, std::uint32_t episode, std::uint32_t /*episodes*/,
        float* observations, std::uint8_t* available, float* world)
    {
        controller_->StartExternalEvaluationCase(defenders::GeneralizedEvaluationCaseOf(seed, episode), episode);
        // The case's overrides apply when the world re-initializes its entities.
        world_->RequestReinitialization();
        StartEpisode(observations, available, world);
    }

    void BeginStep(float* observations, std::uint8_t* available, float* world)
    {
        PrepareStep(observations, available, world);
    }

    ppokit::runtime::StepEnd Step(float const* actions, float* observations, std::uint8_t* available,
        float* world, double* rewards, double* rewardChannels)
    {
        defenders::GeneralizedTeamExchange& exchange = controller_->Exchange();
        for (std::size_t slot = 0; slot < blues_; ++slot)
        {
            std::copy(actions + slot * ActionSize, actions + (slot + 1) * ActionSize, exchange.actions[slot].begin());
        }
        // A world re-initialization asked by another agent (a red) is taken at the next
        // begin_step: the blue team's episode ends only by its own outcome.
        world_->EndStep();
        episodeOpen_ = true;
        WriteTeam(exchange.next, exchange.nextAvailable, blues_, observations, available);
        WriteRewards(exchange, rewards, rewardChannels);
        if (world != nullptr)
        {
            WriteWorld(exchange.scene, blues_, world);
        }
        if (exchange.terminated || exchange.truncated)
        {
            episodeOpen_ = false;
            lastEpisode_ = exchange.episode;
        }
        if (exchange.terminated)
        {
            return ppokit::runtime::StepEnd::Terminated;
        }
        if (exchange.truncated)
        {
            return ppokit::runtime::StepEnd::Truncated;
        }
        return ppokit::runtime::StepEnd::Running;
    }

    /** @brief In task.json order: three flags, then the campaign's numbers and the score. */
    void EpisodeMetrics(double* metrics) const
    {
        metrics[0] = lastEpisode_.objectiveSurvived ? 1.0 : 0.0;
        metrics[1] = lastEpisode_.success ? 1.0 : 0.0;
        metrics[2] = lastEpisode_.timeout ? 1.0 : 0.0;
        metrics[3] = lastEpisode_.blueSurvival;
        metrics[4] = lastEpisode_.redNeutralization;
        metrics[5] = lastEpisode_.patrolQuality;
        metrics[6] = lastEpisode_.formationQuality;
        metrics[7] = defenders::GeneralizedEpisodeScore(lastEpisode_);
    }

    void SetWeights(double const* weights)
    {
        defenders::GeneralizedRewardValues values{};
        std::transform(weights, weights + Channels, values.begin(),
            [](double value) { return static_cast<float>(value); });
        controller_->SetRewardWeights(values);
    }

    void Weights(double* weights) const
    {
        defenders::GeneralizedRewardValues const& values = controller_->RewardWeights();
        std::copy(values.begin(), values.end(), weights);
    }

    /** @brief Apply the common engine's Blue team/red team/scenario payload without editing project files. */
    void ApplyCurriculumStage(std::string const& name, std::string const& payload)
    {
        auto const stage = nlohmann::json::parse(payload);
        if (!stage.is_object() || !stage.contains("blue") || !stage.contains("red")
            || !stage.contains("hostileZones") || !stage.at("blue").is_number_unsigned()
            || !stage.at("red").is_number_unsigned() || !stage.at("hostileZones").is_boolean())
        {
            throw std::invalid_argument("BlueDefense curriculum stage '" + name
                + "' requires blue, red and hostileZones");
        }
        auto scenario = nlohmann::json::parse(std::ifstream(ScenarioOf(project_)));
        auto& settings = scenario.at("blueDefense");
        settings.at("blueTeam")["activeCount"] = stage.at("blue");
        settings.at("redTeam")["activeCount"] = stage.at("red");
        settings.at("restrictedZones")["enabled"] = stage.at("hostileZones");
        if (!stage.at("hostileZones").get<bool>())
        {
            settings.at("restrictedZoneSpawn")["enabledProbability"] = 0.0;
        }
        if (stage.contains("objectiveLayout"))
        {
            settings.at("objectiveSpawn")["mode"] = stage.at("objectiveLayout");
        }
        auto stageFolder = std::make_unique<ScenarioValidationFolder>();
        auto const source = ScenarioOf(project_);
        auto const& folder = stageFolder->Path();
        auto const profiles = std::filesystem::path(scenario.at("agentProfiles").get<std::string>());
        scenario["agentProfiles"] = std::filesystem::absolute(source.parent_path() / profiles).generic_string();
        std::filesystem::path const staged = folder / "scenario.json";
        {
            std::ofstream output(staged);
            output.exceptions(std::ios::badbit | std::ios::failbit);
            output << scenario.dump(2) << '\n';
        }
        std::filesystem::copy_file(project_ / "blue_patrol_training.json", folder / "blue_patrol_training.json",
            std::filesystem::copy_options::overwrite_existing);
        std::optional<defenders::GeneralizedRewardValues> weights;
        if (controller_) weights = controller_->RewardWeights();
        LoadScenario(staged);
        if (weights) controller_->SetRewardWeights(*weights);
        stageFolder_ = std::move(stageFolder);
    }

private:
    /** @brief Rebuild the ECS world at a safe update boundary for a selected curriculum stage. */
    void LoadScenario(std::filesystem::path const& scenario)
    {
        auto world = std::make_unique<ppo::model::Environment>();
        auto controller = ppo::examples::bluedefense::LoadExternalScenario(scenario, *world);
        if (!controller)
            throw std::runtime_error(scenario.string() + ": not a valid BlueDefense scenario (see the messages above)");
        world_ = std::move(world);
        controller_ = std::move(controller);
        blues_ = controller_->Counts().blueMobiles;
        episodeOpen_ = false;
    }

    /** @brief Starts an episode: a re-initialization when one is still running, then its first step begins. */
    void StartEpisode(float* observations, std::uint8_t* available, float* world)
    {
        if (episodeOpen_)
        {
            world_->RequestReinitialization();
            episodeOpen_ = false;
        }
        PrepareStep(observations, available, world);
    }

    /** @brief The world begins its step; the blue controller publishes what the team acts on. */
    void PrepareStep(float* observations, std::uint8_t* available, float* world)
    {
        world_->BeginStep();
        controller_->PrepareExternalStep();
        defenders::GeneralizedTeamExchange const& exchange = controller_->Exchange();
        WriteTeam(exchange.observations, exchange.available, blues_, observations, available);
        if (world != nullptr)
        {
            WriteWorld(exchange.scene, blues_, world);
        }
    }

    /** @brief Each slot's clamped reward and weighted channels (slots beyond P zero). */
    void WriteRewards(defenders::GeneralizedTeamExchange const& exchange, double* rewards,
        double* rewardChannels) const noexcept
    {
        std::fill(rewards, rewards + Capacity, 0.0);
        std::fill(rewardChannels, rewardChannels + Capacity * Channels, 0.0);
        for (std::size_t slot = 0; slot < blues_; ++slot)
        {
            rewards[slot] = exchange.rewards[slot];
            std::copy(exchange.weightedChannels[slot].begin(), exchange.weightedChannels[slot].end(),
                rewardChannels + slot * Channels);
        }
    }

    std::filesystem::path project_;
    std::unique_ptr<ScenarioValidationFolder> stageFolder_;
    std::unique_ptr<ppo::model::Environment> world_;
    std::shared_ptr<defenders::GeneralizedPatrolMappoAgent> controller_;
    std::size_t blues_ = 0;
    /** @brief A step was taken since the episode started and did not end it. */
    bool episodeOpen_ = false;
    defenders::GeneralizedEpisodeEvaluation lastEpisode_{};
};

} // namespace

PPOKIT_ENVIRONMENT(BlueEnvironment, "blue_patrol", ObservationSize, ActionSize, Channels, world_row::Size)

/** @brief Allow changes to the existing simulation settings, preserving entity contracts. */
extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_scenario_editable_v1(char const* pointer)
{
    if (pointer == nullptr) return 0;
    constexpr std::string_view prefix = "/blueDefense/";
    std::string_view const path(pointer);
    return path.substr(0, prefix.size()) == prefix ? 1 : 0;
}

/**
 * @brief Validate a candidate in an isolated folder with the actual BlueDefense loader.
 * @param project Project containing the original scenario and training settings.
 * @param candidate Candidate JSON to validate.
 * @param error Buffer receiving a diagnostic on failure.
 * @param errorSize Capacity of the diagnostic buffer.
 * @return Zero when the scenario initializes successfully.
 */
extern "C" PPOKIT_ENVIRONMENT_EXPORT int ppokit_environment_validate_scenario_v1(
    char const* project, char const* candidate, char* error, std::size_t errorSize)
{
    try
    {
        auto const original = ScenarioOf(project);
        ScenarioValidationFolder const temporary;
        auto const& folder = temporary.Path();
        std::ifstream input(candidate);
        auto document = nlohmann::json::parse(input);
        auto const profiles = std::filesystem::path(document.at("agentProfiles").get<std::string>());
        document["agentProfiles"] = std::filesystem::absolute(original.parent_path() / profiles).generic_string();
        auto const isolated = folder / "blue-validation.json";
        {
            std::ofstream output(isolated);
            output.exceptions(std::ios::badbit | std::ios::failbit);
            output << document.dump(2) << '\n';
        }
        std::filesystem::copy_file(std::filesystem::path(project) / "blue_patrol_training.json",
            folder / "blue_patrol_training.json", std::filesystem::copy_options::overwrite_existing);
        ppo::examples::airspace::RegisterAirspaceMeta();
        ppo::model::Environment world;
        if (!ppo::examples::bluedefense::LoadExternalScenario(isolated, world))
            throw std::runtime_error("BlueDefense rejected the scenario; check team counts and spawn settings.");
        return 0;
    }
    catch (std::exception const& failure)
    {
        ppokit::environment::sdk::CopyError(failure.what(), error, errorSize);
    }
    catch (...)
    {
        ppokit::environment::sdk::CopyError("unknown BlueDefense scenario validation error", error, errorSize);
    }
    return 1;
}
