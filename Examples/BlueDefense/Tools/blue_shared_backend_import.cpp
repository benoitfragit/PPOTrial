/**
 * @file Examples/BlueDefense/Tools/blue_shared_backend_import.cpp
 * @brief Converts Blue MAPPO artifacts (rl_tools backend) into artifacts of the shared graph
 *        backend (ppokit_graph_network), and proves that the networks did not change.
 *
 * Usage:
 * @code
 * blue_shared_backend_import --project <folder with task.json and network.json>
 *     [--training <latest.ppokit> --to-training <latest.ppokit>]
 *     [--policy <policy.ppokit> --to-policy <policy.ppokit>]...
 * @endcode
 *
 * The project is the shared-backend form of BluePatrol (a "team_critic" section with compact
 * or padded critic inputs). Each source is read by the Blue learner (or its production policy)
 * and written for the shared backend (shared_backend_conversion.hpp). Then:
 * - a training state is loaded by TeamGraphLearner and written again: the file must be the
 *   same after its header (weights, both Adams, generator, updates, seed);
 * - its actor, and every policy, must give the same action means as the Blue network, bit
 *   for bit, for teams of 1 to 9 blues on probe observations.
 * Only then is the artifact kept. An existing destination is never overwritten.
 */
#include "generalized_patrol_learner.hpp"
#include "generalized_patrol_production_policy.hpp"
#include "shared_backend_conversion.hpp"

#include "ppokit/component/native_module.hpp"
#include "ppokit/graph/team_graph_learner.hpp"
#include "ppokit/schema/block_graph_schema.hpp"
#include "ppokit/schema/project.hpp"
#include "ppokit/schema/task_schema.hpp"
#include "ppokit/schema/train_schema.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace defenders = ppo::examples::bluedefense::defenders;

constexpr std::size_t Capacity = defenders::TeamCapacity::MaximumBlueMobiles;
constexpr std::size_t ObservationSize = defenders::GeneralizedObservationLayout::Size;
constexpr std::size_t ActionSize = defenders::GeneralizedObservationLayout::ActionSize;
/** @brief Probe observations per team size. */
constexpr int ProbeSteps = 16;

struct Conversion final
{
    std::filesystem::path source;
    std::filesystem::path target;
};

struct Options final
{
    std::filesystem::path project;
    std::filesystem::path training;
    std::filesystem::path toTraining;
    std::vector<Conversion> policies;
};

std::string ValueAfter(int argc, char** argv, int index)
{
    if (index + 1 >= argc)
    {
        throw std::invalid_argument(std::string(argv[index]) + " needs a value");
    }
    return argv[index + 1];
}

Options ParseOptions(int argc, char** argv)
{
    Options options;
    std::filesystem::path pendingPolicy;
    for (int index = 1; index < argc; index += 2)
    {
        std::string const option = argv[index];
        std::string const value = ValueAfter(argc, argv, index);
        if (option == "--project")
            options.project = value;
        else if (option == "--training")
            options.training = value;
        else if (option == "--to-training")
            options.toTraining = value;
        else if (option == "--policy")
        {
            if (!pendingPolicy.empty())
                throw std::invalid_argument("--policy requires a matching --to-policy before another --policy");
            pendingPolicy = value;
        }
        else if (option == "--to-policy")
        {
            if (pendingPolicy.empty())
                throw std::invalid_argument("--to-policy follows its --policy");
            options.policies.push_back({pendingPolicy, value});
            pendingPolicy.clear();
        }
        else
            throw std::invalid_argument("unknown option " + option);
    }
    bool const trainingPair = options.training.empty() == options.toTraining.empty();
    if (options.project.empty() || !trainingPair || !pendingPolicy.empty()
        || (options.training.empty() && options.policies.empty()))
    {
        throw std::invalid_argument("usage: blue_shared_backend_import --project <folder> [--training <file> "
                                    "--to-training <file>] [--policy <file> --to-policy <file>]...");
    }
    return options;
}

void RequireNew(std::filesystem::path const& path)
{
    if (std::filesystem::exists(path))
    {
        throw std::runtime_error(path.string() + " exists: remove it first (nothing is overwritten)");
    }
}

std::string Body(std::filesystem::path const& path)
{
    std::ifstream in(path, std::ios::binary);
    std::string header;
    std::getline(in, header);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

/** @brief Probe observations of a team of `counts`: own state, lidar, tracks and patrol vary. */
void FillProbe(defenders::ActiveTeamCounts counts, int step, defenders::BlueTeamRoles const& roles,
    defenders::GeneralizedPatrolProductionPolicy::Observations& observations,
    defenders::GeneralizedPatrolProductionPolicy::AvailabilityMask& available)
{
    using Layout = defenders::GeneralizedObservationLayout;
    auto const wave = [step](std::size_t feature, std::size_t agent) {
        return std::sin(0.37F * static_cast<float>(step) + 0.11F * static_cast<float>(feature)
            + 0.7F * static_cast<float>(agent));
    };
    auto const tracks = [&](defenders::GeneralizedRawObservation& values, std::size_t offset, std::size_t count,
                            std::size_t features, std::size_t salt) {
        for (std::size_t track = 0; track < count; ++track)
        {
            for (std::size_t feature = 0; feature + 1 < features; ++feature)
            {
                values[offset + track * features + feature] = wave(feature + 17U * track, salt);
            }
            values[defenders::GeneralizedTrackMaskIndex(offset, track, features)] = 1.F;
        }
    };
    available.fill(false);
    for (std::size_t agent = 0; agent < counts.blueMobiles; ++agent)
    {
        auto& observation = observations[agent];
        observation.metadata = {counts.blueMobiles, counts.redMobiles};
        observation.values.fill(0.F);
        for (std::size_t feature = 0; feature < Layout::RoleOffset; ++feature)
        {
            observation.values[feature] = wave(feature, agent);
        }
        tracks(observation.values, Layout::RedOffset, counts.redMobiles, Layout::RedFeatures, agent);
        tracks(observation.values, Layout::BlueOffset, counts.blueMobiles - 1U, Layout::FriendlyFeatures, agent + 3U);
        tracks(observation.values, Layout::ObjectiveOffset, 1U, Layout::FriendlyFeatures, agent + 6U);
        for (std::size_t feature = 0; feature < Layout::Patrol; ++feature)
        {
            observation.values[Layout::PatrolOffset + feature] = 0.5F * wave(feature + 101U, agent);
        }
        defenders::EncodeGeneralizedRole(observation.values, *roles.Assignment(agent));
        defenders::EncodeGeneralizedTeamCounts(observation.values, counts);
        available[agent] = true;
    }
}

/**
 * @brief Whether `shared` (an actor loaded by TeamGraphLearner) gives the means of the Blue
 *        actor in `blueFile`, bit for bit, for teams of 1 to 9 blues.
 */
bool SameMeans(ppokit::graph::TeamGraphLearner& shared, std::filesystem::path const& blueFile)
{
    for (std::size_t blues = 1; blues <= Capacity; ++blues)
    {
        defenders::ActiveTeamCounts const counts{blues, 2U};
        defenders::GeneralizedPatrolProductionPolicy policy(counts, blueFile);
        auto const roles = defenders::BlueTeamRoles::Create(blues, 0U);
        for (int step = 0; step < ProbeSteps; ++step)
        {
            defenders::GeneralizedPatrolProductionPolicy::Observations observations{};
            defenders::GeneralizedPatrolProductionPolicy::AvailabilityMask available{};
            FillProbe(counts, step, *roles, observations, available);
            auto const means = policy.Evaluate(observations, available);
            std::array<float, Capacity * ObservationSize> rows{};
            std::array<std::uint8_t, Capacity> present{};
            for (std::size_t agent = 0; agent < blues; ++agent)
            {
                std::copy(observations[agent].values.begin(), observations[agent].values.end(),
                    rows.begin() + static_cast<std::ptrdiff_t>(agent * ObservationSize));
                present[agent] = 1;
            }
            std::array<float, Capacity * ActionSize> actions{};
            shared.Act(rows.data(), present.data(), actions.data(), true);
            for (std::size_t agent = 0; agent < blues; ++agent)
            {
                if (std::memcmp(means[agent].data(), actions.data() + agent * ActionSize, ActionSize * sizeof(float))
                    != 0)
                {
                    return false;
                }
            }
        }
    }
    return true;
}

/** @brief The project's networks and settings, read once. */
struct SharedProject final
{
    ppokit::schema::TaskSchema task;
    ppokit::schema::BlockGraphSchema graphs;
    ppokit::core::PpoSettings settings;
    ppokit::artifacts::ArtifactIdentity identity;
};

SharedProject ReadProject(std::filesystem::path const& folder)
{
    // Resolve observer libraries exactly as the runner does for a shipped project.
    ppokit::component::SetKnownModules(ppokit::schema::ReadProgramRegistry(PPOKIT_PROGRAM_REGISTRY));
    ppokit::schema::Project files;
    files.root = folder;
    SharedProject project;
    project.task = ppokit::schema::LoadTaskSchema(files.Task());
    project.graphs = ppokit::schema::LoadBlockGraph(files.Network(), project.task);
    if (std::filesystem::exists(files.Training()))
    {
        project.settings = ppokit::schema::LoadTrainingPlan(files.Training(), project.task).ppo.Settings();
    }
    project.identity = ppokit::schema::ArtifactIdentityOf(project.task, project.graphs);
    return project;
}

/** @brief Private staging directory for one conversion; never reuses an existing directory. */
class ConversionStage final
{
public:
    explicit ConversionStage(std::filesystem::path const& target)
        : directory_(target.parent_path() / (target.filename().string() + ".conversion"))
    {
        if (!std::filesystem::create_directory(directory_))
            throw std::runtime_error(directory_.string() + " exists: nothing is overwritten");
    }

    ~ConversionStage()
    {
        std::error_code ignored;
        std::filesystem::remove_all(directory_, ignored);
    }

    ConversionStage(ConversionStage const&) = delete;
    ConversionStage& operator=(ConversionStage const&) = delete;

    /** @brief Path inside the private staging directory. */
    std::filesystem::path File(char const* name) const { return directory_ / name; }

    /**
     * @brief Publish without replacing an existing destination, even if it appeared during conversion.
     * @param staged Validated file in the staging directory.
     * @param target Destination on the same filesystem.
     */
    void Publish(std::filesystem::path const& staged, std::filesystem::path const& target) const
    {
        std::error_code error;
        std::filesystem::create_hard_link(staged, target, error);
        if (error)
            throw std::runtime_error("cannot publish " + target.string() + " without overwriting: " + error.message());
    }

private:
    std::filesystem::path directory_;
};

void ConvertTraining(SharedProject const& project, std::filesystem::path const& source,
    std::filesystem::path const& target)
{
    RequireNew(target);
    defenders::GeneralizedPatrolLearner learner{{3U, 2U}, 1U, 128U};
    learner.LoadTrainingState(source);
    if (target.has_parent_path()) std::filesystem::create_directories(target.parent_path());
    ConversionStage const stage(target);
    std::filesystem::path const staged = stage.File("training.ppokit");
    defenders::WriteSharedBackendTraining(learner, project.task, project.graphs, staged);
    ppokit::graph::TeamGraphLearner shared(project.task, project.graphs, project.settings, 1);
    shared.LoadTraining(staged, project.identity);
    std::filesystem::path const rewritten = stage.File("reloaded.ppokit");
    shared.SaveTraining(rewritten, project.identity);
    bool const sameState = Body(staged) == Body(rewritten);
    if (!sameState || shared.Updates() != learner.StepCount() || !SameMeans(shared, source))
    {
        throw std::runtime_error(source.string() + ": the converted training state does not reproduce the source");
    }
    stage.Publish(staged, target);
    std::cout << source.string() << " -> " << target.string() << ": update " << learner.StepCount()
              << ", same state on reload, same means for 1..9 blues\n";
}

void ConvertPolicy(SharedProject const& project, Conversion const& conversion)
{
    RequireNew(conversion.target);
    defenders::GeneralizedPatrolLearner learner{{3U, 2U}, 1U, 128U};
    learner.LoadActor(conversion.source);
    if (conversion.target.has_parent_path()) std::filesystem::create_directories(conversion.target.parent_path());
    ConversionStage const stage(conversion.target);
    std::filesystem::path const staged = stage.File("policy.ppokit");
    defenders::WriteSharedBackendPolicy(learner, project.task, project.graphs, staged);
    ppokit::graph::TeamGraphLearner shared(project.task, project.graphs,
        ppokit::core::InferenceSettings(project.settings), 1);
    shared.LoadPolicy(staged, project.identity);
    if (!SameMeans(shared, conversion.source))
    {
        throw std::runtime_error(conversion.source.string() + ": the converted policy does not act as the source");
    }
    stage.Publish(staged, conversion.target);
    std::cout << conversion.source.string() << " -> " << conversion.target.string()
              << ": same means for 1..9 blues\n";
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Options const options = ParseOptions(argc, argv);
        SharedProject const project = ReadProject(options.project);
        if (!options.training.empty())
        {
            ConvertTraining(project, options.training, options.toTraining);
        }
        for (Conversion const& conversion : options.policies)
        {
            ConvertPolicy(project, conversion);
        }
    }
    catch (std::exception const& error)
    {
        std::cerr << "blue_shared_backend_import: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
