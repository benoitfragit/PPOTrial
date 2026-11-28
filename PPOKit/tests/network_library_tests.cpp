/**
 * @file PPOKit/tests/network_library_tests.cpp
 * @brief The network DLL of the reach project (ppokit_add_network_library): its C API gives
 *        exactly what the in-process graph learner gives — same decisions, same updates, same
 *        policy after a save — and reports its errors in words.
 */
#include "task.hpp"

#include "ppokit/network/network_library.hpp"
#include "ppokit/component/native_module.hpp"
#include "ppokit/schema/task_schema.hpp"
#include "ppokit/tools/command_line.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

namespace {

using namespace ppokit;
using reach_example::ActionSize;
using reach_example::ObservationSize;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

bool Contains(std::string const& text, std::string const& part)
{
    return text.find(part) != std::string::npos;
}

/** @brief The in-process learner the DLL must match, with the same seed. */
graph::GraphLearner<ObservationSize, ActionSize> InProcess(tools::detail::Setup const& setup, std::uint64_t seed)
{
    tools::detail::Setup seeded = setup;
    seeded.plan.ppo.Set(core::PpoParameter::Seed, static_cast<double>(seed));
    return tools::detail::MakeLearner<ObservationSize, ActionSize>(seeded);
}

/**
 * @brief Collects one rollout of the reach task through the DLL, exactly as the in-process
 *        collector does (decide, step, critic value of the next observation unless terminated).
 */
void CollectThroughDll(network::Learner& learner, reach_example::ReachTask& task)
{
    auto observation = task.Reset();
    while (!learner.Full())
    {
        std::array<float, ActionSize> action{};
        double logProbability = 0.0;
        double value = 0.0;
        learner.Decide(observation.data(), action.data(), logProbability, value);
        std::array<float, ObservationSize> next{};
        auto const outcome = task.Step(action, next);
        PpokitTransition transition{};
        transition.observation = observation.data();
        transition.action = action.data();
        transition.logProbability = logProbability;
        transition.value = value;
        transition.reward = outcome.reward;
        transition.end = static_cast<std::uint32_t>(outcome.end);
        if (outcome.end != runtime::StepEnd::Terminated)
        {
            transition.nextValue = learner.Value(next.data());
        }
        learner.Append(transition);
        observation = runtime::EndsEpisode(outcome.end) ? task.Reset() : next;
    }
}

} // namespace

int main()
{
    std::filesystem::path const project(PPOKIT_REACH_DIR);
    std::filesystem::path const folder = std::filesystem::temp_directory_path() / "ppokit_network_library_tests";
    std::error_code ignored;
    std::filesystem::remove_all(folder, ignored);
    std::filesystem::create_directories(folder);
    try
    {
        auto const library = network::NetworkLibrary::Open(PPOKIT_REACH_NETWORK);
        library->VerifyTaskContract(reach_example::TaskLayout::SemanticHash);
        bool refusedContract = false;
        try
        {
            library->VerifyTaskContract(reach_example::TaskLayout::SemanticHash ^ 1U);
        }
        catch (std::runtime_error const& error)
        {
            refusedContract = std::string(error.what()).find("compiled task contract differs") != std::string::npos;
        }
        Check(refusedContract, "the network library checks its own compiled task contract");
        std::filesystem::path const oldModule(PPOKIT_NOT_A_NETWORK);
        std::unique_ptr<void, decltype(&component::CloseNativeModule)> const missing(
            component::OpenNativeModule(oldModule), &component::CloseNativeModule);
        bool refusedUnversioned = false;
        try
        {
            component::RequireNativeTaskContract(missing.get(), oldModule, reach_example::TaskLayout::SemanticHash);
        }
        catch (std::runtime_error const& error)
        {
            refusedUnversioned = std::string(error.what()).find("missing compiled task contract") != std::string::npos;
        }
        Check(refusedUnversioned, "a module without a semantic identity requires an explicit rebuild");
        PpokitNetworkInfo const& info = library->Info();
        Check(info.abi == PPOKIT_NETWORK_ABI && info.observationSize == ObservationSize
                && info.actionSize == ActionSize && std::string(info.task) == "reach"
                && std::string(info.backend) == "graph",
            "the DLL says what it computes");

        // A native caller receives the settings validation error across the DLL boundary.
        auto const invalidProject = folder / "invalid";
        std::filesystem::copy(project, invalidProject, std::filesystem::copy_options::recursive);
        std::ofstream(invalidProject / "training.json")
            << R"({"schema":"ppokit.train/1","ppo":{"initial_log_std":1,"maximum_log_std":0}})";
        bool refused = false;
        try { network::Learner invalid(library, invalidProject, 11); }
        catch (std::exception const& error) { refused = Contains(error.what(), "initial_log_std"); }
        Check(refused, "DLL reports inconsistent PPO settings");

        tools::detail::Files files;
        files.task = project / "task.json";
        files.graph = project / "network.json";
        files.train = project / "training.json";
        tools::detail::Setup const setup = tools::detail::ReadSetup<ObservationSize, ActionSize>(files);
        constexpr std::uint64_t Seed = 11;

        // Same seed, same decisions.
        network::Learner dll(library, project, Seed);
        auto local = InProcess(setup, Seed);
        std::array<float, ObservationSize> const observation{0.3F, -0.2F, 0.5F};
        std::array<float, ActionSize> action{};
        double logProbability = 0.0;
        double value = 0.0;
        dll.Decide(observation.data(), action.data(), logProbability, value);
        auto const expected = local.Decide(observation);
        Check(action == expected.action && logProbability == expected.logProbability && value == expected.value,
            "a decision of the DLL is the in-process learner's, bit for bit");

        // Same rollout, same update.
        network::Learner trained(library, project, Seed);
        auto reference = InProcess(setup, Seed);
        auto task = reach_example::MakeTask(3);
        CollectThroughDll(trained, task);
        PpokitUpdateStatistics const statistics = trained.Update();
        auto referenceTask = reach_example::MakeTask(3);
        decltype(reference)::Buffer buffer(setup.plan.ppo.RolloutSteps());
        runtime::RolloutCollector<reach_example::ReachTask, decltype(reference), decltype(buffer)> collector(
            referenceTask, reference);
        collector.Collect(buffer);
        core::UpdateStatistics const expectedUpdate = reference.Update(buffer);
        Check(statistics.updates == 1 && trained.Updates() == 1, "the update is counted");
        Check(statistics.policyLoss == expectedUpdate.policyLoss && statistics.valueLoss == expectedUpdate.valueLoss
                && statistics.approximateKl == expectedUpdate.approximateKl,
            "an update through the DLL is the in-process update, bit for bit");

        // Save, then infer: the policy gives the mean of the trained actor.
        std::filesystem::path const training = folder / "latest.ppokit";
        std::filesystem::path const policyFile = folder / "policy.ppokit";
        trained.Save(training, policyFile);
        network::Policy policy(library, project, policyFile);
        std::array<float, ActionSize> mean{};
        policy.Act(observation.data(), mean.data(), true);
        Check(mean == reference.Mean(observation), "the saved policy acts as the trained actor's mean");
        network::Learner resumed(library, project, Seed);
        resumed.Load(training);
        Check(resumed.Updates() == 1, "a training state resumes with its update count");

        // The same shared module accepts independent contracts, not a global task setting.
        auto const graphTask = schema::LoadTaskSchema(project / "task.json");
        auto const shared = network::NetworkLibrary::Open(PPOKIT_SHARED_GRAPH_NETWORK, {}, &graphTask);
        auto const differentTask = schema::ParseTaskSchema(R"({
            "schema": "ppokit.task/1", "name": "other",
            "observation": {"state": {"fields": ["x"]}},
            "actions": {"move": {"head": "gaussian", "names": ["position"]}},
            "rewards": {"R": {}}
        })", "other/task.json");
        auto const other = network::NetworkLibrary::Open(PPOKIT_SHARED_GRAPH_NETWORK, {}, &differentTask);
        shared->VerifyTaskContract(schema::SemanticHash(graphTask));
        other->VerifyTaskContract(schema::SemanticHash(differentTask));
        Check(std::string(shared->Info().task) == graphTask.name && std::string(other->Info().task) == differentTask.name
                && shared->Info().observationSize == graphTask.observation.size
                && other->Info().observationSize == differentTask.observation.size,
            "one DLL retains independent per-client task dimensions");
        network::Learner sharedLearner(shared, project, Seed);
        auto sharedTask = reach_example::MakeTask(3);
        CollectThroughDll(sharedLearner, sharedTask);
        auto const sharedUpdate = sharedLearner.Update();
        Check(sharedUpdate.policyLoss == statistics.policyLoss && sharedUpdate.valueLoss == statistics.valueLoss
                && sharedUpdate.approximateKl == statistics.approximateKl,
            "shared graph DLL matches the fixed-contract update");
        network::Policy sharedPolicy(shared, project, policyFile);
        sharedPolicy.Act(observation.data(), mean.data(), true);
        Check(mean == reference.Mean(observation), "shared graph DLL opens the unchanged policy format");
        network::Learner sharedResumed(shared, project, Seed);
        sharedResumed.Load(training);
        Check(sharedResumed.Updates() == 1, "shared graph DLL resumes the unchanged training format");

        // A partially collected rollout belongs to the current policy and must not be
        // silently dropped by a save or combined with the weights of another checkpoint.
        network::Learner partial(shared, project, Seed);
        partial.Decide(observation.data(), action.data(), logProbability, value);
        PpokitTransition pending{};
        pending.observation = observation.data();
        pending.action = action.data();
        pending.logProbability = logProbability;
        pending.value = value;
        pending.nextValue = value;
        pending.end = PPOKIT_STEP_RUNNING;
        partial.Append(pending);
        auto const unsaved = folder / "partial.ppokit";
        bool partialSaveRefused = false;
        try { partial.Save(unsaved, {}); }
        catch (std::exception const& error) { partialSaveRefused = Contains(error.what(), "update boundary"); }
        Check(partialSaveRefused && !std::filesystem::exists(unsaved), "partial training save is rejected before writing");
        bool partialLoadRefused = false;
        try { partial.Load(training); }
        catch (std::exception const& error) { partialLoadRefused = Contains(error.what(), "update boundary"); }
        Check(partialLoadRefused && partial.Updates() == 0, "partial training load cannot mix policies");
        auto const partialPolicy = folder / "partial_policy.ppokit";
        partial.Save({}, partialPolicy);
        network::Policy readable(shared, project, partialPolicy);
        readable.Act(observation.data(), mean.data(), true);
        bool wrongBoundTask = false;
        try { network::Learner invalid(other, project, Seed); }
        catch (std::exception const& error) { wrongBoundTask = Contains(error.what(), "bound graph task contract differs"); }
        Check(wrongBoundTask, "shared graph DLL rejects a project different from its bound task");
        bool missingTask = false;
        try { network::NetworkLibrary::Open(PPOKIT_SHARED_GRAPH_NETWORK); }
        catch (std::exception const& error) { missingTask = Contains(error.what(), "requires a valid task"); }
        Check(missingTask, "shared graph DLL cannot be used without a task contract");
        auto const multiAgent = schema::LoadTaskSchema(PPOKIT_SAMPLE_TASK);
        auto const teamBackend = network::NetworkLibrary::Open(PPOKIT_SHARED_GRAPH_NETWORK, {}, &multiAgent);
        bool teamKept = teamBackend->Info().agentCapacity == multiAgent.maximumAgents;
        try { network::Learner single(teamBackend, project, Seed); teamKept = false; }
        catch (std::exception const& error) { teamKept = teamKept && Contains(error.what(), "single-agent"); }
        Check(teamKept, "the shared backend opens a team task as a team (team_graph_learner_tests)");

        // Errors in words.
        std::filesystem::path const wrong = folder / "wrong";
        std::filesystem::create_directories(wrong);
        std::filesystem::copy_file(PPOKIT_SAMPLE_TASK, wrong / "task.json");
        std::string message;
        try
        {
            network::Learner refused(library, wrong);
        }
        catch (std::exception const& error)
        {
            message = error.what();
        }
        Check(Contains(message, "is not the task this network DLL was built for (reach): rebuild the DLL"),
            "another task is refused with the reason: " + message);
        message.clear();
        try
        {
            network::NetworkLibrary::Open(PPOKIT_NOT_A_NETWORK);
        }
        catch (std::exception const& error)
        {
            message = error.what();
        }
        Check(Contains(message, "not a PPOKit network library"), "another DLL is refused: " + message);

        // A team DLL has the team functions only, a single-agent DLL the others only.
        auto const team = network::NetworkLibrary::Open(PPOKIT_TEAM_LINE_NETWORK);
        Check(team->Info().agentCapacity == 3 && team->Functions().teamLearnerCreate != nullptr
                && team->Functions().learnerCreate == nullptr,
            "a team DLL binds the team functions");
        Check(library->Info().agentCapacity == 1 && library->Functions().teamLearnerCreate == nullptr,
            "a single-agent DLL has no team functions");
        message.clear();
        try
        {
            network::Learner single(team, folder);
        }
        catch (std::exception const& error)
        {
            message = error.what();
        }
        Check(Contains(message, "have no single-agent policy or learner"), "a team DLL refuses a learner: " + message);
        message.clear();
        try
        {
            network::TeamLearner wrong(library, folder);
        }
        catch (std::exception const& error)
        {
            message = error.what();
        }
        Check(Contains(message, "single-agent networks have no team"), "a single DLL refuses a team learner: " + message);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    std::filesystem::remove_all(folder, ignored);
    if (failures != 0)
    {
        std::cerr << failures << " network library check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit network library tests passed\n";
    return EXIT_SUCCESS;
}
