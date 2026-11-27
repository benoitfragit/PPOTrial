/**
 * @file PPOKit/tests/published_models_tests.cpp
 * @brief Exercise repository checkpoints through the actual runtime network ABI.
 * @details Uses immutable published policies and the committed LATEST where available.
 * No checkpoint in the source tree is written; no production training is started.
 */
#include "ppokit/artifacts/artifact.hpp"
#include "ppokit/artifacts/sha256.hpp"
#include "ppokit/component/native_module.hpp"
#include "ppokit/environment/environment_library.hpp"
#include "ppokit/network/network_library.hpp"
#include "ppokit/schema/policy_reference.hpp"
#include "ppokit/schema/project.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;
using namespace ppokit;

void Require(bool condition, std::string const& message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

/** @brief A valid observation from the real environment, for the loaded models. */
struct Sample final
{
    std::vector<float> observations;
    std::vector<std::uint8_t> available;
    std::vector<float> actions;
};

Sample Observe(schema::Project const& project, schema::TaskSchema const& task,
    std::vector<fs::path> const& registry)
{
    auto const path = schema::FindLibrary(project, project.environmentLibrary, registry);
    Require(!path.empty(), project.name + ": environment module missing");
    auto const library = environment::EnvironmentLibrary::Open(path.string(), {}, project.environmentRunProfile);
    library->VerifyTaskContract(schema::SemanticHash(task));
    auto const& info = library->Info();
    Require(info.agentCapacity == task.maximumAgents && info.observationSize == task.observation.size
        && info.actionSize == task.actionSize, project.name + ": environment dimensions differ");
    environment::Environment env(library, project.root, 771U);
    env.SetLearning(false);
    Sample sample{std::vector<float>(info.agentCapacity * info.observationSize),
        std::vector<std::uint8_t>(info.agentCapacity), std::vector<float>(info.agentCapacity * info.actionSize)};
    // The ABI writes agentCapacity rows for the world as well as for observations.
    std::vector<float> world(static_cast<std::size_t>(info.agentCapacity) * info.worldSize);
    env.Reset(sample.observations.data(), sample.available.data(), world.empty() ? nullptr : world.data());
    Require(std::any_of(sample.available.begin(), sample.available.end(), [](auto value) { return value != 0; }),
        project.name + ": no present agent after reset");
    return sample;
}

using Library = std::shared_ptr<network::NetworkLibrary const>;

/** @brief Open either a policy-only or complete training artifact for deterministic inference. */
std::vector<float> Infer(Library const& library, schema::Project const& project, fs::path const& file,
    Sample const& sample)
{
    std::vector<float> result(sample.actions.size());
    if (library->Info().agentCapacity == 1)
    {
        network::Policy policy(library, project.root, file);
        policy.Act(sample.observations.data(), result.data(), true);
    }
    else
    {
        network::TeamPolicy policy(library, project.root, file);
        policy.Act(sample.observations.data(), sample.available.data(), result.data());
    }
    Require(std::all_of(result.begin(), result.end(), [](float value) { return std::isfinite(value); }),
        file.string() + ": non-finite inference action");
    return result;
}

/** @brief Serialized training payload, excluding its timestamped JSON header. */
std::string Payload(fs::path const& file)
{
    std::ifstream input(file, std::ios::binary);
    Require(static_cast<bool>(input), file.string() + ": cannot open saved payload");
    std::string header;
    std::getline(input, header);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

/** @brief Save a fresh or resumed learner at its update boundary using the production ABI. */
template<typename Learner>
void SaveRoundTrip(Library const& library, schema::Project const& project, fs::path const& source,
    fs::path const& training, fs::path const& policy, std::uint64_t expectedUpdates)
{
    Learner learner(library, project.root, 991U);
    if (!source.empty())
    {
        learner.Load(source);
    }
    Require(learner.Updates() == expectedUpdates, project.name + ": wrong resumed update count");
    learner.Save(training, policy);
}

void Save(Library const& library, schema::Project const& project, fs::path const& source,
    fs::path const& training, fs::path const& policy, std::uint64_t expectedUpdates)
{
    if (library->Info().agentCapacity == 1)
        SaveRoundTrip<network::Learner>(library, project, source, training, policy, expectedUpdates);
    else
        SaveRoundTrip<network::TeamLearner>(library, project, source, training, policy, expectedUpdates);
}

/** @brief The actual published checkpoint must survive a resume/save/reload without losing state. */
void CheckResume(Library const& library, schema::Project const& project, Sample const& sample,
    fs::path const& work, bool historical)
{
    fs::path const original = historical ? project.LatestCheckpoint() : fs::path{};
    if (historical)
        Require(fs::is_regular_file(original), original.string() + ": committed training checkpoint missing");
    auto const originalHeader = historical ? artifacts::ReadArtifactHeader(original) : artifacts::ArtifactHeader{};
    auto const originalDigest = historical ? artifacts::Sha256OfFile(original) : std::string{};
    if (historical)
        Require(originalHeader.kind == artifacts::ArtifactKind::Training, "LATEST is not a training checkpoint");

    auto const first = work / "first.ppokit";
    auto const firstPolicy = work / "first_policy.ppokit";
    auto const second = work / "second.ppokit";
    auto const secondPolicy = work / "second_policy.ppokit";
    Save(library, project, original, first, firstPolicy, originalHeader.updates);
    Save(library, project, first, second, secondPolicy, originalHeader.updates);
    Require(Payload(first) == Payload(second), project.name + ": checkpoint payload changed on reload");
    auto const savedHeader = artifacts::ReadArtifactHeader(second);
    Require(savedHeader.updates == originalHeader.updates, project.name + ": update lost during save");
    if (historical)
    {
        Require(savedHeader.seed == originalHeader.seed, project.name + ": training seed changed during resume");
        Require(artifacts::Sha256OfFile(original) == originalDigest, "source training checkpoint changed");
        Require(Infer(library, project, original, sample) == Infer(library, project, secondPolicy, sample),
            project.name + ": resumed actor differs from original training actor");
    }
    Require(Infer(library, project, first, sample) == Infer(library, project, secondPolicy, sample),
        project.name + ": extracted policy and full training artifact produce different actions");
    Require(Infer(library, project, firstPolicy, sample) == Infer(library, project, secondPolicy, sample),
        project.name + ": reloading changes the deterministic actions");
}

void CheckModels(fs::path const& folder, fs::path const& work, bool historical)
{
    auto const project = schema::LoadProject(folder);
    auto const task = schema::LoadTaskSchema(project.Task());
    auto const registry = schema::ReadProgramRegistry(PPOKIT_PROGRAM_REGISTRY);
    component::SetKnownModules(registry);
    auto const path = schema::FindLibrary(project, project.networkLibrary, registry);
    Require(!path.empty(), project.name + ": network module missing");
    auto const library = network::NetworkLibrary::Open(path.string(), {}, &task);
    library->VerifyTaskContract(schema::SemanticHash(task));
    Require(library->Info().agentCapacity == task.maximumAgents
        && library->Info().observationSize == task.observation.size && library->Info().actionSize == task.actionSize,
        project.name + ": loaded network dimensions differ");

    auto const sample = Observe(project, task, registry);
    auto const references = schema::LoadPolicyReferences(project.root);
    Require(!references.empty(), project.name + ": no published reference");
    for (auto const& reference : references)
    {
        Require(artifacts::Sha256OfFile(reference.file) == reference.sha256, reference.role + ": reference digest differs");
        auto const header = artifacts::ReadArtifactHeader(reference.file);
        Require(!reference.artifactUpdate || header.updates == *reference.artifactUpdate,
            reference.role + ": published update differs");
        Require(Infer(library, project, reference.file, sample) == Infer(library, project, reference.file, sample),
            reference.role + ": deterministic inference differs after reopening");
        std::cout << project.name << " " << reference.role << " inference: PASS\n";
    }
    if (historical)
    {
        auto const file = project.Policy();
        Require(fs::is_regular_file(file), file.string() + ": default policy missing");
        (void)Infer(library, project, file, sample);
    }
    CheckResume(library, project, sample, work, historical);
    std::cout << project.name << (historical ? " committed LATEST" : " fresh training state")
        << " resume/save/reload: PASS\n";
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 4 || (std::string(argv[3]) != "resume" && std::string(argv[3]) != "scratch"))
        return EXIT_FAILURE;
    fs::path const work = argv[2];
    try
    {
        fs::create_directories(work);
        CheckModels(argv[1], work, std::string(argv[3]) == "resume");
        for (char const* name : {"first.ppokit", "first_policy.ppokit", "second.ppokit", "second_policy.ppokit"})
            fs::remove(work / name);
        return EXIT_SUCCESS;
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
