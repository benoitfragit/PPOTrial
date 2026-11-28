/**
 * @file PPOKit/src/environment/environment_library.cpp
 * @brief Loading an environment DLL and calling its C API from C++.
 */
#include "ppokit/environment/environment_library.hpp"

#include "ppokit/component/native_module.hpp"

#include <array>
#include <stdexcept>

namespace ppokit::environment {

namespace {

constexpr std::size_t ErrorCapacity = 1024;

/** @brief Finds one function of the C API, or says which is missing. */
template<typename Function>
void Bind(void* module, char const* name, Function& function, std::filesystem::path const& path)
{
    function = reinterpret_cast<Function>(component::NativeSymbol(module, name));
    if (function == nullptr)
    {
        throw std::runtime_error(path.string() + ": not a PPOKit environment library (no " + name
            + "; export the environment with PPOKIT_ENVIRONMENT)");
    }
}

} // namespace

std::shared_ptr<EnvironmentLibrary const> EnvironmentLibrary::Open(std::string const& reference,
    std::filesystem::path const& directory, bool requireRunProfile)
{
    std::shared_ptr<EnvironmentLibrary> library(new EnvironmentLibrary());
    library->path_ = component::ResolveNativeModule(reference, directory, "environment library");
    library->handle_ = component::OpenNativeModule(library->path_);
    void* const module = library->handle_;
    std::filesystem::path const& path = library->path_;
    EnvironmentFunctions& f = library->functions_;
    Bind(module, "ppokit_environment_info", f.info, path);
    f.info(&library->info_);
    // The version first: a DLL of another ABI may lack functions of this one.
    if (library->info_.abi != PPOKIT_ENVIRONMENT_ABI)
    {
        throw std::runtime_error(path.string() + ": environment ABI " + std::to_string(library->info_.abi)
            + ", this program reads " + std::to_string(PPOKIT_ENVIRONMENT_ABI) + ": rebuild it");
    }
    if (library->info_.agentCapacity == 0)
    {
        throw std::runtime_error(path.string() + ": an environment holds at least one agent");
    }
    Bind(module, "ppokit_environment_create", f.create, path);
    Bind(module, "ppokit_environment_reset", f.reset, path);
    Bind(module, "ppokit_environment_reset_evaluation", f.resetEvaluation, path);
    Bind(module, "ppokit_environment_begin_step", f.beginStep, path);
    Bind(module, "ppokit_environment_episode_metrics", f.episodeMetrics, path);
    Bind(module, "ppokit_environment_step", f.step, path);
    Bind(module, "ppokit_environment_set_learning", f.setLearning, path);
    Bind(module, "ppokit_environment_training_state", f.trainingState, path);
    Bind(module, "ppokit_environment_restore_training_state", f.restoreTrainingState, path);
    Bind(module, "ppokit_environment_apply_curriculum_stage", f.applyCurriculumStage, path);
    Bind(module, "ppokit_environment_set_weights", f.setWeights, path);
    Bind(module, "ppokit_environment_weights", f.weights, path);
    Bind(module, "ppokit_environment_error", f.error, path);
    Bind(module, "ppokit_environment_destroy", f.destroy, path);
    if (requireRunProfile)
    {
        Bind(module, "ppokit_environment_run_profile_v1", f.runProfile, path);
        Bind(module, "ppokit_environment_reset_inference_v1", f.resetInference, path);
    }
    f.scenarioEditable = reinterpret_cast<PpokitScenarioEditableV1*>(
        component::NativeSymbol(module, "ppokit_environment_scenario_editable_v1"));
    f.validateScenario = reinterpret_cast<PpokitValidateScenarioV1*>(
        component::NativeSymbol(module, "ppokit_environment_validate_scenario_v1"));
    return library;
}

bool EnvironmentLibrary::HasScenarioEditor() const noexcept
{
    return functions_.scenarioEditable != nullptr && functions_.validateScenario != nullptr;
}

bool EnvironmentLibrary::CanEditScenario(std::string const& pointer) const noexcept
{
    return HasScenarioEditor() && functions_.scenarioEditable(pointer.c_str()) == 1;
}

void EnvironmentLibrary::ValidateScenario(std::filesystem::path const& project,
    std::filesystem::path const& candidate) const
{
    if (!HasScenarioEditor()) throw std::logic_error("environment has no complete scenario editor/1 capability");
    std::array<char, ErrorCapacity> error{};
    if (functions_.validateScenario(project.string().c_str(), candidate.string().c_str(), error.data(), error.size()) != 0)
        throw std::runtime_error(path_.string() + ": " + error.data());
}

PpokitRunProfileV1 EnvironmentLibrary::RunProfile(std::filesystem::path const& project, std::uint64_t update) const
{
    if (!functions_.runProfile) throw std::logic_error("project has no environment/1 run profile");
    std::array<char, ErrorCapacity> error{};
    PpokitRunProfileV1 profile{};
    if (functions_.runProfile(project.string().c_str(), update, &profile, error.data(), error.size()) != 0)
        throw std::runtime_error(path_.string() + ": " + error.data());
    if (!profile.maxUpdates || !profile.checkpointEvery || !profile.inferenceEpisodes
        || profile.randomizeTrainingSeed > 1 || profile.resumeIfAvailable > 1)
        throw std::runtime_error(path_.string() + ": invalid environment/1 run profile");
    return profile;
}

void EnvironmentLibrary::VerifyTaskContract(std::uint64_t expected) const
{
    component::RequireNativeTaskContract(handle_, path_, expected);
}

EnvironmentLibrary::~EnvironmentLibrary()
{
    if (handle_ != nullptr)
    {
        component::CloseNativeModule(handle_);
    }
}

Environment::Environment(std::shared_ptr<EnvironmentLibrary const> library, std::filesystem::path const& project,
    std::uint64_t seed)
    : library_(std::move(library))
{
    std::array<char, ErrorCapacity> error{};
    environment_ = library_->Functions().create(project.string().c_str(), seed, error.data(), error.size());
    if (environment_ == nullptr)
    {
        throw std::runtime_error(std::string("environment: ") + error.data());
    }
}

Environment::~Environment()
{
    library_->Functions().destroy(environment_);
}

void Environment::Require(int code, char const* call) const
{
    if (code != 0)
    {
        throw std::runtime_error(std::string("environment ") + call + ": " + library_->Functions().error(environment_));
    }
}

void Environment::Reset(float* observations, std::uint8_t* available, float* world)
{
    Require(library_->Functions().reset(environment_, observations, available, world), "reset");
}

void Environment::ResetEvaluation(std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes,
    float* observations, std::uint8_t* available, float* world)
{
    Require(library_->Functions().resetEvaluation(
                environment_, seed, episode, episodes, observations, available, world),
        "reset_evaluation");
}

void Environment::ResetInference(std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes,
    float* observations, std::uint8_t* available, float* world)
{
    auto const reset = library_->Functions().resetInference;
    if (!reset) throw std::logic_error("project has no native inference reset");
    Require(reset(environment_, seed, episode, episodes, observations, available, world), "reset_inference");
}

void Environment::BeginStep(float* observations, std::uint8_t* available, float* world)
{
    Require(library_->Functions().beginStep(environment_, observations, available, world), "begin_step");
}

void Environment::EpisodeMetrics(double* metrics) const
{
    Require(library_->Functions().episodeMetrics(environment_, metrics), "episode_metrics");
}

runtime::StepEnd Environment::Step(float const* actions, float* observations, std::uint8_t* available, float* world,
    double* rewards, double* rewardChannels)
{
    std::uint32_t end = 0;
    Require(library_->Functions().step(
                environment_, actions, observations, available, world, rewards, rewardChannels, &end),
        "step");
    switch (end)
    {
    case PPOKIT_STEP_RUNNING:
        return runtime::StepEnd::Running;
    case PPOKIT_STEP_TERMINATED:
        return runtime::StepEnd::Terminated;
    case PPOKIT_STEP_TRUNCATED:
        return runtime::StepEnd::Truncated;
    default:
        throw std::runtime_error("environment step: unknown end " + std::to_string(end));
    }
}

void Environment::SetWeights(double const* weights)
{
    Require(library_->Functions().setWeights(environment_, weights), "set_weights");
}

void Environment::Weights(double* weights) const
{
    Require(library_->Functions().weights(environment_, weights), "weights");
}

void Environment::SetLearning(bool learning)
{
    Require(library_->Functions().setLearning(environment_, learning ? 1 : 0), "set_learning");
}

std::string Environment::TrainingState() const
{
    std::size_t size = 0;
    Require(library_->Functions().trainingState(environment_, nullptr, 0, &size), "training_state");
    std::string state(size, '\0');
    if (size > 0)
    {
        std::size_t written = 0;
        Require(library_->Functions().trainingState(environment_, state.data(), state.size(), &written),
            "training_state");
        if (written != size)
        {
            throw std::runtime_error("environment training_state: its size changed between two calls");
        }
    }
    return state;
}

void Environment::RestoreTrainingState(std::string const& state)
{
    Require(library_->Functions().restoreTrainingState(environment_, state.data(), state.size()),
        "restore_training_state");
}

void Environment::ApplyCurriculumStage(std::string const& name, std::string const& payload)
{
    Require(library_->Functions().applyCurriculumStage(environment_, name.c_str(), payload.c_str()),
        "apply_curriculum_stage");
}

} // namespace ppokit::environment
