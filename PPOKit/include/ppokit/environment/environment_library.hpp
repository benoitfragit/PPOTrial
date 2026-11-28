/**
 * @file PPOKit/include/ppokit/environment/environment_library.hpp
 * @brief C++ side of an environment DLL (ppokit/environment/environment_abi.h).
 */
#pragma once

#include "ppokit/environment/run_profile.h"
#include "ppokit/runtime/rollout.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace ppokit::environment {

/** @brief Function table of a loaded environment DLL. */
struct EnvironmentFunctions final
{
    PpokitReadRunProfileV1* runProfile = nullptr;
    PpokitResetInferenceV1* resetInference = nullptr;
    PpokitScenarioEditableV1* scenarioEditable = nullptr;
    PpokitValidateScenarioV1* validateScenario = nullptr;
    decltype(&ppokit_environment_info) info = nullptr;
    decltype(&ppokit_environment_create) create = nullptr;
    decltype(&ppokit_environment_reset) reset = nullptr;
    decltype(&ppokit_environment_reset_evaluation) resetEvaluation = nullptr;
    decltype(&ppokit_environment_begin_step) beginStep = nullptr;
    decltype(&ppokit_environment_episode_metrics) episodeMetrics = nullptr;
    decltype(&ppokit_environment_step) step = nullptr;
    decltype(&ppokit_environment_set_learning) setLearning = nullptr;
    decltype(&ppokit_environment_training_state) trainingState = nullptr;
    decltype(&ppokit_environment_restore_training_state) restoreTrainingState = nullptr;
    decltype(&ppokit_environment_apply_curriculum_stage) applyCurriculumStage = nullptr;
    decltype(&ppokit_environment_set_weights) setWeights = nullptr;
    decltype(&ppokit_environment_weights) weights = nullptr;
    decltype(&ppokit_environment_error) error = nullptr;
    decltype(&ppokit_environment_destroy) destroy = nullptr;
};

/** @brief A loaded environment DLL; it stays loaded while an environment uses it. */
class EnvironmentLibrary final
{
public:
    /**
     * @brief Loads an environment DLL by path, or by bare name in `directory`.
     * @throws std::runtime_error when it is missing, is not an environment DLL, or was
     *         built for another ABI.
     */
    static std::shared_ptr<EnvironmentLibrary const> Open(std::string const& reference,
        std::filesystem::path const& directory = {}, bool requireRunProfile = false);

    /** @brief Read the required native profile for the learner update about to start. */
    PpokitRunProfileV1 RunProfile(std::filesystem::path const& project, std::uint64_t update) const;

    /** @brief Both optional scenario editor symbols must be present. */
    bool HasScenarioEditor() const noexcept;
    bool CanEditScenario(std::string const& pointer) const noexcept;
    /** @brief Validate a candidate with the native parser; throws its diagnostic on rejection. */
    void ValidateScenario(std::filesystem::path const& project, std::filesystem::path const& candidate) const;

    /** @brief Check the immutable semantic identity of the loaded module against task.json. */
    void VerifyTaskContract(std::uint64_t expected) const;

    ~EnvironmentLibrary();
    EnvironmentLibrary(EnvironmentLibrary const&) = delete;
    EnvironmentLibrary& operator=(EnvironmentLibrary const&) = delete;

    PpokitEnvironmentInfo const& Info() const noexcept { return info_; }
    EnvironmentFunctions const& Functions() const noexcept { return functions_; }
    std::filesystem::path const& Path() const noexcept { return path_; }

private:
    EnvironmentLibrary() = default;

    std::filesystem::path path_;
    void* handle_ = nullptr;
    EnvironmentFunctions functions_;
    PpokitEnvironmentInfo info_{};
};

/** @brief One environment of a project. Every call throws the DLL's reason. */
class Environment final
{
public:
    Environment(std::shared_ptr<EnvironmentLibrary const> library, std::filesystem::path const& project,
        std::uint64_t seed);
    ~Environment();
    Environment(Environment const&) = delete;
    Environment& operator=(Environment const&) = delete;

    /** @brief Starts an episode: observations and `available` have Info().agentCapacity rows. */
    void Reset(float* observations, std::uint8_t* available, float* world);
    /** @brief Starts episode `episode` of `episodes` of an evaluation campaign, from `seed`. */
    void ResetEvaluation(std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes, float* observations,
        std::uint8_t* available, float* world);
    /** @brief Starts one episode using the native inference profile declared by the project. */
    void ResetInference(std::uint32_t seed, std::uint32_t episode, std::uint32_t episodes, float* observations,
        std::uint8_t* available, float* world);
    /** @brief Before a step that is not an episode's first: the state the agents act on (see the ABI). */
    void BeginStep(float* observations, std::uint8_t* available, float* world);
    /** @brief Metrics of the episode that just ended: Info().metricCount values. */
    void EpisodeMetrics(double* metrics) const;
    /**
     * @brief One step of the agents present.
     * @param rewards agentCapacity rewards; @param rewardChannels agentCapacity x channels.
     * @return How the step ended, for the team.
     */
    runtime::StepEnd Step(float const* actions, float* observations, std::uint8_t* available, float* world,
        double* rewards, double* rewardChannels);
    void SetWeights(double const* weights);
    void Weights(double* weights) const;
    /** @brief Whether the episodes that follow feed a learner (train) or only play (see the ABI). */
    void SetLearning(bool learning);
    /** @brief The training state to keep with the run's checkpoint (see the ABI). */
    std::string TrainingState() const;
    /** @brief Restores a state of TrainingState; the empty state starts a fresh run. */
    void RestoreTrainingState(std::string const& state);
    /** @brief Applies the payload selected by the common curriculum engine. */
    void ApplyCurriculumStage(std::string const& name, std::string const& payload);
    PpokitEnvironmentInfo const& Info() const noexcept { return library_->Info(); }

private:
    void Require(int code, char const* call) const;

    std::shared_ptr<EnvironmentLibrary const> library_;
    PpokitEnvironment* environment_ = nullptr;
};

} // namespace ppokit::environment
