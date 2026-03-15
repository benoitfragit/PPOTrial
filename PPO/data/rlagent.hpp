#pragma once

#include <memory>
#include <cstdint>
#include <filesystem>
#include "trainingnet.hpp"

namespace ppo::model {

/**
 * @brief High-level reinforcement-learning agent wrapper.
 *
 * This class coordinates policy inference, rollout buffering, training steps,
 * and model persistence through the PPO backend.
 */
class RLAgent {
public:
                RLAgent         (std::shared_ptr<ITrainingNet> policy,
                                 std::string const& modelpath,
                                 long long state_dim, 
                                 long long action_dim, 
                                 long long buffer_dim,
                                 PPOParamters const& parameters,
                                 float* const stateVector,
                                 float* const actionVector);

                ~RLAgent        (void);

                RLAgent         (RLAgent const&) = delete;
    RLAgent&    operator=       (RLAgent const&) = delete;
                RLAgent         (RLAgent&&)      = delete;
    RLAgent&    operator=       (RLAgent&&)      = delete;

    /** @brief Initializes internal training and inference state. */
    void        Init            (void);
    /** @brief Computes and stores the next action from the current state. */
    void        ChooseAction    (void);
    /**
     * @brief Pushes transition feedback for the previously chosen action.
     * @param reward Reward observed after the action.
     * @param done Episode termination flag.
     */
    void        SetNewState     (float reward, bool done);
    /** @brief Indicates whether the agent is currently in training mode. */
    bool        IsTraining      (void) const noexcept;
    /** @brief Returns the current training step index. */
    long long   GetCurrentStep  (void) const noexcept;
    /** @brief Returns the configured maximum number of steps. */
    long long   GetMaxStep      (void) const noexcept;
    /**
     * @brief Saves model weights and metadata to disk.
     * @param modelpath Destination path.
     */
    void        Save            (std::filesystem::path const& modelpath);
    /** @brief Returns last optimization metrics. */
    PPOOutputs const& GetOutputs(void) const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::model */