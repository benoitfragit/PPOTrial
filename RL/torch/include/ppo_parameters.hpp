#pragma once

namespace ppo::model {
struct PPOParamters
{
    /**
    * @brief Discount factor.
    */
    float m_gamma{ 0.99F };
    /**
    * @brief Generalized advantage estimation lambda.
    */
    float m_lambda_gae{ 0.95F };
    /**
    * @brief PPO clipping factor.
    */
    float m_clip_epsilon{ 0.2F };
    /**
    * @brief Critic loss weight.
    */
    float m_value_coef{ 0.5F };
    /**
    * @brief Entropy bonus weight.
    */
    float m_entropy_coef{ 0.02F };
    /**
    * @brief Optimizer learning rate.
    */
    float m_learningReate{ 3e-4F };
    /**
    * @brief Number of PPO epochs per update.
    */
    int m_K_epochs{ 4 };
    /**
    * @brief Mini-batch size.
    */
    int m_batch_size{ 64 };
    /**
    * @brief Target KL divergence threshold.
    */
    float m_target_kl{ 0.01F };
};

struct PPOOutputs
{
    /**
    * @brief Total PPO loss.
    */
    float m_loss{ 0.F };
    /**
    * @brief Actor loss term.
    */
    float m_actorloss{ 0.F };
    /**
    * @brief Critic loss term.
    */
    float m_valueloss{ 0.F };
    /**
    * @brief Entropy regularization term.
    */
    float m_entropyloss{ 0.F };
};
}
