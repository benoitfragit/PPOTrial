/**
 * @file Framework/RL/Runtime/include/ppo/runtime/ppoparameters.hpp
 * @brief Interface ou implémentation du composant ppoparameters.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

namespace ppo::model {

/**
* @brief PPO hyper-parameters shared by training agents.
*/
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

} // namespace ppo::model
