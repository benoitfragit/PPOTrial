#pragma once

// RLtools CPU adapter. All simulation/reward equations stay in patrol_core.hpp;
// the RL backend remains a replaceable implementation detail.
#include "patrol_core.hpp"
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>

#include <random>
#include <type_traits>

namespace rl_tools::rl::environments {
template<typename Scalar, typename Index>
struct BluePatrol
{
    using T = Scalar;
    using TI = Index;
    using State = ppo::examples::twovstwo::defenders::State;
    static constexpr TI OBSERVATION_DIM =
        static_cast<TI>(ppo::examples::twovstwo::defenders::ObservationLayout::Size);
    static constexpr TI OBSERVATION_DIM_PRIVILEGED = OBSERVATION_DIM;
    static constexpr TI ACTION_DIM =
        static_cast<TI>(ppo::examples::twovstwo::defenders::ObservationLayout::ActionSize);
    ppo::examples::twovstwo::defenders::Parameters parameters{};
};
} // namespace rl_tools::rl::environments

namespace rl_tools {
// Deliberately declared before PPO's on-policy runner is included: its
// templated dispatch must resolve the exact environment overloads.
template<typename DEVICE, typename T, typename TI>
void malloc(DEVICE&, rl::environments::BluePatrol<T, TI> const&) {}

template<typename DEVICE, typename T, typename TI>
void init(DEVICE&, rl::environments::BluePatrol<T, TI> const&, bool = false) {}

template<typename DEVICE, typename T, typename TI, typename RNG>
void sample_initial_state(DEVICE&, rl::environments::BluePatrol<T, TI> const& env,
                          typename rl::environments::BluePatrol<T, TI>::State& state,
                          RNG& rng)
{
    std::uniform_int_distribution<int> side{0, 1};
    std::uniform_int_distribution<std::size_t> role{0U, 1U};
    std::uniform_real_distribution<float> phase{
        0.F, ppo::examples::twovstwo::defenders::CircuitLength(env.parameters)};
    state = ppo::examples::twovstwo::defenders::InitialState(
        env.parameters, side(rng) == 0 ? -1 : 1, role(rng), phase(rng));
}

template<typename DEVICE, typename T, typename TI, typename ACTION_SPEC, typename RNG>
T step(DEVICE&, rl::environments::BluePatrol<T, TI> const& env,
       typename rl::environments::BluePatrol<T, TI>::State const& state,
       Matrix<ACTION_SPEC> const& actions,
       typename rl::environments::BluePatrol<T, TI>::State& next, RNG&)
{
    static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == 2);
    using namespace ppo::examples::twovstwo::defenders;
    next = Advance(env.parameters, state, Action{
        static_cast<float>(get(actions, 0, 0)),
        static_cast<float>(get(actions, 0, 1))});
    return env.parameters.dt;
}

template<typename DEVICE, typename T, typename TI, typename ACTION_SPEC, typename RNG>
T reward(DEVICE&, rl::environments::BluePatrol<T, TI> const& env,
         typename rl::environments::BluePatrol<T, TI>::State const& before,
         Matrix<ACTION_SPEC> const& actions,
         typename rl::environments::BluePatrol<T, TI>::State const& after, RNG&)
{
    using namespace ppo::examples::twovstwo::defenders;
    return TeamReward(env.parameters, before, Action{
        static_cast<float>(get(actions, 0, 0)),
        static_cast<float>(get(actions, 0, 1))}, after);
}

template<typename DEVICE, typename T, typename TI, typename OBS_SPEC, typename RNG>
void observe(DEVICE&, rl::environments::BluePatrol<T, TI> const& env,
             typename rl::environments::BluePatrol<T, TI>::State const& state,
             Matrix<OBS_SPEC>& target, RNG&)
{
    static_assert(OBS_SPEC::ROWS == 1);
    static_assert(OBS_SPEC::COLS ==
        ppo::examples::twovstwo::defenders::ObservationLayout::Size);
    auto const obs = ppo::examples::twovstwo::defenders::Observe(env.parameters, state);
    for (TI i = 0; i < OBS_SPEC::COLS; ++i)
        set(target, 0, i, static_cast<T>(obs[i]));
}

template<typename DEVICE, typename T, typename TI, typename OBS_SPEC, typename RNG>
void observe_privileged(DEVICE& device, rl::environments::BluePatrol<T, TI> const& env,
                        typename rl::environments::BluePatrol<T, TI>::State const& state,
                        Matrix<OBS_SPEC>& target, RNG& rng)
{
    // Local critic for PPO v1. MAPPO will use a separate centralized view.
    observe(device, env, state, target, rng);
}

template<typename DEVICE, typename T, typename TI, typename RNG>
bool terminated(DEVICE&, rl::environments::BluePatrol<T, TI> const& env,
                typename rl::environments::BluePatrol<T, TI>::State const& state,
                RNG&)
{
    // The patrol horizon is a genuine finite-horizon episode. Marking it as a
    // terminal value boundary preserves its final reward in RLtools GAE.
    return state.failed || state.step >= env.parameters.episodeSteps;
}
} // namespace rl_tools

#include "attention_actor.hpp"
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>

namespace ppo::examples::twovstwo::defenders {

namespace rlt = rl_tools;
using Device = rlt::devices::DEVICE_FACTORY<>;
using Index = typename Device::index_t;
using Engine = decltype(rlt::random::default_engine(typename Device::SPEC::RANDOM{}));
using Environment = rlt::rl::environments::BluePatrol<float, Index>;

struct PPOParameters : rlt::rl::algorithms::ppo::DefaultParameters<float, Index>
{
    static constexpr Index N_EPOCHS = 4;
    static constexpr float GAMMA = 0.99F;
    static constexpr float LAMBDA = 0.95F;
    static constexpr float EPSILON_CLIP = 0.2F;
    static constexpr float ACTION_ENTROPY_COEFFICIENT = 0.01F;
    static constexpr bool NORMALIZE_OBSERVATIONS = false; // Features normalized in Observe.
};

template<Index RolloutSteps>
struct TrainingParametersFor
    : rlt::rl::algorithms::ppo::loop::core::Parameters<float, Index, Environment>
{
    static_assert(RolloutSteps >= 128 && RolloutSteps <= 4096,
                  "Patrol rollout capacity is outside the supported range");
    using PPO_PARAMETERS = PPOParameters;
    static constexpr Index N_ENVIRONMENTS = 2;
    static constexpr Index ON_POLICY_RUNNER_STEPS_PER_ENV = RolloutSteps;
    static constexpr Index BATCH_SIZE = 128;
    static constexpr Index ACTOR_HIDDEN_DIM = 128;
    static constexpr Index CRITIC_HIDDEN_DIM = 128;
    static constexpr Index ACTOR_NUM_LAYERS = 3;
    static constexpr Index CRITIC_NUM_LAYERS = 3;
    static constexpr Index EPISODE_STEP_LIMIT = 4096;
    static constexpr Index STEP_LIMIT = 300;
};

template<typename T,typename TI,typename ENV,typename PARAMETERS>
struct AttentionApproximators
    : rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsMLP<T,TI,ENV,PARAMETERS>
{
    using Base=rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsMLP<T,TI,ENV,PARAMETERS>;
    using ACTOR_TYPE=AttentionActor<typename Base::ACTOR_SPEC>;
};

template<Index RolloutSteps>
using TrainingConfigFor = rlt::rl::algorithms::ppo::loop::core::Config<
    float, Index, Engine, Environment, TrainingParametersFor<RolloutSteps>,
    AttentionApproximators>;

using TrainingParameters = TrainingParametersFor<2048>;
using TrainingConfig = TrainingConfigFor<2048>;
using TrainingState = typename TrainingConfig::template State<TrainingConfig>;

} // namespace ppo::examples::twovstwo::defenders
