/**
 * @file Examples/Airspace/Agents/Policy/interception_policy.cpp
 * @brief Forward pass of the RLSNA actor.
 */
#include "interception_policy.hpp"

#include <stdexcept>

namespace rl {
namespace {

constexpr char Architecture[] = "interception_mlp";

} // namespace

InterceptionPolicy::InterceptionPolicy(std::filesystem::path const& path, std::size_t observationSize,
    std::size_t actionSize)
    : hidden1_(Hidden), hidden2_(Hidden)
{
    LegacyPolicyFile const file(path);
    if (file.Architecture() != Architecture)
    {
        throw std::runtime_error(path.string() + ": architecture " + file.Architecture() + ", expected "
            + Architecture);
    }
    layer1_ = LegacyDense(file, "fc1", observationSize, Hidden);
    layer2_ = LegacyDense(file, "fc2", Hidden, Hidden);
    mean_ = LegacyDense(file, "mu_head", Hidden, actionSize);
}

ppo::core::Action InterceptionPolicy::ChooseAction(ppo::core::Observation const& observation)
{
    if (observation.size() != layer1_.Inputs())
    {
        throw std::invalid_argument("InterceptionPolicy: observation dimension mismatch");
    }
    layer1_.Apply(observation.data(), hidden1_.data());
    ApplyRelu(hidden1_.data(), Hidden);
    layer2_.Apply(hidden1_.data(), hidden2_.data());
    ApplyRelu(hidden2_.data(), Hidden);
    ppo::core::Action action(mean_.Outputs());
    mean_.Apply(hidden2_.data(), action.data());
    return action;
}

} // namespace rl
