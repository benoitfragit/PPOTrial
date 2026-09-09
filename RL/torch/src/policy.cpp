#include "torch_policy.hpp"
#include "actorcriticnet.hpp"
#include "rlagent.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ppo::rl {
namespace {
void validate_dimensions(std::size_t observations,std::size_t actions,std::size_t rollout)
{
    constexpr std::size_t maximum=1000000;
    if(observations==0 || actions==0 || rollout==0 || observations>maximum || actions>maximum
        || rollout>maximum || observations>maximum/rollout || actions>maximum/rollout)
        throw std::invalid_argument("invalid or excessive PPO dimensions");
}
void copy_observation(const Value& value,std::vector<float>& destination)
{
    const auto* source=std::get_if<std::vector<float>>(&value);
    if(source==nullptr || source->size()!=destination.size()) throw std::invalid_argument("PPO observation shape mismatch");
    for(float element:*source)
        if(!std::isfinite(element)) throw std::invalid_argument("nonfinite observation");
    std::copy(source->begin(),source->end(),destination.begin());
}
Value bounded_action(const std::vector<float>& raw)
{
    std::vector<float> result=raw;
    for(auto& element:result)
    {
        if(!std::isfinite(element)) throw std::runtime_error("nonfinite policy action");
        element=std::clamp(element,-1.F,1.F);
    }
    return result;
}
void validate_parameters(const model::PPOParamters& p)
{
    if(!std::isfinite(p.m_gamma) || p.m_gamma<0 || p.m_gamma>1
       || !std::isfinite(p.m_lambda_gae) || p.m_lambda_gae<0 || p.m_lambda_gae>1
       || !std::isfinite(p.m_clip_epsilon) || p.m_clip_epsilon<=0
       || !std::isfinite(p.m_learningReate) || p.m_learningReate<=0
       || !std::isfinite(p.m_value_coef) || p.m_value_coef<0
       || !std::isfinite(p.m_entropy_coef) || p.m_entropy_coef<0
       || !std::isfinite(p.m_target_kl) || p.m_target_kl<=0
       || p.m_K_epochs<=0 || p.m_K_epochs>10000 || p.m_batch_size<=0)
        throw std::invalid_argument("invalid PPO parameters");
}
}
struct TorchLearner::Impl
{
    std::vector<float> state;
    std::vector<float> action;
    std::shared_ptr<::rl::ActorCriticNet> network;
    std::unique_ptr<model::RLAgent> agent;
    float gamma=0;
    bool pending=false;
};
TorchLearner::TorchLearner(std::size_t observation_size,std::size_t action_size,std::size_t rollout_size,
                         const std::filesystem::path& checkpoint_root,const model::PPOParamters& parameters)
    :impl_(std::make_unique<Impl>())
{
    validate_dimensions(observation_size,action_size,rollout_size);
    validate_parameters(parameters);
    impl_->state.resize(observation_size);
    impl_->action.resize(action_size);
    impl_->network=std::make_shared<::rl::ActorCriticNet>(static_cast<int64_t>(observation_size),static_cast<int64_t>(action_size));
    impl_->agent=std::make_unique<model::RLAgent>(impl_->network,checkpoint_root.string(),
        static_cast<long long>(observation_size),static_cast<long long>(action_size),static_cast<long long>(rollout_size),
        parameters,impl_->state.data(),impl_->action.data(),false);
    impl_->gamma=parameters.m_gamma;
    impl_->agent->Init();
}
TorchLearner::~TorchLearner()=default;
Value TorchLearner::act(const Value& observation)
{
    if(impl_->pending) throw std::logic_error("observe required before next action");
    copy_observation(observation,impl_->state);
    if(!impl_->agent->IsTraining()) impl_->agent->Init();
    impl_->agent->ChooseAction();
    auto result=bounded_action(impl_->action);
    impl_->pending=true;
    return result;
}
void TorchLearner::observe(const StepResult& transition)
{
    if(!impl_->pending) throw std::logic_error("act required before observe");
    copy_observation(transition.observation,impl_->state);
    float final_value=0;
    if(transition.truncated && !transition.terminated)
    {
        torch::NoGradGuard no_grad;
        const auto tensor=torch::from_blob(impl_->state.data(),{1,static_cast<int64_t>(impl_->state.size())},torch::kFloat);
        const auto outputs=impl_->network->forward(tensor);
        final_value=std::get<2>(outputs).item<float>();
    }
    const float reward=bootstrap_reward(transition.reward,transition.terminated,transition.truncated,final_value,impl_->gamma);
    impl_->agent->SetNewState(reward,transition.terminated || transition.truncated);
    impl_->pending=false;
}
void TorchLearner::save_candidate(const std::filesystem::path& path)
{
    if(impl_->pending) throw std::logic_error("cannot checkpoint a pending transition");
    impl_->agent->Save(path);
}
struct FrozenTorchPolicy::Impl
{
    std::vector<float> state;
    std::size_t action_size=0;
    std::unique_ptr<::rl::ActorCriticNet> network;
};
FrozenTorchPolicy::FrozenTorchPolicy(std::size_t observation_size,std::size_t action_size,const std::filesystem::path& checkpoint)
    :impl_(std::make_unique<Impl>())
{
    validate_dimensions(observation_size,action_size,1);
    impl_->state.resize(observation_size);
    impl_->action_size=action_size;
    impl_->network=std::make_unique<::rl::ActorCriticNet>(static_cast<int64_t>(observation_size),static_cast<int64_t>(action_size));
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint.string());
    impl_->network->load(archive);
    impl_->network->eval();
}
FrozenTorchPolicy::~FrozenTorchPolicy()=default;
Value FrozenTorchPolicy::act(const Value& observation)
{
    copy_observation(observation,impl_->state);
    torch::NoGradGuard no_grad;
    const auto tensor=torch::from_blob(impl_->state.data(),{1,static_cast<int64_t>(impl_->state.size())},torch::kFloat);
    const auto action=impl_->network->forward_mu(tensor).contiguous();
    if(action.numel()!=static_cast<int64_t>(impl_->action_size)) throw std::runtime_error("policy action shape mismatch");
    const auto* first=action.data_ptr<float>();
    return bounded_action(std::vector<float>(first,first+impl_->action_size));
}
} // namespace ppo::rl
