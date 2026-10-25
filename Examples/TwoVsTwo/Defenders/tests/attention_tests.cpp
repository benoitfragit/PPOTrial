#include "patrol_rltools.hpp"
#include <cmath>
#include <iostream>

namespace blue=ppo::examples::twovstwo::defenders;
namespace rlt=rl_tools;

int main()
{
    blue::Device device{};
    using Spec=rlt::matrix::Specification<float,blue::Index,1,blue::ObservationLayout::Size>;
    rlt::MatrixDynamic<Spec> input{},output{},grad{};
    rlt::malloc(device,input);
    rlt::malloc(device,output);
    rlt::malloc(device,grad);
    rlt::set_all(device,input,0.F);
    rlt::set_all(device,grad,0.F);
    blue::TrainingConfig::NN::ACTOR_TYPE actor{};
    rlt::malloc(device,actor);
    auto rng=rlt::random::default_engine(typename blue::Device::SPEC::RANDOM(),42U);
    rlt::init_weights(device,actor,rng);
    rlt::zero_gradient(device,actor);
    using Layout=blue::ObservationLayout;

    // All hostile slots masked, even if non-mask features contain garbage.
    for(blue::Index i=Layout::RedOffset;i<Layout::FriendOffset;++i)
        rlt::set(input,0,i,0.75F);
    for(blue::Index j=0;j<Layout::RedTracks;++j)
        rlt::set(input,0,Layout::RedOffset+j*Layout::RedFeatures+
                 Layout::RedFeatures-1,0.F);
    auto const empty=actor.red.Transform(input,0,Layout::RedOffset,output);
    if(empty.count!=0) return 1;
    for(blue::Index i=0;i<24;++i)
        if(rlt::get(output,0,Layout::RedOffset+i)!=0.F)return 2;
    rlt::set(grad,0,Layout::RedOffset,1.F);
    actor.red.Differentiate(input,0,Layout::RedOffset,grad,empty);
    if(rlt::get(actor.red.encoderW.gradient,0,0)!=0.F)return 3;

    // Both friend and objective tracks valid. Finite-difference gradient
    // must match the analytic PPO chain rule through masked attention.
    rlt::set(input,0,0,0.23F);
    for(blue::Index j=0;j<Layout::FriendlyTracks;++j)
    {
        rlt::set(input,0,Layout::FriendOffset+j*Layout::FriendlyFeatures,
                 j==0?0.7F:-0.35F);
        rlt::set(input,0,Layout::FriendOffset+j*Layout::FriendlyFeatures+
                 Layout::FriendlyFeatures-1,1.F);
    }
    auto const valid=actor.friendlies.Transform(input,0,Layout::FriendOffset,output);
    if(valid.count!=2)return 4;
    rlt::set_all(device,grad,0.F);
    rlt::set(grad,0,Layout::FriendOffset,1.F);
    actor.friendlies.Differentiate(input,0,Layout::FriendOffset,grad,valid);
    float const analytic=rlt::get(actor.friendlies.encoderW.gradient,0,0);
    float const old=rlt::get(actor.friendlies.encoderW.parameters,0,0);
    float constexpr epsilon=1e-3F;
    rlt::set(actor.friendlies.encoderW.parameters,0,0,old+epsilon);
    actor.friendlies.Transform(input,0,Layout::FriendOffset,output);
    float const positive=rlt::get(output,0,Layout::FriendOffset);
    rlt::set(actor.friendlies.encoderW.parameters,0,0,old-epsilon);
    actor.friendlies.Transform(input,0,Layout::FriendOffset,output);
    float const negative=rlt::get(output,0,Layout::FriendOffset);
    rlt::set(actor.friendlies.encoderW.parameters,0,0,old);
    float const numerical=(positive-negative)/(2.F*epsilon);
    if(!std::isfinite(analytic)||!std::isfinite(numerical)||
       std::abs(analytic-numerical)>0.003F) {
        std::cerr<<"Analytic gradient "<<analytic<<" differs from "<<numerical<<'\n';
        return 5;
    }
    rlt::free(device,actor);
    rlt::free(device,input);
    rlt::free(device,output);
    rlt::free(device,grad);
    return 0;
}
