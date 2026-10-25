#pragma once
// Fully differentiable RLtools masked track attention. Separate hostile/friendly
// branches, shared per-track encoders, learned query, softmax, mean/max pool.
// Designed to be called from native PPO, including backward gradients.
#include "patrol_core.hpp"
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

namespace ppo::examples::twovstwo::defenders {
namespace rlt = rl_tools;

template<typename T, typename TI, TI F, TI N, TI D>
struct TrackAttention {
    static constexpr TI Features=F, Tracks=N, Width=D, OutputWidth=3*D;
    template<TI R, TI C, typename KIND>
    using Param = typename rlt::nn::parameters::Adam::template instance<
        typename rlt::nn::parameters::Plain::template spec<
            rlt::MatrixDynamic<rlt::matrix::Specification<T,TI,R,C>>,
            rlt::nn::parameters::groups::Normal,KIND>>;
    using W=rlt::nn::parameters::categories::Weights;
    using B=rlt::nn::parameters::categories::Biases;
    Param<D,F,W> encoderW{};
    Param<1,D,B> encoderB{};
    Param<D,ObservationLayout::Own,W> queryW{};
    Param<1,D,B> queryB{};
    using Vector=std::array<T,D>;
    struct Cache {
        std::array<Vector,N> features{};
        std::array<T,N> probabilities{};
        std::array<bool,N> valid{};
        Vector query{}, context{}, mean{}, max{};
        TI count=0;
    };
    template<typename IN,typename OUT>
    Cache Transform(IN const& src,TI row,TI offset,OUT& dst) const {
        Cache c{};
        T const scale=T(1)/std::sqrt(T(D));
        c.max.fill(-std::numeric_limits<T>::infinity());
        for(TI d=0;d<D;++d) {
            T z=rlt::get(queryB.parameters,0,d);
            for(TI i=0;i<ObservationLayout::Own;++i)
                z+=rlt::get(queryW.parameters,d,i)*rlt::get(src,row,i);
            c.query[d]=std::tanh(z);
        }
        std::array<T,N> scores{};
        T largest=-std::numeric_limits<T>::infinity();
        for(TI j=0;j<N;++j){
            auto const base=offset+j*F;
            c.valid[j]=rlt::get(src,row,base+F-1)>T(0.5);
            if(!c.valid[j])continue;
            ++c.count;
            T score=0;
            for(TI d=0;d<D;++d){
                T z=rlt::get(encoderB.parameters,0,d);
                for(TI i=0;i<F;++i)
                    z+=rlt::get(encoderW.parameters,d,i)*rlt::get(src,row,base+i);
                T const h=std::tanh(z);
                c.features[j][d]=h;
                c.mean[d]+=h;
                c.max[d]=std::max(c.max[d],h);
                score+=c.query[d]*h*scale;
            }
            largest=std::max(largest,score);
            scores[j]=score;
        }
        if(c.count){
            T sum=0;
            for(TI j=0;j<N;++j)
                if(c.valid[j]){
                    c.probabilities[j]=std::exp(scores[j]-largest);
                    sum+=c.probabilities[j];
                }
            for(TI j=0;j<N;++j)
                if(c.valid[j]){
                    c.probabilities[j]/=sum;
                    for(TI d=0;d<D;++d)
                        c.context[d]+=c.probabilities[j]*c.features[j][d];
                }
            for(TI d=0;d<D;++d)c.mean[d]/=T(c.count);
        }
        for(TI d=0;d<D;++d){
            rlt::set(dst,row,offset+d,c.context[d]);
            rlt::set(dst,row,offset+D+d,c.mean[d]);
            rlt::set(dst,row,offset+2*D+d,c.count?c.max[d]:T(0));
        }
        return c;
    }
    template<typename IN,typename DERIV>
    void Differentiate(IN const& input,TI row,TI offset,
                       DERIV const& deriv,Cache const& c) {
        if(!c.count)return;
        Vector dq{};
        T const scale=T(1)/std::sqrt(T(D));
        for(TI j=0;j<N;++j){
            if(!c.valid[j])continue;
            T dot=0;
            for(TI d=0;d<D;++d)
                dot+=rlt::get(deriv,row,offset+d)*(c.features[j][d]-c.context[d]);
            T const dscore=c.probabilities[j]*dot*scale;
            for(TI d=0;d<D;++d){
                TI ties=0;
                for(TI k=0;k<N;++k)
                    if(c.valid[k]&&c.features[k][d]==c.max[d])++ties;
                T const grad=c.probabilities[j]*rlt::get(deriv,row,offset+d)
                    +dscore*c.query[d]
                    +rlt::get(deriv,row,offset+D+d)/T(c.count)
                    +(c.features[j][d]==c.max[d]
                       ?rlt::get(deriv,row,offset+2*D+d)/T(ties):T(0));
                T const dz=grad*(T(1)-c.features[j][d]*c.features[j][d]);
                rlt::increment(encoderB.gradient,0,d,dz);
                for(TI i=0;i<F;++i)
                    rlt::increment(encoderW.gradient,d,i,
                        dz*rlt::get(input,row,offset+j*F+i));
                dq[d]+=dscore*c.features[j][d];
            }
        }
        for(TI d=0;d<D;++d){
            T const dz=dq[d]*(T(1)-c.query[d]*c.query[d]);
            rlt::increment(queryB.gradient,0,d,dz);
            for(TI i=0;i<ObservationLayout::Own;++i)
                rlt::increment(queryW.gradient,d,i,
                               dz*rlt::get(input,row,i));
        }
    }
};

template<typename SPEC>
struct AttentionActor : rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC> {
    using Base=rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<SPEC>;
    using T=typename SPEC::T;
    using TI=typename SPEC::TI;
    template<TI B,typename TAG=rlt::MatrixDynamicTag>
    using Buffer=typename Base::template Buffer<B,TAG>;
    TrackAttention<T,TI,ObservationLayout::RedFeatures,ObservationLayout::RedTracks,8> red;
    TrackAttention<T,TI,ObservationLayout::FriendlyFeatures,ObservationLayout::FriendlyTracks,6> friendlies;
};
} // namespace ppo::examples::twovstwo::defenders

namespace rl_tools {
namespace blue_attention_detail {
template<typename DEVICE,typename BLOCK> void malloc(DEVICE& d,BLOCK& b){
    rl_tools::malloc(d,b.encoderW);rl_tools::malloc(d,b.encoderB);
    rl_tools::malloc(d,b.queryW);rl_tools::malloc(d,b.queryB);
}
template<typename DEVICE,typename BLOCK> void free(DEVICE& d,BLOCK& b){
    rl_tools::free(d,b.encoderW);rl_tools::free(d,b.encoderB);
    rl_tools::free(d,b.queryW);rl_tools::free(d,b.queryB);
}
template<typename DEVICE,typename BLOCK> void zero(DEVICE& d,BLOCK& b){
    rl_tools::set_all(d,b.encoderW.gradient,0);rl_tools::set_all(d,b.encoderB.gradient,0);
    rl_tools::set_all(d,b.queryW.gradient,0);rl_tools::set_all(d,b.queryB.gradient,0);
}
template<typename DEVICE,typename BLOCK,typename OPT>
void reset(DEVICE& d,BLOCK& b,OPT& opt){
    rl_tools::_reset_optimizer_state(d,b.encoderW,opt);
    rl_tools::_reset_optimizer_state(d,b.encoderB,opt);
    rl_tools::_reset_optimizer_state(d,b.queryW,opt);
    rl_tools::_reset_optimizer_state(d,b.queryB,opt);
}
template<typename DEVICE,typename BLOCK,typename OPT>
void update(DEVICE& d,BLOCK& b,OPT& opt){
    rl_tools::update(d,b.encoderW,opt);rl_tools::update(d,b.encoderB,opt);
    rl_tools::update(d,b.queryW,opt);rl_tools::update(d,b.queryB,opt);
}
template<typename DEVICE,typename BLOCK,typename RNG>
void initialize(DEVICE& d,BLOCK& b,RNG& rng){
    auto fill=[&](auto& m,bool weight){
        using MATRIX=std::remove_reference_t<decltype(m)>;
        for(typename DEVICE::index_t i=0;i<MATRIX::SPEC::ROWS;++i)
            for(typename DEVICE::index_t j=0;j<MATRIX::SPEC::COLS;++j)
                rl_tools::set(m,i,j,weight
                    ?rl_tools::random::uniform_real_distribution(
                        typename DEVICE::SPEC::RANDOM(),-0.05F,0.05F,rng):0.F);
    };
    fill(b.encoderW.parameters,true);fill(b.encoderB.parameters,false);
    fill(b.queryW.parameters,true);fill(b.queryB.parameters,false);
}
template<typename DEVICE,typename ACTOR,typename IN,typename OUT>
void transform(DEVICE& d,ACTOR const& actor,IN const& input,OUT& output){
    rl_tools::copy(d,d,input,output);
    using L=ppo::examples::twovstwo::defenders::ObservationLayout;
    for(typename ACTOR::TI row=0;row<IN::SPEC::ROWS;++row){
        actor.red.Transform(input,row,L::RedOffset,output);
        actor.friendlies.Transform(input,row,L::FriendOffset,output);
        for(typename ACTOR::TI j=L::RedOffset+24;j<L::FriendOffset;++j)
            rl_tools::set(output,row,j,0.F);
    }
}
} // namespace blue_attention_detail

template<typename DEVICE,typename SPEC>
void malloc(DEVICE& d,ppo::examples::twovstwo::defenders::AttentionActor<SPEC>& a){
    using B=typename ppo::examples::twovstwo::defenders::AttentionActor<SPEC>::Base;
    rl_tools::malloc(d,static_cast<B&>(a));
    blue_attention_detail::malloc(d,a.red);blue_attention_detail::malloc(d,a.friendlies);
}
template<typename DEVICE,typename SPEC>
void free(DEVICE& d,ppo::examples::twovstwo::defenders::AttentionActor<SPEC>& a){
    using B=typename ppo::examples::twovstwo::defenders::AttentionActor<SPEC>::Base;
    blue_attention_detail::free(d,a.red);blue_attention_detail::free(d,a.friendlies);
    rl_tools::free(d,static_cast<B&>(a));
}
template<typename DEVICE,typename SPEC,typename RNG>
void init_weights(DEVICE& d,ppo::examples::twovstwo::defenders::AttentionActor<SPEC>& a,RNG& rng){
    using B=typename ppo::examples::twovstwo::defenders::AttentionActor<SPEC>::Base;
    rl_tools::init_weights(d,static_cast<B&>(a),rng);
    blue_attention_detail::initialize(d,a.red,rng);
    blue_attention_detail::initialize(d,a.friendlies,rng);
}
template<typename DEVICE,typename SPEC,typename OPT>
void _reset_optimizer_state(DEVICE& d,ppo::examples::twovstwo::defenders::AttentionActor<SPEC>& a,OPT& opt){
    using B=typename ppo::examples::twovstwo::defenders::AttentionActor<SPEC>::Base;
    rl_tools::_reset_optimizer_state(d,static_cast<B&>(a),opt);
    blue_attention_detail::reset(d,a.red,opt);blue_attention_detail::reset(d,a.friendlies,opt);
}
template<typename DEVICE,typename SPEC>
void zero_gradient(DEVICE& d,ppo::examples::twovstwo::defenders::AttentionActor<SPEC>& a){
    using B=typename ppo::examples::twovstwo::defenders::AttentionActor<SPEC>::Base;
    rl_tools::zero_gradient(d,static_cast<B&>(a));
    blue_attention_detail::zero(d,a.red);blue_attention_detail::zero(d,a.friendlies);
}
template<typename DEVICE,typename SPEC,typename OPT>
void update(DEVICE& d,ppo::examples::twovstwo::defenders::AttentionActor<SPEC>& a,OPT& opt){
    using B=typename ppo::examples::twovstwo::defenders::AttentionActor<SPEC>::Base;
    rl_tools::update(d,static_cast<B&>(a),opt);
    blue_attention_detail::update(d,a.red,opt);blue_attention_detail::update(d,a.friendlies,opt);
}
template<typename DEVICE,typename SPEC,typename IN_SPEC,typename OUT_SPEC,typename BUF>
void evaluate(DEVICE& d,ppo::examples::twovstwo::defenders::AttentionActor<SPEC> const& a,
              Matrix<IN_SPEC> const& input,Matrix<OUT_SPEC>& output,BUF& buffer){
    MatrixDynamic<matrix::Specification<typename SPEC::T,typename SPEC::TI,
        IN_SPEC::ROWS,IN_SPEC::COLS>> transformed{};
    rl_tools::malloc(d,transformed);
    blue_attention_detail::transform(d,a,input,transformed);
    using MLP=nn_models::mlp::NeuralNetwork<SPEC>;
    rl_tools::evaluate(d,static_cast<MLP const&>(a),transformed,output,buffer);
    rl_tools::free(d,transformed);
}
template<typename DEVICE,typename SPEC,typename IN_SPEC,typename OUT_SPEC>
void forward(DEVICE& d,ppo::examples::twovstwo::defenders::AttentionActor<SPEC>& a,
             Matrix<IN_SPEC> const& input,Matrix<OUT_SPEC>& output){
    MatrixDynamic<matrix::Specification<typename SPEC::T,typename SPEC::TI,
        IN_SPEC::ROWS,IN_SPEC::COLS>> transformed{};
    rl_tools::malloc(d,transformed);
    blue_attention_detail::transform(d,a,input,transformed);
    using MLP=nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>;
    rl_tools::forward(d,static_cast<MLP&>(a),transformed,output);
    rl_tools::free(d,transformed);
}
template<typename DEVICE,typename SPEC,typename IN_SPEC,typename OUT_SPEC,typename BUF>
void backward(DEVICE& d,ppo::examples::twovstwo::defenders::AttentionActor<SPEC>& a,
              Matrix<IN_SPEC> const& input,Matrix<OUT_SPEC>& d_output,BUF& buffer){
    using T=typename SPEC::T;using TI=typename SPEC::TI;
    MatrixDynamic<matrix::Specification<T,TI,IN_SPEC::ROWS,IN_SPEC::COLS>> transformed{},d_input{};
    rl_tools::malloc(d,transformed);rl_tools::malloc(d,d_input);
    blue_attention_detail::transform(d,a,input,transformed);
    using MLP=nn_models::mlp::NeuralNetworkBackwardGradient<SPEC>;
    rl_tools::backward_full(d,static_cast<MLP&>(a),transformed,d_output,d_input,buffer);
    using L=ppo::examples::twovstwo::defenders::ObservationLayout;
    for(TI row=0;row<IN_SPEC::ROWS;++row){
        auto rc=a.red.Transform(input,row,L::RedOffset,transformed);
        a.red.Differentiate(input,row,L::RedOffset,d_input,rc);
        auto fc=a.friendlies.Transform(input,row,L::FriendOffset,transformed);
        a.friendlies.Differentiate(input,row,L::FriendOffset,d_input,fc);
    }
    rl_tools::free(d,d_input);rl_tools::free(d,transformed);
}
} // namespace rl_tools

// Adam's generic step/reset templates are defined upstream before this actor.
// Put the exact-type overloads in the actor's associated namespace so C++
// argument-dependent lookup reaches the attention parameters as well as MLP.
namespace ppo::examples::twovstwo::defenders {
template<typename DEVICE,typename SPEC,typename OPT>
void update(DEVICE& d,AttentionActor<SPEC>& actor,OPT& optimizer)
{
    rl_tools::update(d,actor,optimizer);
}
template<typename DEVICE,typename SPEC,typename OPT>
void _reset_optimizer_state(DEVICE& d,AttentionActor<SPEC>& actor,OPT& optimizer)
{
    rl_tools::_reset_optimizer_state(d,actor,optimizer);
}
} // namespace ppo::examples::twovstwo::defenders
