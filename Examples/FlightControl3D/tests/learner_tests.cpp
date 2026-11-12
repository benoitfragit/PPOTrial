#include "test_support.hpp"
#include "flight_learner.hpp"
#include <fstream>
using namespace ppo::examples::flight;

namespace {
void Hybrid(Config const& c)
{
    PolicyVector output{};
    for (std::size_t i=0;i<ContinuousActions;++i) { output[i]=0.1*static_cast<double>(i); output[ContinuousActions+i]=-0.5; }
    output.back()=0.3;
    std::mt19937 rng(4);
    auto a=Sample(output,c.training,rng,false);
    Check(Near(LogProbability(output,a,c.training),a.logProbability),"joint sampled log probability");
    auto other=a; other.gearDown=!a.gearDown;
    Check(!Near(LogProbability(output,other,c.training),a.logProbability),"Bernoulli included in joint likelihood");
    for (double advantage : {-1.7,1.4})
    {
        for (double ratio : {0.6,1.0,1.5})
        {
            a.logProbability=LogProbability(output,a,c.training)-std::log(ratio);
            auto const loss=ClippedLoss(output,a,advantage,c.training);
            for (std::size_t i=0;i<PolicyOutputs;++i)
            {
                auto plus=output,minus=output;
                plus[i]+=1e-6; minus[i]-=1e-6;
                double const numerical=(ClippedLoss(plus,a,advantage,c.training).loss-ClippedLoss(minus,a,advantage,c.training).loss)/2e-6;
                Check(Near(numerical,loss.derivative[i],1e-5),"hybrid finite-difference PPO gradient");
            }
        }
    }
    for(double logit : {-1000.,1000.})
    {
        output.back()=logit;
        Check(std::isfinite(LogProbability(output,a,c.training)) && std::isfinite(Entropy(output,c.training)),"stable extreme Bernoulli logits");
    }
    a.raw={-1000,1000,0,-1000,1000};
    auto const controls=ToControls(a);
    Check(controls.elevator==-1 && controls.aileron==1 && controls.throttle==0 && controls.brake==1,"bounded Gaussian transform");
    std::array<Transition,3> rows{};
    rows[0].reward=1; rows[0].value=2; rows[0].nextValue=3; rows[0].truncated=true;
    rows[1].reward=1000; rows[1].terminated=true;
    rows[2].reward=1; rows[2].nextValue=5;
    ComputeAdvantages(rows.data(),rows.size(),c.training);
    Check(Near(rows[0].advantage,1+c.training.gamma*3-2),"timeout bootstrap; no GAE leak across reset");
    Check(Near(rows[1].advantage,1000),"terminal no bootstrap");
}
void Training(Config c)
{
    TemporaryDirectory temporary("flight-learner-"); auto const path=temporary.path/"flight.txt";
    c.training.rolloutSteps=16; c.training.minibatch=8; c.training.epochs=2;
    FlightLearner learner(c); FlightEnvironment environment(c);
    auto observation=environment.Reset();
    auto const original=learner.Evaluate(observation);
    for (std::size_t i=0;i<c.training.rolloutSteps;++i)
    {
        auto const action=learner.Choose(observation);
        auto const result=environment.Step(ToControls(action));
        auto const next=environment.ObservationNow();
        learner.Record(observation,action,result,next);
        observation=(result.terminated || result.truncated) ? environment.Reset() : next;
    }
    Check(learner.Ready(),"rollout filled");
    Rejects([&] { learner.Save(path); },"incomplete checkpoint boundary");
    learner.Optimize();
    Check(learner.Stats().updates==1 && std::isfinite(learner.Stats().valueLoss),"real RLtools PPO update");
    Check(learner.Evaluate(observation)!=original,"actor changes after PPO");
    learner.Save(path);
    FlightLearner restored(c); restored.Load(path);
    Check(learner.Evaluate(observation)==restored.Evaluate(observation),"checkpoint exact actor roundtrip");
    auto const a=learner.Choose(observation), b=restored.Choose(observation);
    Check(a.raw==b.raw && a.gearDown==b.gearDown && a.logProbability==b.logProbability,"checkpoint exact sampling/RNG roundtrip");
    for (std::size_t i=0;i<c.training.rolloutSteps;++i)
    {
        auto const action=learner.Choose(observation), resumed=restored.Choose(observation);
        Check(action.raw==resumed.raw,"resumed RNG");
        StepResult feedback{}; feedback.reward=static_cast<double>(i%3);
        learner.Record(observation,action,feedback,observation);
        restored.Record(observation,resumed,feedback,observation);
    }
    learner.Optimize(); restored.Optimize();
    Check(learner.Evaluate(observation)==restored.Evaluate(observation),"optimizer moments and age resumed exactly");
    auto const preserved=restored.Evaluate(observation);
    { std::ofstream stream(path); stream<<"broken"; }
    Rejects([&] { restored.Load(path); },"corrupt checkpoint accepted");
    Check(restored.Evaluate(observation)==preserved,"corrupt load is transactional");
    observation[0]=2;
    Rejects([&] { learner.Evaluate(observation); },"unnormalized observation accepted");
    for (std::size_t width : {32U,128U}) { c.training.hiddenWidth=width; FlightLearner variant(c); variant.Evaluate({}); }
}
}
int main()
{
    try { auto const c=LoadConfig(PPO_FLIGHT_CONFIG); Hybrid(c); Training(c); }
    catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
    std::cout<<"Hybrid PPO/RLtools regressions passed\n";
}
