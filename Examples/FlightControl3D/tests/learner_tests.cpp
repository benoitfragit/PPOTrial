#include "test_support.hpp"
#include "flight_learner.hpp"
#include <fstream>
#include <iomanip>
using namespace ppo::examples::flight;

namespace {
void Hybrid(Config const& c)
{
    auto settings=c.training;
    settings.entropy=0.001; // Keep entropy derivatives covered independently of the scenario default.
    PolicyVector output{};
    for (std::size_t i=0;i<ContinuousActions;++i)
    {
        output[i]=0.1*static_cast<double>(i);
        output[ContinuousActions+i]=0.5*(settings.minLogStd+settings.maxLogStd);
    }
    output.back()=0.3;
    std::mt19937 rng(4);
    auto a=Sample(output,settings,rng,false);
    Check(Near(LogProbability(output,a,settings),a.logProbability),"joint sampled log probability");
    auto other=a; other.gearDown=!a.gearDown;
    Check(!Near(LogProbability(output,other,settings),a.logProbability),"Bernoulli included in joint likelihood");
    for (double advantage : {-1.7,1.4})
    {
        for (double ratio : {0.6,1.0,1.5})
        {
            a.logProbability=LogProbability(output,a,settings)-std::log(ratio);
            auto const loss=ClippedLoss(output,a,advantage,settings);
            for (std::size_t i=0;i<PolicyOutputs;++i)
            {
                auto plus=output,minus=output;
                plus[i]+=1e-6; minus[i]-=1e-6;
                double const numerical=(ClippedLoss(plus,a,advantage,settings).loss-ClippedLoss(minus,a,advantage,settings).loss)/2e-6;
                Check(Near(numerical,loss.derivative[i],1e-5),"hybrid finite-difference PPO gradient");
            }
        }
    }
    for(double logit : {-1000.,1000.})
    {
        output.back()=logit;
        Check(std::isfinite(LogProbability(output,a,settings)) && std::isfinite(Entropy(output,settings)),"stable extreme Bernoulli logits");
    }
    // The environment clips raw Gaussian actions, just like BlueAgent.
    a.raw={-1000,1000,0,-1000,1000};
    auto const controls=ToControls(a,c.initial.throttle);
    Check(controls.elevator==-1 && controls.aileron==1 && controls.rudder==0
        && controls.throttle==0 && controls.brake==1,"clamped Gaussian controls");
    a.raw={0.25,-0.5,0.75,0,0};
    auto const neutral=ToControls(a,c.initial.throttle);
    Check(Near(neutral.elevator,0.25) && Near(neutral.aileron,-0.5)
        && Near(neutral.rudder,0.75) && Near(neutral.throttle,c.initial.throttle)
        && neutral.brake==0,"linear flight controls and reference throttle");
    a.raw[3]=-0.5;
    Check(Near(ToControls(a,c.initial.throttle).throttle,c.initial.throttle*0.5),
        "negative throttle commands span the range to idle");
    a.raw[3]=0.5;
    Check(Near(ToControls(a,c.initial.throttle).throttle,(1+c.initial.throttle)*0.5),
        "positive throttle commands span the range to full");
    a.raw[4]=0.5;
    Check(Near(ToControls(a,c.initial.throttle).brake,0.5),"positive brake commands are linear");
    std::array<Transition,3> rows{};
    rows[0].reward=1; rows[0].value=2; rows[0].nextValue=3; rows[0].truncated=true;
    rows[1].reward=1000; rows[1].terminated=true;
    rows[2].reward=1; rows[2].nextValue=5;
    ComputeAdvantages(rows.data(),rows.size(),c.training);
    Check(Near(rows[0].advantage,1+c.training.gamma*3-2),"timeout bootstrap; no GAE leak across reset");
    Check(Near(rows[1].advantage,1000),"terminal no bootstrap");
}
void Exploration(Config const& c)
{
    FlightLearner learner(c);
    auto const output=learner.Evaluate({});
    Check(std::exp(c.training.initialLogStd)<0.051 && std::exp(c.training.maxLogStd)<0.14,
        "delivered flight scenario bounds exploration below full-deflection noise");
    for(std::size_t axis=0;axis<ContinuousActions;++axis)
        Check(Near(output[axis],0) && Near(output[ContinuousActions+axis],c.training.initialLogStd),
            "initial actor retains full linear control authority with the configured small Gaussian");
    auto saturated=output;
    for(std::size_t axis=0;axis<ContinuousActions;++axis) saturated[ContinuousActions+axis]=100;
    std::mt19937 rng(5243U);
    double square=0;
    constexpr std::size_t samples=2048;
    for(std::size_t sample=0;sample<samples;++sample)
    {
        auto const action=Sample(saturated,c.training,rng,false);
        Check(std::isfinite(action.logProbability),"bounded exploration keeps finite latent likelihoods");
        square+=action.raw[0]*action.raw[0];
        for(double value:action.raw)
            Check(std::abs(value)<1,"fixed-seed bounded exploration does not clip neutral flight controls");
    }
    double const expected=std::exp(c.training.maxLogStd);
    Check(std::abs(std::sqrt(square/static_cast<double>(samples))-expected)<0.01,
        "saturated log-std output obeys the configured exploration ceiling");
}
void ActiveControls(Config c)
{
    c.training.entropy=0.01;
    PolicyVector output{};
    for(std::size_t i=0;i<ContinuousActions;++i) output[ContinuousActions+i]=c.training.initialLogStd;
    std::mt19937 rng(91U);
    for(bool gearActive:{false,true}) for(bool brakeActive:{false,true})
    {
        auto const action=Sample(output,c.training,rng,false,brakeActive,gearActive,false);
        auto changed=output;
        if(!brakeActive) {changed[4]=100;changed[ContinuousActions+4]=100;}
        if(!gearActive) changed.back()=100;
        Check(Near(LogProbability(changed,action,c.training),action.logProbability),
            "ignored controls cannot change the PPO likelihood ratio");
        auto const loss=ClippedLoss(output,action,1.2,c.training);
        for(std::size_t axis=0;axis<PolicyOutputs;++axis)
        {
            auto plus=output,minus=output;plus[axis]+=1e-6;minus[axis]-=1e-6;
            double const numerical=(ClippedLoss(plus,action,1.2,c.training).loss
                -ClippedLoss(minus,action,1.2,c.training).loss)/2e-6;
            Check(Near(numerical,loss.derivative[axis],1e-5),"masked PPO and entropy gradients match finite differences");
        }
        if(!brakeActive) Check(action.raw[4]==0 && loss.derivative[4]==0
            && loss.derivative[ContinuousActions+4]==0,"airborne brakes are neither sampled nor optimized");
        if(!gearActive) Check(!action.gearDown && loss.derivative.back()==0,"fixed capture gear is neither sampled nor optimized");
    }
    for(auto stage:{CurriculumStage::Loc,CurriculumStage::Glide,CurriculumStage::Joint})
    {
        c.training.curriculumStage=stage;
        FlightLearner learner(c);Observation observation{};
        for(int sample=0;sample<32;++sample)
        {
            auto const action=learner.Choose(observation);
            Check(!action.brakeActive && action.raw[4]==0
                && action.gearActive==(stage==CurriculumStage::Joint)
                && (action.gearActive || action.gearDown==c.initial.gearDown),
                "native learner selects action masks from lesson and physical phase");
        }
        observation[RolloutObservation]=1;
        Check(learner.Choose(observation).brakeActive,"brake learning becomes active during rollout");
    }
}
void Training(Config c)
{
    TemporaryDirectory temporary("flight-learner-"); auto const path=temporary.path/"flight.txt";
    c.training.rolloutSteps=16; c.training.minibatch=8; c.training.epochs=2;
    FlightLearner learner(c); FlightEnvironment environment(c);
    auto observation=environment.Reset();
    auto const original=learner.Evaluate(observation);
    double rewardSum{};
    std::array<double,RewardCount> rewardSums{};
    for (std::size_t i=0;i<c.training.rolloutSteps;++i)
    {
        auto const action=learner.Choose(observation);
        auto const result=environment.Step(ToControls(action,c.initial.throttle));
        auto const next=environment.ObservationNow();
        learner.Record(observation,action,result,next);
        rewardSum+=result.reward;
        for(std::size_t channel=0;channel<RewardCount;++channel)
            rewardSums[channel]+=result.rewards[channel];
        observation=(result.terminated || result.truncated) ? environment.Reset() : next;
    }
    Check(learner.Ready(),"rollout filled");
    Rejects([&] { learner.Save(path); },"incomplete checkpoint boundary");
    learner.Optimize();
    Check(learner.Stats().updates==1 && std::isfinite(learner.Stats().valueLoss),"real RLtools PPO update");
    Check(Near(learner.Stats().meanReward,rewardSum/static_cast<double>(c.training.rolloutSteps)),
        "rollout mean reward diagnostic");
    for(std::size_t channel=0;channel<RewardCount;++channel)
        Check(Near(learner.Stats().meanRewards[channel],
            rewardSums[channel]/static_cast<double>(c.training.rolloutSteps)),
            "rollout mean reward-channel diagnostic");
    Check(learner.Evaluate(observation)!=original,"actor changes after PPO");
    learner.Save(path);
    {
        std::ifstream checkpoint(path);
        for(std::string line;std::getline(checkpoint,line);)
            Check(line.empty() || (line.back()!=' ' && line.back()!='\t'),
                "checkpoint lines have no trailing whitespace");
    }
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
void KlStopsActorOnly(Config c)
{
    c.training.hiddenWidth=32;c.training.rolloutSteps=128;c.training.minibatch=16;
    c.training.epochs=4;c.training.actorRate=0.01;c.training.targetKl=1e-6;
    FlightLearner learner(c);Observation observation{};
    for(std::size_t row=0;row<c.training.rolloutSteps;++row)
    {
        auto const action=learner.Choose(observation);
        StepResult step{};step.terminated=true;step.reward=action.raw[0];
        learner.Record(observation,action,step,observation);
    }
    learner.Optimize();auto const& stats=learner.Stats();
    Check(stats.actorStoppedByKl && stats.maximumApproximateKl>c.training.targetKl
        && stats.actorSteps==1 && stats.updates==1,
        "excessive likelihood drift stops subsequent actor steps within the rollout");
    TemporaryDirectory directory("flight-kl-");auto const path=directory.path/"checkpoint.txt";
    learner.Save(path);
    std::ifstream stream(path);std::string line,signature;
    std::getline(stream,line);stream>>std::quoted(signature);
    std::size_t actorAge{},criticAge{};stream>>actorAge>>criticAge;
    Check(actorAge==1+stats.actorSteps
        && criticAge==1+c.training.epochs*c.training.rolloutSteps/c.training.minibatch,
        "KL stopping freezes actor Adam age while all critic minibatches run");
    FlightLearner resumed(c);resumed.Load(path);
    Check(resumed.Evaluate(observation)==learner.Evaluate(observation),"KL-stopped checkpoint resumes exactly");
    c.training.targetKl=0;Rejects([&] {Validate(c);},"zero target KL rejected");
    c.training.targetKl=std::numeric_limits<double>::infinity();
    Rejects([&] {Validate(c);},"non-finite target KL rejected");
}
}
int main()
{
    try { auto const c=LoadConfig(PPO_FLIGHT_CONFIG); Hybrid(c); Exploration(c); ActiveControls(c); Training(c); KlStopsActorOnly(c); }
    catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
    std::cout<<"Hybrid PPO/RLtools regressions passed\n";
}
