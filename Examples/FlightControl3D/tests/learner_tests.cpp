/**
 * @file learner_tests.cpp
 * @brief Regressions of the PPO learner: Gaussian policy, gradients, KL guard and checkpoints.
 *
 * Protects: the portable RNG checkpoint (libstdc++ and MSVC streams), the finite-difference
 * check of the clipped PPO gradients, exploration, the inactive brake axis in flight,
 * observation bounds, the KL guard and its frozen actor state, the analytic KL and the
 * trimmed neutral approach mapping. Also links a second RLtools backend to check that the
 * namespace wrapper avoids symbol clashes.
 *
 * Read next: Training/flight_policy.hpp, Training/flight_learner.cpp.
 */
#include "test_support.hpp"
#include "flight_learner.hpp"
#include "flight_rng_checkpoint.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
using namespace ppo::examples::flight;

namespace {
void PortableRngCheckpoint()
{
    constexpr std::size_t count=std::mt19937::state_size;
    std::array<std::uint32_t,count> seeded{};
    seeded[0]=5489U;
    for(std::size_t i=1;i<count;++i)
        seeded[i]=1812433253U*(seeded[i-1]^(seeded[i-1]>>30U))+static_cast<std::uint32_t>(i);
    auto block=seeded;
    for(std::size_t i=0;i<count;++i)
    {
        auto const joined=(block[i]&0x80000000U)|(block[(i+1)%count]&0x7fffffffU);
        block[i]=block[(i+397)%count]^(joined>>1U)^((joined&1U) ? 0x9908b0dfU : 0U);
    }
    for(std::size_t position:{0U,1U,227U,501U,623U,624U})
    {
        std::ostringstream encoded;
        for(auto const value:block) encoded<<value<<' ';
        encoded<<position<<'\n';
        std::istringstream input(encoded.str());
        auto actual=detail::ReadCheckpointRng(input);
        std::mt19937 expected(5489U);
        expected.discard(position);
        for(std::size_t i=0;i<1500;++i)
            Check(actual()==expected(),"libstdc++ RNG state preserves the full future sequence");
    }
    std::ostringstream encoded;
    for(auto const value:seeded) encoded<<value<<' ';
    std::istringstream input(encoded.str());
    auto actual=detail::ReadCheckpointRng(input);
    std::mt19937 expected(5489U);
    for(std::size_t i=0;i<1500;++i)
        Check(actual()==expected(),"MSVC RNG state preserves the full future sequence");
    std::istringstream corrupt("1 2 3\n");
    Rejects([&] { detail::ReadCheckpointRng(corrupt); },"truncated RNG state is rejected");
}
void GaussianPolicy(Config const& c)
{
    auto settings=c.training;
    settings.entropy=0.001; // Keep entropy derivatives covered independently of the scenario default.
    PolicyDistribution policy{};
    for (std::size_t i=0;i<ContinuousActions;++i)
    {
        policy.output[i]=0.1*static_cast<double>(i);
        policy.logStandardDeviations[i]=-0.4+0.05*static_cast<double>(i);
    }
    std::mt19937 rng(4);
    auto a=Sample(policy,rng,false);
    std::mt19937 noiseA(42),noiseB(42);
    auto const first=Sample(policy,noiseA,false);
    auto const firstReplica=Sample(policy,noiseB,false);
    auto const second=Sample(policy,noiseA,false);
    auto const secondReplica=Sample(policy,noiseB,false);
    Check(first.raw==firstReplica.raw && second.raw==secondReplica.raw && first.raw!=second.raw,
        "successive exploration samples vary while a seed reproduces the sequence");
    Check(Near(LogProbability(policy,a),a.logProbability),"sampled log probability");
    for (double advantage : {-1.7,1.4})
    {
        for (double ratio : {0.6,1.0,1.5})
        {
            a.logProbability=LogProbability(policy,a)-std::log(ratio);
            auto const loss=ClippedLoss(policy,a,advantage,settings);
            for (std::size_t i=0;i<PolicyOutputs;++i)
            {
                auto plus=policy,minus=policy;
                plus.output[i]+=1e-6; minus.output[i]-=1e-6;
                double const numerical=(ClippedLoss(plus,a,advantage,settings).loss-ClippedLoss(minus,a,advantage,settings).loss)/2e-6;
                Check(Near(numerical,loss.outputDerivative[i],1e-5),"actor-output finite-difference PPO gradient");
            }
            for(std::size_t i=0;i<ContinuousActions;++i)
            {
                auto plus=policy,minus=policy;
                plus.logStandardDeviations[i]+=1e-6;
                minus.logStandardDeviations[i]-=1e-6;
                double const numerical=(ClippedLoss(plus,a,advantage,settings).loss
                    -ClippedLoss(minus,a,advantage,settings).loss)/2e-6;
                Check(Near(numerical,loss.logStandardDeviationDerivative[i],1e-5),
                    "global log-standard-deviation finite-difference PPO gradient");
            }
        }
    }
    // The environment clips raw Gaussian actions, just like BlueAgent.
    a.raw={-1000,1000,0,-1000,1000};
    auto const controls=ToControls(a,c.initial);
    Check(controls.elevator==-1 && controls.aileron==1 && controls.rudder==0
        && controls.throttle==0 && controls.brake==1,"clamped Gaussian controls");
    a.raw[0]=1000;
    Check(ToControls(a,c.initial).elevator==1,"the trimmed elevator still reaches full nose-down");
    a.raw={0,0,0,0,0};
    a.brakeActive=false;
    auto const trimmed=ToControls(a,c.initial);
    Check(Near(trimmed.elevator,c.initial.elevatorTrim) && Near(trimmed.throttle,c.initial.throttle)
        && trimmed.aileron==0 && trimmed.rudder==0 && trimmed.brake==0,
        "a zero actor output requests the trimmed approach controls");
    a.raw={0.25,-0.5,0.75,0,0};
    auto const neutral=ToControls(a,c.initial);
    Check(Near(neutral.elevator,c.initial.elevatorTrim+0.25) && Near(neutral.aileron,-0.5)
        && Near(neutral.rudder,0.75) && Near(neutral.throttle,c.initial.throttle)
        && neutral.brake==0,"linear flight controls around the trim");
    a.raw[3]=-0.5;
    Check(Near(ToControls(a,c.initial).throttle,c.initial.throttle*0.5),
        "negative throttle commands span the range to idle");
    a.raw[3]=0.5;
    Check(Near(ToControls(a,c.initial).throttle,(1+c.initial.throttle)*0.5),
        "positive throttle commands span the range to full");
    a.raw[4]=0.5;
    Check(ToControls(a,c.initial).brake==0,"the brake stays released in flight");
    // On the ground the whole raw range maps linearly to the brake: no
    // Gaussian sample is lost in a clamped dead zone, and zero brakes halfway.
    a.brakeActive=true;
    for(auto const& [raw,expected] : std::array<std::pair<double,double>,6>{
        {{-1000,0},{-1,0},{-0.5,0.25},{0,0.5},{0.5,0.75},{1000,1}}})
    {
        a.raw[4]=raw;
        Check(Near(ToControls(a,c.initial).brake,expected),"rollout brake commands span the whole raw range");
    }
    // The mean that previously left the brake clamped to zero now brakes firmly.
    a.raw[4]=-0.19;
    Check(Near(ToControls(a,c.initial).brake,0.405),"a slightly negative brake output still brakes");
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
    auto const policy=learner.Evaluate({});
    for(std::size_t axis=0;axis<ContinuousActions;++axis)
        Check(std::isfinite(policy.output[axis])
            && Near(policy.logStandardDeviations[axis],InitialLogStandardDeviation),
            "the flight actor starts with finite means and a bounded global deviation");
    Observation probe{};probe[0]=0.5F;probe[1]=-0.5F;
    auto const initialized=learner.Evaluate(probe);
    Check(std::any_of(initialized.output.begin(),initialized.output.end(),
        [](double value) { return std::abs(value)>1e-6; }),
        "BlueAgent RLtools initialization keeps the actor output layer trainable");
    std::mt19937 rng(5243U);
    double square=0;
    constexpr std::size_t samples=2048;
    for(std::size_t sample=0;sample<samples;++sample)
    {
        auto const action=Sample(policy,rng,false);
        Check(std::isfinite(action.logProbability),"the global Gaussian keeps finite latent likelihoods");
        double const centered=action.raw[0]-policy.output[0];
        square+=centered*centered;
    }
    Check(std::abs(std::sqrt(square/static_cast<double>(samples))
            -std::exp(InitialLogStandardDeviation))<0.01,
        "flight exploration uses the bounded global Gaussian scale");
}
/** @brief The brake is sampled and optimized only while rolling on the ground. */
void ActiveControls(Config c)
{
    c.training.entropy=0.01;
    PolicyDistribution policy{};
    policy.logStandardDeviations.fill(-0.25);
    std::mt19937 rng(91U);
    for(bool brakeActive:{false,true})
    {
        auto const action=Sample(policy,rng,false,brakeActive);
        auto changed=policy;
        if(!brakeActive)
        {
            changed.output[BrakeAction]=100;
            changed.logStandardDeviations[BrakeAction]=100;
        }
        Check(Near(LogProbability(changed,action),action.logProbability),
            "an ignored control cannot change the PPO likelihood ratio");
        auto const loss=ClippedLoss(policy,action,1.2,c.training);
        for(std::size_t axis=0;axis<PolicyOutputs;++axis)
        {
            auto plus=policy,minus=policy;
            plus.output[axis]+=1e-6;minus.output[axis]-=1e-6;
            double const numerical=(ClippedLoss(plus,action,1.2,c.training).loss
                -ClippedLoss(minus,action,1.2,c.training).loss)/2e-6;
            Check(Near(numerical,loss.outputDerivative[axis],1e-5),"masked actor-output gradients match finite differences");
        }
        if(!brakeActive) Check(action.raw[BrakeAction]==0 && loss.outputDerivative[BrakeAction]==0
            && loss.logStandardDeviationDerivative[BrakeAction]==0,
            "airborne brakes are neither sampled nor optimized");
    }
    FlightLearner learner(c);Observation observation{};
    for(int sample=0;sample<32;++sample)
    {
        auto const action=learner.Choose(observation);
        Check(!action.brakeActive && action.raw[BrakeAction]==0,"the brake is inactive in flight");
    }
    observation[RolloutObservation]=1;
    Check(learner.Choose(observation).brakeActive,"brake learning becomes active during rollout");
}

/** @brief Every observation slot is bounded by one through collection and PPO. */
void ObservationBounds(Config c)
{
    c.training.hiddenWidth=32;c.training.rolloutSteps=2;c.training.minibatch=2;c.training.epochs=1;
    FlightLearner learner(c);
    for(std::size_t index=0;index<ObservationSize;++index)
    {
        Observation bounded{};
        bounded[index]=ObservationBound;learner.Evaluate(bounded);
        bounded[index]=-ObservationBound;learner.Evaluate(bounded);
        bounded[index]=std::nextafter(ObservationBound,std::numeric_limits<float>::infinity());
        Rejects([&] { learner.Evaluate(bounded); },"values beyond the observation bound are rejected");
    }
    Observation invalid{};invalid[2]=std::numeric_limits<float>::quiet_NaN();
    Rejects([&] { learner.Evaluate(invalid); },"non-finite features are rejected");
    Observation saturated{};saturated.fill(ObservationBound);
    for(int step=0;step<2;++step)
    {
        auto const action=learner.Choose(saturated);
        learner.Record(saturated,action,StepResult{},saturated);
    }
    learner.Optimize();
    Check(learner.Stats().updates==1,"saturated observations survive PPO collection and optimization");
}void Training(Config c)
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
        auto const result=environment.Step(ToControls(action,c.initial));
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
    Check(learner.Stats().updates==1
        && learner.Stats().actorSteps<=c.training.epochs*c.training.rolloutSteps/c.training.minibatch
        && learner.Stats().criticSteps==c.training.epochs*c.training.rolloutSteps/c.training.minibatch
        && learner.Stats().actorSteps+learner.Stats().actorKlSkips==learner.Stats().criticSteps
        && std::isfinite(learner.Stats().valueLoss),
        "actor minibatches honor the KL gate while the full critic completes a PPO update");
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
        std::string magic;
        std::size_t observations{},actions{},outputs{},width{};
        std::uint64_t updates{};
        Check(static_cast<bool>(checkpoint>>magic>>observations>>actions>>outputs>>width>>updates)
            && magic=="PPO_FLIGHT_POLICY_V3"
            && observations==ObservationSize && actions==ContinuousActions
            && outputs==PolicyOutputs && width==c.training.hiddenWidth
            && updates==learner.Stats().updates,
            "the checkpoint identifies the Gaussian actor and its global deviations");
        checkpoint.clear();checkpoint.seekg(0);
        for(std::string line;std::getline(checkpoint,line);)
            Check(line.empty() || (line.back()!=' ' && line.back()!='\t'),
                "checkpoint lines have no trailing whitespace");
    }
    FlightLearner restored(c); restored.Load(path);
    Check(learner.Evaluate(observation)==restored.Evaluate(observation),"checkpoint exact actor roundtrip");
    auto const a=learner.Choose(observation), b=restored.Choose(observation);
    Check(a.raw==b.raw && a.logProbability==b.logProbability,"checkpoint exact sampling/RNG roundtrip");
    for (std::size_t i=0;i<c.training.rolloutSteps;++i)
    {
        auto const action=learner.Choose(observation), resumed=restored.Choose(observation);
        Check(action.raw==resumed.raw,"resumed RNG");
        // Identical terminal transitions isolate optimizer restoration from the
        // deliberately non-serialized, previously shuffled minibatch index order.
        StepResult feedback{}; feedback.reward=1; feedback.terminated=true;
        learner.Record(observation,a,feedback,observation);
        restored.Record(observation,b,feedback,observation);
    }
    learner.Optimize(); restored.Optimize();
    Check(learner.Evaluate(observation)==restored.Evaluate(observation),
        "optimizer resume reproduces the actor on order-invariant minibatches");
    auto const restoredPath=temporary.path/"restored-flight.txt";
    learner.Save(path);
    restored.Save(restoredPath);
    Check(SameLearnerCheckpoint(path,restoredPath),
        "optimizer moments, Adam ages, weights and RNG resume exactly");
    auto const preserved=restored.Evaluate(observation);
    { std::ofstream stream(path); stream<<"PPO_FLIGHT_HYBRID_V2 39 5 6 128 1\n"; }
    Rejects([&] { restored.Load(path); },"a previous hybrid checkpoint is rejected");
    Check(restored.Evaluate(observation)==preserved,"an incompatible load is transactional");
    { std::ofstream stream(path); stream<<"broken"; }
    Rejects([&] { restored.Load(path); },"corrupt checkpoint accepted");
    Check(restored.Evaluate(observation)==preserved,"a corrupt load is transactional");
    observation[0]=2;
    Rejects([&] { learner.Evaluate(observation); },"unnormalized observation accepted");
    for (std::size_t width : {32U,128U,256U}) { c.training.hiddenWidth=width; FlightLearner variant(c); variant.Evaluate({}); }
}
/** @brief Guard each actor minibatch like RLMission while retaining full critic training. */
void KlProtection(Config c)
{
    c.training.hiddenWidth=32;c.training.rolloutSteps=128;c.training.minibatch=16;
    c.training.epochs=4;
    for(bool const strict : {false,true})
    {
        c.training.actorRate=strict ? 0.01 : 0.0003;
        c.training.targetKl=strict ? 0.005 : 1.0;
        FlightLearner learner(c);
        TemporaryDirectory directory("flight-kl-protection-");
        auto const after=directory.path/"after.txt";
        std::array<Observation,128> observations{};
        std::array<PolicyAction,128> actions{};
        for(std::size_t row=0;row<c.training.rolloutSteps;++row)
        {
            auto& observation=observations[row];
            observation[0]=static_cast<float>(row)/128.F;observation[1]=-observation[0];
            actions[row]=learner.Choose(observation);
            StepResult step{};step.terminated=true;
            step.reward=actions[row].raw[0]+0.3*actions[row].raw[1];
            learner.Record(observation,actions[row],step,observation);
        }
        learner.Optimize();auto const& stats=learner.Stats();
        auto const minibatches=c.training.epochs*c.training.rolloutSteps/c.training.minibatch;
        Check(stats.updates==1 && stats.criticSteps==minibatches
            && stats.actorSteps+stats.actorKlSkips==minibatches
            && std::isfinite(stats.maximumSampledKl),
            "every actor minibatch is applied or skipped while the critic completes");
        if(strict)
            Check(stats.actorKlSkips>0 && stats.actorSteps<minibatches
                && stats.maximumSampledKl>=1.5*c.training.targetKl,
                "a high sampled minibatch KL skips at least one actor update");
        else
            Check(stats.actorKlSkips==0 && stats.actorSteps==minibatches,
                "permitted minibatches all update the actor");
        learner.Save(after);
        std::ifstream stream(after);std::string line,signature;
        std::getline(stream,line);stream>>std::quoted(signature);
        std::size_t actorAge{},criticAge{};stream>>actorAge>>criticAge;
        Check(actorAge==1+stats.actorSteps && criticAge==1+minibatches,
            "skipped actor batches do not advance the actor Adam age while the critic advances");
        FlightLearner resumed(c);resumed.Load(after);
        Check(resumed.Evaluate(observations.back())==learner.Evaluate(observations.back()),
            "a KL-protected actor checkpoint resumes exactly");
    }
}
/** @brief Read serialized actor weights, deviations and their Adam moments. */
std::string CheckpointActorBlock(std::filesystem::path const& path)
{
    std::ifstream input(path);std::string line,result;
    for(int i=0;i<4;++i) Check(static_cast<bool>(std::getline(input,line)),"checkpoint header");
    // Three layers of weights/biases, three Adam matrices each, and log-std matrices.
    for(int i=0;i<21;++i)
    {
        Check(static_cast<bool>(std::getline(input,line)),"checkpoint actor block");
        result+=line+'\n';
    }
    return result;
}
/** @brief A rejected minibatch must never update actor parameters or Adam state. */
void KlRejectedActorState(Config c)
{
    c.training.hiddenWidth=32;c.training.rolloutSteps=8;c.training.minibatch=4;c.training.epochs=2;
    FlightLearner learner(c);Observation observation{};
    TemporaryDirectory directory("flight-kl-rejected-state-");
    auto const before=directory.path/"before.txt",after=directory.path/"after.txt";
    learner.Save(before);
    auto action=learner.Choose(observation,true);
    // Synthetic stale log probabilities force every sampled KL above the threshold.
    action.logProbability+=1.0;
    StepResult step{};step.reward=1;step.terminated=true;
    for(std::size_t i=0;i<c.training.rolloutSteps;++i)
        learner.Record(observation,action,step,observation);
    learner.Optimize();learner.Save(after);
    auto const& stats=learner.Stats();
    Check(stats.actorSteps==0 && stats.actorKlSkips==4 && stats.criticSteps==4
        && Near(stats.maximumSampledKl,1.0,1e-5),
        "a sampled KL above 1.5 times the target skips all actor steps but keeps critic steps");
    Check(CheckpointActorBlock(before)==CheckpointActorBlock(after),
        "skipped actor updates leave weights, log deviations and Adam moments identical");
    std::ifstream checkpoint(after);std::string line,signature;
    std::getline(checkpoint,line);checkpoint>>std::quoted(signature);
    std::size_t actorAge{},criticAge{};checkpoint>>actorAge>>criticAge;
    Check(actorAge==1 && criticAge==5,"skipping never advances the actor Adam age");
}
void AnalyticKl()
{
    PolicyDistribution previous{};
    previous.logStandardDeviations.fill(-0.5);
    PolicyAction active{};
    Check(PolicyKl(previous,previous,active)==0,"identical policies have zero exact KL");
    auto changed=previous;changed.output[0]=0.01;
    double const expected=0.01*0.01/(2*std::exp(-1.0));
    Check(Near(PolicyKl(previous,changed,active),expected,1e-12),
        "the Gaussian mean KL agrees with the closed-form univariate value");
    changed=previous;changed.logStandardDeviations[0]=-0.4;
    Check(Near(PolicyKl(previous,changed,active),0.1+0.5*(std::exp(-0.2)-1),1e-12),
        "the Gaussian variance KL uses the old-to-new direction");
    active.brakeActive=false;
    changed=previous;changed.output[BrakeAction]=100;changed.logStandardDeviations[BrakeAction]=100;
    Check(PolicyKl(previous,changed,active)==0,"an inactive control cannot consume the KL budget");
}
/** @brief The Bernoulli gear head: likelihood, entropy, KL, PPO gradient, sampling and controls. */
void GearPolicy(Config const& c)
{
    auto settings=c.training;
    settings.entropy=0.01;
    PolicyDistribution policy{};
    policy.logStandardDeviations.fill(-0.5);
    for(double const logit : {-1.3,0.0,2.1})
    {
        policy.output[GearOutput]=logit;
        double const p=1/(1+std::exp(-logit));
        Check(Near(GearEntropy(logit),-p*std::log(p)-(1-p)*std::log(1-p),1e-12),"Bernoulli gear entropy");
        for(double const other : {-0.4,0.7})
        {
            double const q=1/(1+std::exp(-other));
            Check(Near(GearKl(logit,other),p*std::log(p/q)+(1-p)*std::log((1-p)/(1-q)),1e-12),
                "Bernoulli gear KL in the old-to-new direction");
        }
        for(bool const down : {false,true})
        {
            PolicyAction action{};action.gearDown=down;action.brakeActive=false;
            double const gaussian=LogProbability(policy,action);
            action.gearActive=true;
            double const expected=down ? std::log(p) : std::log(1-p);
            Check(Near(LogProbability(policy,action)-gaussian,expected,1e-12),
                "an active gear draw adds its Bernoulli log-probability");
            for(double const advantage : {-1.1,0.9})
            {
                for(double const ratio : {0.7,1.0,1.4})
                {
                    action.logProbability=LogProbability(policy,action)-std::log(ratio);
                    auto const loss=ClippedLoss(policy,action,advantage,settings);
                    auto plus=policy,minus=policy;
                    plus.output[GearOutput]+=1e-6;minus.output[GearOutput]-=1e-6;
                    double const numerical=(ClippedLoss(plus,action,advantage,settings).loss
                        -ClippedLoss(minus,action,advantage,settings).loss)/2e-6;
                    Check(Near(numerical,loss.outputDerivative[GearOutput],1e-6),
                        "gear-logit finite-difference PPO gradient, entropy included");
                }
            }
            action.gearActive=false;
            Check(LogProbability(policy,action)==gaussian && Entropy(policy,false,false)==Entropy(policy,false)
                && ClippedLoss(policy,action,1.0,settings).outputDerivative[GearOutput]==0,
                "a latched gear contributes nothing to PPO");
        }
    }
    // Deterministic and stochastic draws.
    std::mt19937 rng(9);
    policy.output[GearOutput]=0.2;
    Check(Sample(policy,rng,true,false,true).gearDown,"a deterministic gear extends for a non-negative logit");
    policy.output[GearOutput]=-0.2;
    Check(!Sample(policy,rng,true,false,true).gearDown,"a deterministic gear stays up for a negative logit");
    std::mt19937 withoutGear(5),inactiveGear(5);
    Sample(policy,withoutGear,false,false);
    auto const inactive=Sample(policy,inactiveGear,false,false,false);
    Check(withoutGear==inactiveGear && !inactive.gearActive && !inactive.gearDown,
        "an inactive gear draws no random number");
    policy.output[GearOutput]=std::log(0.7/0.3);
    int extensions=0;
    constexpr int Draws=20000;
    for(int draw=0;draw<Draws;++draw)
    {
        extensions+=Sample(policy,rng,false,false,true).gearDown ? 1 : 0;
    }
    Check(Near(static_cast<double>(extensions)/Draws,0.7,0.02),"the stochastic gear follows sigmoid(logit)");
    // Physical mapping: only an active extension request moves the lever.
    PolicyAction action{};action.gearActive=true;action.gearDown=true;
    Check(ToControls(action,c.initial).gear==1,"an active extension request commands the gear down");
    action.gearDown=false;
    Check(ToControls(action,c.initial).gear==0,"an active refusal leaves the lever up");
    action.gearActive=false;action.gearDown=true;
    Check(ToControls(action,c.initial).gear==0,"an inactive gear never requests anything");
}
} // namespace
/**
 * @brief A zero actor output holds the 3-degree approach from short curriculum finals.
 *
 * An untrained actor starts near zero, so the neutral controls must fly the
 * glide instead of diving: otherwise every early episode crashes before the
 * runway and PPO never observes a landing.
 */
void TrimmedNeutralApproach(Config c)
{
    c.training.curriculum.rolloutStartShare=0;
    constexpr int SettlingActions=50;
    constexpr int MeasuredActions=150;
    constexpr double MaximumSinkErrorMps=1.0;
    constexpr double MaximumSpeedErrorMps=4.0;
    for(std::uint32_t seed : {11U,12U,13U})
    {
        FlightEnvironment environment(c,DefaultDataRoot(),seed);
        environment.SetDifficulty(0);
        environment.Reset();
        auto const neutral=ToControls(PolicyAction{},c.initial);
        double sinkSum=0,speedSum=0;
        for(int action=0;action<SettlingActions+MeasuredActions;++action)
        {
            auto const result=environment.Step(neutral);
            Check(!result.terminated && !result.truncated,"the trimmed neutral approach stays airborne");
            if(action<SettlingActions) continue;
            sinkSum+=environment.Aircraft().velocity.up;
            speedSum+=environment.Aircraft().airspeed;
        }
        double const glideSink=-c.mission.approachSpeed*std::tan(c.mission.glideDegrees*DegreesToRadians);
        Check(Near(sinkSum/MeasuredActions,glideSink,MaximumSinkErrorMps)
            && Near(speedSum/MeasuredActions,c.mission.approachSpeed,MaximumSpeedErrorMps),
            "the trimmed neutral controls fly the 3-degree glide at approach speed");
    }
}

int main()
{
    try
    {
        auto const c=LoadConfig(PPO_FLIGHT_CONFIG);
        PortableRngCheckpoint();
        GaussianPolicy(c); Exploration(c); ActiveControls(c); ObservationBounds(c); Training(c); AnalyticKl();
        KlProtection(c); KlRejectedActorState(c); TrimmedNeutralApproach(c); GearPolicy(c);
    }
    catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
    std::cout<<"Gaussian PPO/RLtools regressions passed\n";
}
