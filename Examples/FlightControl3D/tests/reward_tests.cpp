#include "test_support.hpp"
#include "flight_mission.hpp"
#include "flight_reward.hpp"
using namespace ppo::examples::flight;

namespace {
AircraftState At(Config const& c,double along,double cross,double height)
{
    AircraftState aircraft{};
    aircraft.position=FromRunway(c.runway,along,cross,height);
    aircraft.airspeed=c.mission.approachSpeed;
    aircraft.heading=c.runway.headingDegrees*DegreesToRadians;
    aircraft.velocity=Scale(RunwayForward(c.runway),c.mission.approachSpeed);
    aircraft.gearPosition=1;
    return aircraft;
}
double Channel(StepResult const& step,Reward reward)
{
    return step.rewards[static_cast<std::size_t>(reward)];
}
double Potential(Config const& c,MissionState const& mission,AircraftState const& aircraft,Reward channel)
{
    return ShapingPotentials(c,mission,aircraft)[static_cast<std::size_t>(channel)];
}
double PathPotential(Config const& c,MissionState const& mission,AircraftState const& aircraft)
{
    return Potential(c,mission,aircraft,Reward::Path);
}
Config T6Config()
{
    auto c=LoadConfig(std::filesystem::path(PPO_FLIGHT_CONFIG));
    c.runway.origin={};c.runway.headingDegrees=0;
    return c;
}
MissionState Flying(Phase phase=Phase::Approach)
{
    MissionState mission{};
    mission.phase=phase;
    return mission;
}
double SubstepGamma(Config const& c)
{
    return std::pow(c.training.gamma,1.0/static_cast<double>(c.mission.actionSubsteps));
}
constexpr std::array<Reward,3> PotentialChannels{Reward::Path,Reward::VerticalSpeed,Reward::Speed};

/** @brief Every shaping channel is gamma times the next potential minus the previous one. */
void PotentialShaping()
{
    auto const c=T6Config();
    auto const before=At(c,-2500,200,200);
    auto closer=At(c,-2400,200,200);closer.airspeed+=3;closer.velocity.up=-2;
    auto const mission=Flying();
    double const gamma=SubstepGamma(c);
    auto forwardMission=mission,stationaryMission=mission;
    auto const forward=AdvanceMission(c,forwardMission,before,closer);
    auto const stationary=AdvanceMission(c,stationaryMission,before,before);
    for(auto const channel : PotentialChannels)
    {
        double const phi0=Potential(c,mission,before,channel),phi1=Potential(c,mission,closer,channel);
        Check(Near(Channel(forward,channel),gamma*phi1-phi0,1e-10)
            && Near(Channel(stationary,channel),(gamma-1)*phi0,1e-10),
            "a shaping channel is gamma times the next potential minus the previous potential");
        Check(IsPotentialChannel(channel),"the channel is declared as a potential");
    }
    Check(Potential(c,mission,closer,Reward::Path)>Potential(c,mission,before,Reward::Path),
        "getting closer increases the Path potential");
    auto reversed=closer;reversed.velocity=Scale(before.velocity,-1);
    auto reversedMission=mission;
    auto const badDirection=AdvanceMission(c,reversedMission,before,reversed);
    Check(Channel(badDirection,Reward::Direction)<Channel(forward,Reward::Direction),
        "an opposite velocity has a greater direction cost");
    for(auto const channel : {Reward::Touchdown,Reward::Stop,Reward::Crash})
        Check(Channel(forward,channel)==0,"event rewards remain zero in flight");

    auto timeoutMission=mission;
    timeoutMission.steps=static_cast<std::uint64_t>(c.mission.maxSeconds/c.mission.physicsDt);
    auto const timeout=AdvanceMission(c,timeoutMission,before,closer);
    Check(timeout.truncated && !timeout.terminated,"the time limit truncates");
    for(auto const channel : PotentialChannels)
        Check(Near(Channel(timeout,channel),gamma*Potential(c,mission,closer,channel)-Potential(c,mission,before,channel),1e-10),
            "a timeout retains the bootstrapped potentials");
    auto invalid=before;invalid.finite=false;
    auto crashMission=mission;
    auto const crash=AdvanceMission(c,crashMission,before,invalid);
    Check(crash.terminated && Near(Channel(crash,Reward::Crash),Weight(c.reward,Reward::Crash)),"a crash is terminal");
    for(auto const channel : PotentialChannels)
        Check(Near(Channel(crash,channel),-Potential(c,mission,before,channel)),"a crash clears every potential");
}

/** @brief A closed trajectory cannot farm any shaping potential. */
void ShapingTelescopes()
{
    auto const c=T6Config();
    auto mission=Flying();
    auto const start=At(c,-2500,100,150);
    auto closer=At(c,-2400,100,150);closer.airspeed=c.mission.approachSpeed-4;closer.velocity.up=-5;
    double const gamma=SubstepGamma(c);
    auto const forward=AdvanceMission(c,mission,start,closer);
    auto const backward=AdvanceMission(c,mission,closer,start);
    Check(!forward.terminated && !backward.terminated,"the loop stays in flight");
    for(auto const channel : PotentialChannels)
        Check(Near(Channel(forward,channel)+gamma*Channel(backward,channel),
            (gamma*gamma-1)*Potential(c,Flying(),start,channel),1e-10),
            "out-and-back shaping telescopes without a positive recurrent reward");
}

/** @brief Passing the threshold and the aim point, then rolling, never lowers the Path potential. */
void PathAfterThreshold()
{
    auto const c=T6Config();
    auto const approach=Flying();
    double const aimAlong=AimPointAlongMeters(c);
    double const justBefore=PathPotential(c,approach,At(c,aimAlong-1e-6,10,3));
    double const justAfter=PathPotential(c,approach,At(c,aimAlong+1e-6,10,3));
    Check(Near(justBefore,justAfter,1e-6),"the potential is continuous across the aim point");
    double previous=0;
    for(double along : {-3000.0,-1000.0,0.0,aimAlong})
    {
        double const potential=PathPotential(c,approach,At(c,along,0,ApproachPathHeight(c,along)));
        Check(potential>previous,"descending the glide plane toward the aim point increases the potential");
        previous=potential;
    }
    Check(Near(previous,Weight(c.reward,Reward::Path)),"the potential reaches its maximum at the aim point");
    Check(Near(PathPotential(c,approach,At(c,aimAlong+300,0,5)),PathPotential(c,approach,At(c,aimAlong+1500,0,5))),
        "flying along the runway past the aim point is neither rewarded nor penalized");
    auto rollout=Flying(Phase::Rollout);rollout.touchedDown=true;
    for(double along : {100.0,aimAlong,2000.0,3500.0})
        Check(Near(PathPotential(c,rollout,At(c,along,5,1.5)),Weight(c.reward,Reward::Path)),
            "rolling down the runway keeps the full potential: no contradiction with stopping");
    auto stopped=rollout;stopped.outcome=Outcome::Stopped;
    Check(PathPotential(c,stopped,At(c,2000,0,1.5))==0,"an absorbing stop carries no potential");
}

/** @brief The vertical-speed potential favours the commanded sink rate in flight only. */
void VerticalSpeedShaping()
{
    auto const c=T6Config();
    auto const mission=Flying();
    auto aircraft=At(c,-3000,0,ApproachPathHeight(c,-3000));
    double const weight=Weight(c.reward,Reward::VerticalSpeed);
    double const target=TargetVerticalSpeed(Guide(c,mission,aircraft));
    Check(target<-2 && target>-3.5,"the commanded sink on the 3-degree plane is about 2.9 m/s");
    aircraft.velocity.up=target;
    Check(Near(Potential(c,mission,aircraft,Reward::VerticalSpeed),0,1e-9),"the commanded sink has zero potential");
    aircraft.velocity.up=target-1;
    Check(Near(Potential(c,mission,aircraft,Reward::VerticalSpeed),-weight/VerticalSpeedErrorScaleMps,1e-9),
        "a 1 m/s sink error costs half of the weight");
    aircraft.velocity.up=target+1;
    Check(Near(Potential(c,mission,aircraft,Reward::VerticalSpeed),-weight/VerticalSpeedErrorScaleMps,1e-9),
        "climbing above the commanded sink is penalized symmetrically");
    aircraft.velocity.up=-30;
    Check(Near(Potential(c,mission,aircraft,Reward::VerticalSpeed),-weight),"the potential saturates at the weight");
    auto flare=At(c,AimPointAlongMeters(c)+200,0,1);
    double const flareTarget=TargetVerticalSpeed(Guide(c,mission,flare));
    Check(flareTarget<0 && flareTarget>-1,"close to the deck the commanded sink becomes gentle");
    auto rolling=At(c,800,0,1.5);rolling.weightOnWheels=true;rolling.velocity.up=-3;
    Check(Potential(c,Flying(Phase::Rollout),rolling,Reward::VerticalSpeed)==0,"there is no vertical reference on the ground");
}

/** @brief The speed potential holds the target airspeed in flight and rewards braking on the ground. */
void SpeedShaping()
{
    auto const c=T6Config();
    double const weight=Weight(c.reward,Reward::Speed);
    auto const mission=Flying();
    auto aircraft=At(c,-6000,0,ApproachPathHeight(c,-6000));
    double const target=Guide(c,mission,aircraft).targetSpeed;
    aircraft.airspeed=target;
    Check(Near(Potential(c,mission,aircraft,Reward::Speed),0,1e-9),"flying at the target airspeed has zero potential");
    aircraft.airspeed=target+5;
    Check(Near(Potential(c,mission,aircraft,Reward::Speed),-weight*5/AirspeedErrorScaleMps,1e-9),
        "flying too fast is penalized in proportion to the error");
    aircraft.airspeed=target-5;
    Check(Near(Potential(c,mission,aircraft,Reward::Speed),-weight*5/AirspeedErrorScaleMps,1e-9),
        "flying too slow is penalized as well");
    auto const rollout=Flying(Phase::Rollout);
    double previous=-1;
    for(double speed : {c.mission.flareSpeed,40.0,20.0,5.0,0.0})
    {
        auto rolling=At(c,1000,0,1.5);rolling.weightOnWheels=true;rolling.velocity=Scale(RunwayForward(c.runway),speed);
        double const potential=Potential(c,rollout,rolling,Reward::Speed);
        Check(potential>previous,"slowing down on the ground increases the potential");
        previous=potential;
    }
    Check(Near(previous,weight),"a stopped aircraft reaches the full potential");
    auto touchdown=At(c,400,0,0.2);touchdown.velocity=Scale(RunwayForward(c.runway),c.mission.flareSpeed);
    touchdown.airspeed=c.mission.flareSpeed;touchdown.flapPosition=c.initial.approachFlap;
    auto onWheels=touchdown;onWheels.weightOnWheels=true;
    Check(Near(Potential(c,mission,touchdown,Reward::Speed),0,1e-9)
        && Near(Potential(c,rollout,onWheels,Reward::Speed),0,1e-9),
        "an on-speed touchdown keeps the speed potential continuous");
}

/** @brief The touchdown quality rewards a gentle, on-speed contact in the touchdown zone. */
void TouchdownQualityFactors()
{
    auto const c=T6Config();
    auto Quality=[&c](double along,double cross,double sink,double speed)
    {
        MissionState mission{};
        mission.touchdownFromThreshold=along;mission.touchdownCrossMeters=cross;
        mission.touchdownSinkMps=sink;mission.touchdownSpeedMps=speed;
        return TouchdownQuality(c,mission);
    };
    double const ideal=Quality(400,0,0,c.mission.flareSpeed);
    Check(Near(ideal,1),"an ideal touchdown has full quality");
    Check(Quality(400,0,1,52)<ideal && Quality(400,0,3,52)<Quality(400,0,1,52)
        && Quality(400,0,1,52)>0.9 && Near(Quality(400,0,c.mission.maxTouchdownSink,52),0),
        "the quality decreases quadratically with the sink rate");
    Check(Quality(400,0,0,58)<ideal && Quality(400,0,0,c.mission.maxTouchdownSpeed)>=0.5-1e-12,
        "a fast touchdown keeps at least half of the quality");
    Check(Near(Quality(TouchdownZoneLengthMeters,0,0,52),1) && Quality(1500,0,0,52)<1
        && Near(Quality(c.runway.lengthMeters/2,0,0,52),0),"a long landing loses its quality by mid-runway");
    Check(Quality(-25,0,0,52)<ideal && Near(Quality(400,c.runway.widthMeters/2+c.runway.landingMarginMeters,0,52),0),
        "a touchdown in the safety margin keeps a reduced quality");

    auto landing=At(c,400,0,2.1);landing.velocity={52,0,-1};landing.airspeed=52;
    auto wheels=landing;wheels.weightOnWheels=true;wheels.velocity.up=0;
    auto landingMission=Flying();
    auto const touchdown=AdvanceMission(c,landingMission,landing,wheels);
    Check(landingMission.touchedDown && Near(landingMission.touchdownQuality,Quality(400,0,1,52))
        && Near(Channel(touchdown,Reward::Touchdown),Weight(c.reward,Reward::Touchdown)*landingMission.touchdownQuality),
        "a safe wheel contact is rewarded once with its quality");
    wheels.velocity={};wheels.airspeed=0;
    StepResult stop{};
    for(int step=0;step<3000 && landingMission.outcome==Outcome::Flying;++step)
        stop=AdvanceMission(c,landingMission,wheels,wheels);
    Check(landingMission.outcome==Outcome::Stopped && Near(Channel(stop,Reward::Stop),Weight(c.reward,Reward::Stop)),
        "the stop reward depends only on the landing area, not on the touchdown sink");
}
} // namespace

int main()
{
    try
    {
        PotentialShaping();
        ShapingTelescopes();
        PathAfterThreshold();
        VerticalSpeedShaping();
        SpeedShaping();
        TouchdownQualityFactors();
    }
    catch(std::exception const& error)
    {
        std::cerr<<error.what()<<'\n';
        return 1;
    }
    std::cout<<"Flight reward regressions passed\n";
}
