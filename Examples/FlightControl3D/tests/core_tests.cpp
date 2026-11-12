#include "test_support.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <limits>
using namespace ppo::examples::flight;

namespace {
AircraftState Approach(Config const& c, double along=-1000, double cross=0)
{
    AircraftState a{};
    a.position=FromRunway(c.runway,along,cross,c.runway.thresholdHeightMeters-along*std::tan(c.mission.glideDegrees*DegreesToRadians));
    a.airspeed=c.mission.approachSpeed;
    a.heading=c.runway.headingDegrees*DegreesToRadians;
    a.velocity=FromRunway(c.runway,a.airspeed,0,-a.airspeed*std::tan(c.mission.glideDegrees*DegreesToRadians));
    a.gearPosition=1;
    return a;
}
void Geometry(Config const& c)
{
    auto const plan=MakeFlightPlan(c);
    Check(plan.count==2*(c.mission.straightSamples+c.mission.turnSamples),"plan point count");
    for (double h : {0.,90.,181.,359.})
    {
        auto runway=c.runway; runway.headingDegrees=h;
        auto const xy=ToRunway(runway,FromRunway(runway,-1234,57,99));
        Check(Near(xy[0],-1234) && Near(xy[1],57),"runway rotation/inverse");
    }
    MissionState s{};
    auto a=Approach(c);
    auto const first=AdvanceMission(c,plan,s,a,a);
    Check(s.phase==Phase::Approach && !first.terminated,"approach switch");
    auto const g=Guide(c,plan,s,a);
    Check(g.ilsValid && Near(g.localizer,0) && Near(g.glideError,0),"on LOC/glide");
    Check(s.capturedLocalizer && s.capturedGlide,"capture events");
    Check(first.rewards[static_cast<std::size_t>(Reward::EnterLocalizer)]>0,"first capture bonus");
    auto const again=AdvanceMission(c,plan,s,a,a);
    Check(again.rewards[5]==0 && again.rewards[6]==0,"captures must be one-shot");
    auto far=Approach(c,-c.mission.ilsRangeMeters-1);
    AdvanceMission(c,plan,s,a,far);
    Check(s.phase==Phase::Approach,"approach cannot revert to patrol");
    MissionState back{}; back.phase=Phase::Approach;
    auto behind=Approach(c,1000);
    behind.position.up=100;
    auto const outside=AdvanceMission(c,plan,back,behind,behind);
    Check(!Guide(c,plan,back,behind).ilsValid && !back.capturedGlide && outside.rewards[1]==0,"ILS backside suppression");
    auto invalid=Approach(c,-1000,2000);
    Check(!Guide(c,plan,back,invalid).ilsValid,"ILS sector");
    auto obs=Observe(c,plan,s,a);
    Check(obs.size()==ObservationNames.size(),"observation schema");
    for (float value : obs) Check(std::isfinite(value) && std::abs(value)<=1,"normalized finite observation");
    a.velocity.up=std::numeric_limits<double>::quiet_NaN();
    for (float value : Observe(c,plan,s,a)) Check(value==0,"invalid terminal observation");
    auto failure=AdvanceMission(c,plan,s,far,a);
    Check(failure.terminated && !failure.truncated && failure.reward==c.reward.weights[9],"NaN FDM crash");
}
void Landing(Config c)
{
    auto const plan=MakeFlightPlan(c);
    auto before=Approach(c,300); before.position.up=1.8; before.velocity.up=-1;
    auto after=before; after.weightOnWheels=true; after.velocity.up=0;
    MissionState s{}; s.phase=Phase::Approach;
    auto landed=AdvanceMission(c,plan,s,before,after);
    Check(s.touchedDown && s.phase==Phase::Rollout && !landed.terminated,"safe touchdown continues to stop");
    Check(Near(s.touchdownFromThreshold,300) && landed.rewards[7]==c.reward.weights[7],"touchdown distance and bonus");
    auto rolling=after;
    rolling.position=FromRunway(c.runway,600,0,1.8); rolling.velocity={};
    auto roll=AdvanceMission(c,plan,s,after,rolling);
    Check(!roll.terminated && roll.rewards[7]==0 && s.groundRoll>=299,"no repeated touchdown; roll distance");
    for (std::size_t i=0; i<static_cast<std::size_t>(c.mission.stoppedHoldSeconds/c.mission.physicsDt)+2; ++i)
    {
        if (s.outcome!=Outcome::Flying) break;
        roll=AdvanceMission(c,plan,s,rolling,rolling);
    }
    Check(roll.terminated && s.outcome==Outcome::Stopped && Near(s.stopFromThreshold,600),"confirmed stop location");
    Check(roll.rewards[8]==c.reward.weights[8],"stop reward");
    for (int test=0; test<7; ++test)
    {
        auto b=before, a=after;
        if (test==0) a.gearPosition=0;
        if (test==1) a.gearPosition=c.mission.gearDownMinimum-0.01;
        if (test==2) b.velocity.up=-c.mission.maxTouchdownSink-1; // Impact solver may already have removed sink after contact.
        if (test==3) a.position=FromRunway(c.runway,300,c.runway.widthMeters,1.8);
        if (test==4) a.structuralContact=true;
        if (test==5) a.roll=(c.mission.maxTouchdownRollDegrees+1)*DegreesToRadians;
        if (test==6) a.heading+=Pi;
        MissionState crash{}; crash.phase=Phase::Approach;
        auto const result=AdvanceMission(c,plan,crash,b,a);
        Check(result.terminated && crash.outcome==Outcome::Crash && !crash.touchedDown && result.rewards[7]==0,"unsafe landing rejected");
    }
    MissionState overrun{}; overrun.phase=Phase::Rollout; overrun.touchedDown=true;
    auto off=rolling; off.position=FromRunway(c.runway,c.runway.lengthMeters+1,0,2);
    Check(AdvanceMission(c,plan,overrun,rolling,off).terminated && overrun.outcome==Outcome::Crash,"runway overrun");
    MissionState timeout{}; timeout.phase=Phase::Approach;
    timeout.steps=static_cast<std::uint64_t>(std::ceil(c.mission.maxSeconds/c.mission.physicsDt));
    auto const timed=AdvanceMission(c,plan,timeout,before,before);
    Check(timed.truncated && !timed.terminated,"time limit bootstraps");
    auto const noRepeat=AdvanceMission(c,plan,timeout,before,before);
    Check(noRepeat.reward==0 && noRepeat.truncated,"terminal reward not repeated");
}
void Configuration()
{
    std::ifstream source(PPO_FLIGHT_CONFIG); nlohmann::json valid; source>>valid;
    TemporaryDirectory temporary("flight-config-"); auto const path=temporary.path/"config.json";
    for (int test=0; test<8; ++test)
    {
        auto json=valid;
        if (test==0) json["normalization"]["speed"]=0;
        if (test==1) json["mission"]["turnSamples"]=128;
        if (test==2) json["training"]["minibatch"]=3;
        if (test==3) json["mission"]["actionSubsteps"]=-1;
        if (test==4) json["initial"]["gearDown"]=1;
        if (test==5) json["mission"]["hiddenConstant"]=3;
        if (test==6) json["training"]["checkpointDirectory"]="../BluePatrol/prod";
        if (test==7) json["mission"].erase("physicsDt");
        { std::ofstream stream(path); stream<<json; }
        Rejects([&] { LoadConfig(path); },"invalid JSON accepted");
    }
}
void RandomizationAndMotion(Config const& c)
{
    std::mt19937 rng(91),same(91);
    bool differentPositions=false,differentAttitudes=false;
    AircraftState first{};
    for (std::size_t sample=0;sample<1000;++sample)
    {
        auto const motion=SampleRunway(c,rng),repeated=SampleRunway(c,same);
        auto const runway=RunwayAt(motion,0),copy=RunwayAt(repeated,0);
        auto const aircraft=SampleAircraft(c,runway,rng),replica=SampleAircraft(c,copy,same);
        auto const local=DeckCoordinates(runway,aircraft.position);
        double const distance=std::hypot(local.north,local.east);
        Check(distance>=c.initial.distanceMeters.minimum-1e-7 && distance<=c.initial.distanceMeters.maximum+1e-7,"random spawn distance bounds");
        Check(local.up>=c.initial.heightMeters.minimum-1e-7 && local.up<=c.initial.heightMeters.maximum+1e-7,"random spawn height bounds");
        Check(aircraft.airspeed>=c.initial.speedMps.minimum && aircraft.airspeed<=c.initial.speedMps.maximum,"random spawn speed bounds");
        Check(aircraft.roll>=c.initial.rollDegrees.minimum*DegreesToRadians && aircraft.roll<=c.initial.rollDegrees.maximum*DegreesToRadians,"random roll bounds");
        Check(aircraft.pitch>=c.initial.pitchDegrees.minimum*DegreesToRadians && aircraft.pitch<=c.initial.pitchDegrees.maximum*DegreesToRadians,"random pitch bounds");
        Check(aircraft.position.north==replica.position.north && aircraft.heading==replica.heading && aircraft.airspeed==replica.airspeed,"seeded reproducible randomization");
        if (sample==0) first=aircraft;
        else { differentPositions=differentPositions || aircraft.position.north!=first.position.north;
            differentAttitudes=differentAttitudes || aircraft.heading!=first.heading || aircraft.roll!=first.roll; }
        double const t=1.3,epsilon=1e-4;
        auto const deck=RunwayAt(motion,t);
        auto const point=FromRunway(deck,600,8,0);
        auto const plus=FromRunway(RunwayAt(motion,t+epsilon),600,8,0);
        auto const minus=FromRunway(RunwayAt(motion,t-epsilon),600,8,0);
        auto const numerical=Scale(Subtract(plus,minus),1/(2*epsilon));
        auto const velocity=DeckVelocity(deck,point);
        Check(Near(velocity.north,numerical.north,1e-5) && Near(velocity.east,numerical.east,1e-5)
            && Near(velocity.up,numerical.up,1e-5),"heave/pitch/roll contact velocity matches surface derivative");
        auto const transformed=DeckCoordinates(deck,FromRunway(deck,-2100,27,500));
        Check(Near(transformed.north,-2100) && Near(transformed.east,27) && Near(transformed.up,500),"moving tilted runway inverse");
    }
    Check(differentPositions && differentAttitudes,"episodes must not share a fixed initial aircraft state");
    auto moving=c;
    moving.runway.velocity={8,0,1};
    MissionState s{};s.phase=Phase::Approach;
    auto a=Approach(moving,300);a.position.up=2;a.velocity={8,0,1};a.weightOnWheels=true;
    auto const plan=MakeFlightPlan(moving);
    AdvanceMission(moving,plan,s,a,a);
    Check(s.touchedDown,"touchdown limits use deck-relative sink");
    s.stoppedSeconds=moving.mission.stoppedHoldSeconds;
    auto const result=AdvanceMission(moving,plan,s,a,a);
    Check(result.terminated && s.outcome==Outcome::Stopped,"stop is relative to moving runway");
}
}
int main()
{
    try { auto const c=LoadConfig(PPO_FLIGHT_CONFIG); Geometry(c); Landing(c); Configuration(); RandomizationAndMotion(c); }
    catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
    std::cout<<"Flight mission/configuration regressions passed\n";
}
