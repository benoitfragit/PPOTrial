/**
 * @file core_tests.cpp
 * @brief Deterministic regressions of Core/: configuration, geometry, guidance, landing,
 *        observation, spawns, curriculum starts, evaluation plans and display references.
 *
 * Protects: the strict JSON schema and its validation rules, the checkpoint signature
 * compatibility rules, the frame conversions, the aim point and the flare reference, the
 * touchdown limits and crash diagnostics, the observation bounds and resolutions, the start
 * samplers and the stratified evaluation plan. Uses the production scenario
 * (PPO_FLIGHT_CONFIG); no JSBSim and no network.
 *
 * Read next: reward_tests.cpp, simulator_tests.cpp.
 */
#include "test_support.hpp"
#include "flight_landing.hpp"
#include "flight_mission.hpp"
#include "flight_observation.hpp"
#include "flight_reward.hpp"
#include "flight_t6_performance.hpp"
#include "../Renderer/flight_display.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <limits>
#include <utility>
using namespace ppo::examples::flight;

namespace {
Config T6Config()
{
    return LoadConfig(std::filesystem::path(PPO_FLIGHT_CONFIG));
}

/** @brief Aircraft on the 3-degree plane through the threshold crossing height, flying down the runway axis. */
AircraftState Approach(Config const& c, double along=-1000, double cross=0)
{
    double const gradient=std::tan(c.mission.glideDegrees*DegreesToRadians);
    AircraftState a{};
    a.position=FromRunway(c.runway,along,cross,c.runway.thresholdHeightMeters-along*gradient);
    a.airspeed=c.mission.approachSpeed;
    a.heading=c.runway.headingDegrees*DegreesToRadians;
    a.velocity=Add(Scale(RunwayForward(c.runway),a.airspeed),Scale(RunwayNormal(c.runway),-a.airspeed*gradient));
    a.gearPosition=1;
    return a;
}

MissionState Flying(Phase phase=Phase::Approach)
{
    MissionState mission{};
    mission.phase=phase;
    return mission;
}

double Channel(StepResult const& step,Reward reward) { return step.rewards[static_cast<std::size_t>(reward)]; }

/** @brief The scenario exposes only the retained direct landing task. */
void Configuration()
{
    auto const c=T6Config();
    std::ifstream source(PPO_FLIGHT_CONFIG);nlohmann::json json;source>>json;
    Check(json.at("reward").at("weights").size()==RewardCount,"the scenario exposes exactly the reward channels");
    for(auto const* key : {"patrolEnabled","racetrackRadius","gearLockedDown","gearDamageSpeedMps","approachHeadingToleranceDegrees"})
        Check(!json.at("mission").contains(key),"removed mission settings are absent");
    Check(!json.at("training").contains("curriculumAircraft") && !json.at("training").at("curriculum").contains("stage")
        && !json.at("training").at("curriculum").contains("pilot"),"removed pilot-lesson settings are absent");
    for(auto const* key : {"speedMps","heaveAmplitudeMeters","pitchAmplitudeDegrees","rollAmplitudeDegrees"})
        Check(!json.at("runwayRandomization").contains(key),"the runway no longer moves");
    Check(Near(c.mission.flareSpeed,52.0),"T6 flare keeps a model-derived airborne speed reserve");
    Check(T6StallCalibratedSpeed(0.33)>38 && T6StallCalibratedSpeed(0.33)<39,
        "the pinned mass, wing area and flap lift imply a 38-39 m/s stall CAS");

    auto invalid=json;invalid["mission"]["patrolEnabled"]=false;
    TemporaryDirectory directory("flight-config-");
    auto const path=directory.path/"scenario.json";
    {std::ofstream output(path);output<<invalid;}
    Rejects([&] { LoadConfig(path); },"unknown legacy settings are rejected");
    invalid=json;invalid["formatVersion"]=1;
    {std::ofstream output(path);output<<invalid;}
    Rejects([&] { LoadConfig(path); },"the previous configuration format is rejected");
}

/** @brief The scenario weights keep the landing objective dominant and visible from a far start. */
void RewardBalance()
{
    auto const c=T6Config();
    auto const weight=[&c](Reward channel) { return Weight(c.reward,channel); };
    Check(weight(Reward::Stop)>weight(Reward::Touchdown) && weight(Reward::Touchdown)>0 && weight(Reward::Crash)<0,
        "stopping is the main objective and a crash is penalized");
    double const shaping=weight(Reward::Path)+weight(Reward::VerticalSpeed)+weight(Reward::Speed);
    Check(shaping<weight(Reward::Stop)+weight(Reward::Touchdown),
        "the full shaping amplitude stays below the landing rewards");
    // A far start lasts about 250 s of flight; the discounted Stop must stay visible.
    double const farStartSteps=c.initial.distanceMeters.maximum/c.mission.approachSpeed/c.StepSeconds();
    Check(std::pow(c.training.gamma,farStartSteps)>0.05,"the discount horizon reaches the landing from a far start");
    Check(c.mission.maxSeconds*c.mission.approachSpeed>1.5*c.initial.distanceMeters.maximum,
        "the time limit leaves room for the longest approach and the rollout");
}

/** @brief Warm starts accept reward and spawn changes, never physics or guidance changes. */
void SignatureCompatibility()
{
    auto const c=T6Config();
    auto signature=nlohmann::json::parse(c.checkpointSignature);
    Check(!signature.contains("training") && !signature.contains("evaluation") && signature.contains("contractVersion"),
        "the checkpoint contract excludes optimizer and evaluation settings");
    auto changed=signature;changed["reward"]["weights"]["Path"]=1.0;changed["initial"]["distanceMeters"]={1000,2000};
    Check(changed.dump()!=c.checkpointSignature && CompatibleTrainingSignature(changed.dump(),c.checkpointSignature),
        "reward and spawn changes keep a compatible warm start");
    for(auto const* field : {"throttle","elevatorTrim"})
    {
        changed=signature;changed["initial"][field]=changed["initial"][field].get<double>()+0.05;
        Check(!CompatibleTrainingSignature(changed.dump(),c.checkpointSignature),
            "the trimmed controls behind a zero actor output are part of the contract");
    }
    for(auto const* field : {"glideDegrees","flareSpeed","maxTouchdownSink"})
    {
        changed=signature;changed["mission"][field]=changed["mission"][field].get<double>()+1.0;
        Check(!CompatibleTrainingSignature(changed.dump(),c.checkpointSignature),"mission changes are incompatible");
    }
    changed=signature;changed["contractVersion"]=0;
    Check(!CompatibleTrainingSignature(changed.dump(),c.checkpointSignature),"another contract version is incompatible");
    Check(!CompatibleTrainingSignature("broken",c.checkpointSignature),"malformed signatures are incompatible");
}

/** @brief Runway frames are exact inverses and the body axes are orthonormal. */
void Geometry()
{
    auto c=T6Config();
    c.runway.origin={150,-200,15};
    for(double heading : {0.,90.,181.,359.})
    {
        c.runway.headingDegrees=heading;
        auto const local=DeckCoordinates(c.runway,FromRunway(c.runway,-1234,57,99));
        Check(Near(local.north,-1234) && Near(local.east,57) && Near(local.up,99),"runway rotation/inverse");
    }
    for(double roll : {-0.5,0.0,0.7}) for(double pitch : {-0.2,0.1}) for(double heading : {0.0,2.0})
    {
        auto const axes=AircraftAxes(roll,pitch,heading);
        Check(Near(Length(axes.forward),1) && Near(Length(axes.right),1) && Near(Length(axes.up),1)
            && Near(Dot(axes.forward,axes.right),0) && Near(Dot(axes.forward,axes.up),0)
            && Near(Dot(axes.right,axes.up),0),"body axes are orthonormal");
    }
    auto const level=AircraftAxes(0,0,0);
    Check(Near(level.forward.north,1) && Near(level.right.east,1) && Near(level.up.up,1),
        "a level aircraft heading north has north, east and up body axes");
}

/** @brief Target speed decelerates before the flare and keeps the stall reserve. */
void TargetSpeed()
{
    auto const c=T6Config();
    auto approach=Flying();
    for(auto const sample : {std::pair<double,double>{-2700,56},{-2350,54},{-2000,52},{0,52}})
    {
        auto aircraft=Approach(c,sample.first);aircraft.flapPosition=c.initial.approachFlap;
        Check(Near(Guide(c,approach,aircraft).targetSpeed,sample.second),"T6 approach speed follows the distance ramp");
    }
    auto aircraft=Approach(c,c.runway.lengthMeters/2);aircraft.flapPosition=c.initial.approachFlap;
    auto rollout=Flying(Phase::Rollout);rollout.touchedDown=true;
    Check(Guide(c,rollout,aircraft).targetSpeed>0,"an airborne bounce cannot request zero speed");
    aircraft.weightOnWheels=true;
    Check(Near(Guide(c,rollout,aircraft).targetSpeed,0),"the rollout target is zero only on the wheels");
    aircraft.weightOnWheels=false;
    for(double flap : {0.0,0.33,1.0}) for(double along : {-1000.0,-300.0,0.0,4000.0})
    {
        aircraft.position=FromRunway(c.runway,along,0,5);aircraft.flapPosition=flap;
        Check(Guide(c,approach,aircraft).targetSpeed>=1.3*T6StallCalibratedSpeed(flap),
            "every airborne target reserves 30 percent above the current-flap stall CAS");
    }
}

/** @brief The glide plane crosses the threshold at its crossing height and meets the runway at the aim point. */
void AimPointOnRunway()
{
    auto c=T6Config();
    c.runway.origin={150,-200,25};c.runway.headingDegrees=123;
    double const aimAlong=AimPointAlongMeters(c);
    auto const aim=DeckCoordinates(c.runway,AimPoint(c));
    Check(aimAlong>0 && aimAlong<c.runway.lengthMeters/2 && Near(aim.north,aimAlong) && Near(aim.east,0) && Near(aim.up,0)
        && OnLandingSurface(c.runway,aim.north,aim.east) && LandingRewardFactor(c.runway,aim.north,aim.east)==1,
        "the aim point lies on the runway centreline, inside the real runway");
    Check(Near(ApproachPathHeight(c,0),c.runway.thresholdHeightMeters) && Near(ApproachPathHeight(c,aimAlong),0)
        && Near(ApproachPathHeight(c,aimAlong+500),0),"the glide plane crosses the threshold at its crossing height");
    Check(Near(ApproachPathHeight(c,-5000),c.runway.thresholdHeightMeters+5000*std::tan(c.mission.glideDegrees*DegreesToRadians)),
        "spawns and guidance share the same glide plane");
}

/** @brief Far out the direction points at the aim point; near it the track converges onto the centreline. */
void ApproachDirection()
{
    auto c=T6Config();
    c.runway.origin={150,-200,25};c.runway.headingDegrees=123;
    double const aimAlong=AimPointAlongMeters(c);
    for(double cross : {-300.0,0.0,250.0}) for(double along : {-9000.0,-3000.0,-1000.0})
    {
        auto const aircraft=Approach(c,along,cross);
        auto const guidance=Guide(c,Flying(),aircraft);
        auto const toAim=UnitDirection(Subtract(AimPoint(c),aircraft.position));
        Check(guidance.targetValid && Near(Dot(guidance.targetDirection,toAim),1,1e-9)
            && Near(Length(Subtract(guidance.target,AimPoint(c))),0,1e-6),
            "far from the runway the commanded direction points straight at the aim point");
        auto const observation=Observe(c,Flying(),aircraft);
        auto const body=InBodyAxes(AircraftAxes(aircraft.roll,aircraft.pitch,aircraft.heading),guidance.targetDirection);
        Check(Near(observation[Slot(ObservationIndex::TargetDirectionForward)],body.north,1e-6)
            && Near(observation[Slot(ObservationIndex::TargetDirectionRight)],body.east,1e-6)
            && Near(observation[Slot(ObservationIndex::TargetDirectionUp)],body.up,1e-6),
            "the policy observes the same commanded direction in body axes");
        auto truncated=Flying();truncated.outcome=Outcome::Timeout;
        Check(Observe(c,truncated,aircraft)==observation,"a truncated episode preserves its target for bootstrapping");
    }
    // Around the switch from the aim point to the sliding centreline target the direction stays continuous.
    double const switchAlong=aimAlong-CentrelineLookaheadMeters;
    auto before=Approach(c,switchAlong-0.01,20),after=Approach(c,switchAlong+0.01,20);
    Check(Near(Dot(Guide(c,Flying(),before).targetDirection,Guide(c,Flying(),after).targetDirection),1,1e-6),
        "the direction is continuous where the target starts sliding along the centreline");
    for(double along : {aimAlong-100,aimAlong,aimAlong+400,c.runway.lengthMeters*0.75})
    {
        auto aircraft=Approach(c,along,20);aircraft.position=FromRunway(c.runway,along,20,30);
        auto const direction=RunwayAxes(c.runway,Guide(c,Flying(),aircraft).targetDirection);
        Check(direction.north>0 && direction.east<0 && direction.up<0,
            "over the runway the direction keeps flying forward, converges to the centreline and descends");
    }
    auto rolling=Approach(c,800,10);rolling.position=FromRunway(c.runway,800,10,1.5);rolling.weightOnWheels=true;
    auto const rollout=Guide(c,Flying(Phase::Rollout),rolling);
    auto const rollDirection=RunwayAxes(c.runway,rollout.targetDirection);
    Check(rollout.targetValid && rollout.mode==GuidanceMode::Rollout && rollDirection.north>0 && rollDirection.east<0
        && Near(rollDirection.up,0),"while rolling the direction is level, forward and back to the centreline");
    for(auto const outcome : {Outcome::Crash,Outcome::Stopped})
    {
        auto terminal=Flying();terminal.outcome=outcome;
        Check(!Guide(c,terminal,Approach(c)).targetValid,"a terminal mission has no active target");
    }
}

/** @brief Following the commanded direction from short final flares onto the runway past the aim point. */
void FlareTrajectory()
{
    auto c=T6Config();
    c.runway.origin={0,0,15};c.runway.headingDegrees=0;
    auto aircraft=Approach(c,-2000,15);
    auto const mission=Flying();
    bool reachedWheels=false;
    for(int step=0;step<20000;++step)
    {
        auto const guidance=Guide(c,mission,aircraft);
        aircraft.velocity=Scale(guidance.targetDirection,c.mission.flareSpeed);
        auto const local=DeckCoordinates(c.runway,aircraft.position);
        double const sink=-aircraft.velocity.up;
        Check(sink>0 && aircraft.velocity.north>0,"the reference neither climbs, hovers nor turns back");
        if(local.up<=0.5)
        {
            Check(sink<c.mission.maxTouchdownSink/4 && local.north>=0 && local.north<=c.runway.lengthMeters/2
                && std::abs(local.east)<1,"the flare reaches the wheels gently, on the runway centreline");
            reachedWheels=true;break;
        }
        aircraft.position=Add(aircraft.position,Scale(aircraft.velocity,c.mission.physicsDt));
    }
    Check(reachedWheels,"the reference trajectory reaches wheel contact in finite time");
}
/** @brief The landing-area factor decreases from the real runway edge to zero at the margin. */
void LandingQuality()
{
    auto const c=T6Config();
    Check(c.runway.landingMarginMeters==50,"the T6 physical safety margin is 50 metres");
    for(int edge=0;edge<4;++edge)
    {
        double previous=1;
        for(double distance : {0.0,5.0,25.0,50.0})
        {
            double const along=edge==2 ? -distance : edge==3 ? c.runway.lengthMeters+distance : c.runway.lengthMeters/2;
            double const cross=edge==0 ? -c.runway.widthMeters/2-distance : edge==1 ? c.runway.widthMeters/2+distance : 0;
            double const quality=LandingRewardFactor(c.runway,along,cross);
            Check(Near(quality,1-distance/c.runway.landingMarginMeters) && quality<=previous,
                "quality decreases monotonically from the real edge to zero at the margin");
            previous=quality;
            auto before=Approach(c);before.position=FromRunway(c.runway,along,cross,2.1);
            before.velocity=Scale(RunwayForward(c.runway),50);before.velocity.up=-1;before.airspeed=50;
            auto after=before;after.weightOnWheels=true;after.velocity={};after.airspeed=0;
            auto mission=Flying();
            auto result=AdvanceMission(c,mission,before,after);
            Check(mission.touchedDown && Near(mission.landingAreaFactor,quality)
                && Near(mission.touchdownQuality,TouchdownQuality(c,mission))
                && Near(Channel(result,Reward::Touchdown),Weight(c.reward,Reward::Touchdown)*mission.touchdownQuality),
                "Touchdown pays its quality, which includes the real-runway distance factor");
            for(int step=0;step<300 && mission.outcome==Outcome::Flying;++step) result=AdvanceMission(c,mission,after,after);
            Check(mission.outcome==Outcome::Stopped && Near(Channel(result,Reward::Stop),Weight(c.reward,Reward::Stop)*quality),
                "Stop uses the landing-area factor");
        }
        double const along=edge==2 ? -60 : edge==3 ? c.runway.lengthMeters+60 : c.runway.lengthMeters/2;
        double const cross=edge==0 ? -c.runway.widthMeters/2-60 : edge==1 ? c.runway.widthMeters/2+60 : 0;
        Check(!OnLandingSurface(c.runway,along,cross) && Near(LandingRewardFactor(c.runway,along,cross),0.0),
            "outside the safety margin the ground is not landable");
    }
}

/** @brief Crash diagnostics preserve simultaneous failures and pre-impact measurements. */
void CrashDiagnostics()
{
    auto c=T6Config();
    c.runway.landingMarginMeters=0;
    auto before=Approach(c,300);before.position=FromRunway(c.runway,300,0,1.8);
    before.velocity=Add(Scale(RunwayForward(c.runway),c.mission.maxTouchdownSpeed+5),
        Scale(RunwayNormal(c.runway),-c.mission.maxTouchdownSink-2));
    auto after=before;after.weightOnWheels=true;
    after.velocity=Scale(RunwayForward(c.runway),10);
    after.roll=(c.mission.maxTouchdownRollDegrees+2)*DegreesToRadians;
    auto mission=Flying();
    CrashDiagnostic diagnostic{};
    auto const result=AdvanceMission(c,mission,before,after,&diagnostic);
    auto const& speed=diagnostic.conditions[static_cast<std::size_t>(CrashReason::Speed)];
    auto const& sink=diagnostic.conditions[static_cast<std::size_t>(CrashReason::Sink)];
    auto const& roll=diagnostic.conditions[static_cast<std::size_t>(CrashReason::Roll)];
    Check(result.terminated && mission.outcome==Outcome::Crash && speed.failed && sink.failed && roll.failed,
        "simultaneous impact speed, sink and roll failures are reported");
    Check(Near(speed.value,c.mission.maxTouchdownSpeed+5) && Near(speed.limit,c.mission.maxTouchdownSpeed)
        && Near(sink.value,c.mission.maxTouchdownSink+2) && Near(roll.value,c.mission.maxTouchdownRollDegrees+2),
        "impact diagnostics show the worst pre/post values and actual limits");
    Check(!diagnostic.conditions[static_cast<std::size_t>(CrashReason::Gear)].failed,
        "satisfied landing conditions are not reported");

    mission=Flying();diagnostic={};
    auto invalid=before;invalid.finite=false;
    AdvanceMission(c,mission,before,invalid,&diagnostic);
    Check(mission.outcome==Outcome::Crash && diagnostic.conditions[static_cast<std::size_t>(CrashReason::InvalidState)].failed,
        "an invalid aircraft state has a crash reason");

    mission=Flying();diagnostic={};
    auto safe=Approach(c,300);safe.position=FromRunway(c.runway,300,0,1.8);
    safe.velocity=Scale(RunwayForward(c.runway),10);safe.weightOnWheels=true;
    Check(!AdvanceMission(c,mission,safe,safe,&diagnostic).terminated && mission.outcome==Outcome::Flying,
        "requesting diagnostics does not change a safe touchdown");

    mission=Flying();diagnostic={};
    auto badAttitude=safe;badAttitude.pitch=(c.mission.minTouchdownPitchDegrees-2)*DegreesToRadians;
    badAttitude.heading+=(c.mission.maxTouchdownHeadingDegrees+3)*DegreesToRadians;badAttitude.gearPosition=0;
    AdvanceMission(c,mission,safe,badAttitude,&diagnostic);
    auto const& pitch=diagnostic.conditions[static_cast<std::size_t>(CrashReason::PitchLow)];
    auto const& heading=diagnostic.conditions[static_cast<std::size_t>(CrashReason::Heading)];
    auto const& gear=diagnostic.conditions[static_cast<std::size_t>(CrashReason::Gear)];
    Check(mission.outcome==Outcome::Crash && pitch.failed && heading.failed && gear.failed
        && Near(pitch.value,c.mission.minTouchdownPitchDegrees-2) && Near(heading.value,c.mission.maxTouchdownHeadingDegrees+3)
        && Near(gear.limit,c.mission.gearDownMinimum),"pitch, heading and gear report their failed limit");

    mission=Flying(Phase::Rollout);diagnostic={};
    auto outside=safe;outside.position=FromRunway(c.runway,c.runway.lengthMeters+1,0,2);outside.weightOnWheels=false;
    AdvanceMission(c,mission,safe,outside,&diagnostic);
    Check(mission.outcome==Outcome::Crash && diagnostic.conditions[static_cast<std::size_t>(CrashReason::LandingArea)].failed,
        "leaving the landing area during rollout is reported even without contact");
}

/** @brief A safe touchdown rolls to a confirmed stop; unsafe contacts end in a crash. */
void Landing()
{
    auto c=T6Config();
    c.runway.landingMarginMeters=0;
    auto before=Approach(c,300);before.position=FromRunway(c.runway,300,0,1.8);before.velocity.up=-1;
    auto after=before;after.weightOnWheels=true;after.velocity.up=0;
    auto mission=Flying();
    auto const landed=AdvanceMission(c,mission,before,after);
    Check(mission.touchedDown && mission.phase==Phase::Rollout && !landed.terminated,"a safe touchdown continues to the stop");
    Check(Near(mission.touchdownFromThreshold,300) && Near(Channel(landed,Reward::Touchdown),Weight(c.reward,Reward::Touchdown)*mission.touchdownQuality),
        "touchdown distance and bonus");
    auto rolling=after;rolling.position=FromRunway(c.runway,600,0,1.8);rolling.velocity={};
    auto roll=AdvanceMission(c,mission,after,rolling);
    Check(!roll.terminated && Channel(roll,Reward::Touchdown)==0 && mission.groundRoll>=299,"no repeated touchdown; roll distance");
    for(std::size_t i=0;i<static_cast<std::size_t>(c.mission.stoppedHoldSeconds/c.mission.physicsDt)+2;++i)
    {
        if(mission.outcome!=Outcome::Flying) break;
        roll=AdvanceMission(c,mission,rolling,rolling);
    }
    Check(roll.terminated && mission.outcome==Outcome::Stopped && Near(mission.stopFromThreshold,600),"confirmed stop location");
    Check(Near(Channel(roll,Reward::Stop),Weight(c.reward,Reward::Stop)),"stop reward");

    auto wheelTest=Flying();
    auto precise=after;precise.wheelDataValid=true;
    for(auto& wheel:precise.wheels) wheel={FromRunway(c.runway,302,0,0),true};
    AdvanceMission(c,wheelTest,before,precise);
    Check(Near(wheelTest.touchdownFromThreshold,302),"touchdown is measured at the first physical wheel contact");
    wheelTest=Flying();
    precise.wheels[0].point=FromRunway(c.runway,302,c.runway.widthMeters,0);
    Check(AdvanceMission(c,wheelTest,before,precise).terminated && !wheelTest.touchedDown,"all wheels must remain inside the runway");
    for(int test=0;test<7;++test)
    {
        auto b=before,a=after;
        if(test==0) a.gearPosition=0;
        if(test==1) a.gearPosition=c.mission.gearDownMinimum-0.01;
        if(test==2) b.velocity.up=-c.mission.maxTouchdownSink-1; // The impact solver may already have removed sink.
        if(test==3) a.position=FromRunway(c.runway,300,c.runway.widthMeters,1.8);
        if(test==4) a.structuralContact=true;
        if(test==5) a.roll=(c.mission.maxTouchdownRollDegrees+1)*DegreesToRadians;
        if(test==6) a.heading+=Pi;
        auto crash=Flying();
        auto const result=AdvanceMission(c,crash,b,a);
        Check(result.terminated && crash.outcome==Outcome::Crash && !crash.touchedDown
            && Channel(result,Reward::Touchdown)==0,"an unsafe landing is rejected");
    }
    auto overrun=Flying(Phase::Rollout);overrun.touchedDown=true;
    auto off=rolling;off.position=FromRunway(c.runway,c.runway.lengthMeters+1,0,2);
    Check(AdvanceMission(c,overrun,rolling,off).terminated && overrun.outcome==Outcome::Crash,"runway overrun");
    auto timeout=Flying();
    timeout.steps=static_cast<std::uint64_t>(std::ceil(c.mission.maxSeconds/c.mission.physicsDt));
    auto const timed=AdvanceMission(c,timeout,before,before);
    Check(timed.truncated && !timed.terminated,"the time limit bootstraps");
    auto const noRepeat=AdvanceMission(c,timeout,before,before);
    Check(noRepeat.reward==0 && noRepeat.truncated,"no reward after the episode boundary");
    auto escaped=Flying();
    auto const outside=AdvanceMission(c,escaped,Approach(c,-c.mission.maxDistanceMeters+1),Approach(c,-c.mission.maxDistanceMeters-1));
    Check(outside.truncated && escaped.outcome==Outcome::Timeout,"leaving the recoverable distance truncates the episode");
}

/** @brief The elevated deck plane cannot invent contact above the sea; native contacts are authoritative. */
void FiniteSurfaceContacts()
{
    auto c=T6Config();
    c.runway.origin={100,-200,15};
    for(double const heading:{0.0,90.0,181.0,359.0})
    {
        c.runway.headingDegrees=heading;
        for(auto const position:std::array<std::array<double,2>,4>{{
            {{-60,0}},{{c.runway.lengthMeters+60,0}},{{500,-c.runway.widthMeters}},{{500,c.runway.widthMeters}}}})
        {
            for(bool const native:{false,true})
            {
                auto aircraft=Approach(c,position[0],position[1]);
                aircraft.position=FromRunway(c.runway,position[0],position[1],0.1);
                aircraft.wheelDataValid=native;
                auto mission=Flying();
                auto const step=AdvanceMission(c,mission,aircraft,aircraft);
                Check(!step.terminated && !mission.contactAttempted && mission.outcome==Outcome::Flying,
                    "the elevated runway plane cannot cause an off-deck impact above the sea");
                aircraft.position.up=c.runwayRandomization.seaHeightMeters+0.1;
                aircraft.structuralContact=native;
                mission=Flying();
                auto const impact=AdvanceMission(c,mission,aircraft,aircraft);
                Check(impact.terminated && mission.outcome==Outcome::Crash
                    && Near(Channel(impact,Reward::Crash),Weight(c.reward,Reward::Crash)),
                    "a real sea impact stays a penalized terminal failure");
            }
        }
        auto above=Approach(c,500);above.position=FromRunway(c.runway,500,0,0.1);above.wheelDataValid=true;
        auto mission=Flying();
        Check(!AdvanceMission(c,mission,above,above).terminated && !mission.contactAttempted,
            "native wheel and structure contacts are authoritative above the deck");
        above.structuralContact=true;
        Check(AdvanceMission(c,mission,above,above).terminated && mission.outcome==Outcome::Crash,
            "native structural contact cannot be masked by the geometric fallback");
    }
}

/** @brief The signed logarithm is odd, monotonic, fine near zero and saturates at its maximum. */
void SignedLogCompression()
{
    Check(SignedLog(0,5,2000)==0 && Near(SignedLog(2000,5,2000),1) && SignedLog(1e6,5,2000)==1
        && Near(SignedLog(-37,5,2000),-SignedLog(37,5,2000)),"the compression is odd and saturates");
    float previous=-2;
    for(double value : {-1000.0,-50.0,-1.0,0.0,1.0,2.0,10.0,100.0,1000.0})
    {
        float const current=SignedLog(value,5,2000);
        Check(current>previous,"the compression is strictly monotonic");
        previous=current;
    }
    Check(SignedLog(3,5,2000)-SignedLog(1,5,2000)>0.04F,"two metres near the deck remain distinguishable");
    Check(std::isfinite(SignedLog(std::numeric_limits<double>::quiet_NaN(),5,2000)),"a non-finite value maps to zero");
}

/** @brief Observations live in runway axes: identical local situations observe identically on any runway. */
void ObservationConditioning()
{
    auto const config=T6Config();
    auto const mission=Flying();
    auto reference=config;reference.runway.origin={0,0,15};reference.runway.headingDegrees=0;
    auto const referenceObservation=Observe(reference,mission,Approach(reference,-6000,300));
    for(double heading : {37.0,90.0,181.0,359.0})
    {
        auto c=config;c.runway.origin={-800,1200,14};c.runway.headingDegrees=heading;
        auto const observation=Observe(c,mission,Approach(c,-6000,300));
        for(std::size_t i=0;i<ObservationSize;++i)
            Check(Near(observation[i],referenceObservation[i],1e-5),"observations do not depend on the runway pose");
    }
    auto const c=reference;
    auto aircraft=Approach(c,-6000,300);
    aircraft.velocity.up=-3;aircraft.rollRate=0.1;aircraft.pitchRate=-0.1;aircraft.yawRate=0.05;
    auto const observation=Observe(c,mission,aircraft);
    Check(observation[Slot(ObservationIndex::Cross)]>0
        && Near(observation[Slot(ObservationIndex::Cross)],-Observe(c,mission,Approach(c,-6000,-300))[Slot(ObservationIndex::Cross)],1e-6),
        "the cross-track position is signed and symmetric");
    Check(Near(observation[Slot(ObservationIndex::VelocityUp)],-0.3,1e-6)
        && Near(observation[Slot(ObservationIndex::RollRate)],0.2,1e-6)
        && Near(observation[Slot(ObservationIndex::PitchRate)],-0.2,1e-6),
        "vertical speed and body rates keep their physical signs and scales");
    Check(Near(observation[Slot(ObservationIndex::GlidePathError)],0,1e-6)
        && Near(observation[Slot(ObservationIndex::AirspeedError)],0,1e-6),
        "an aircraft on the glide plane at the target speed observes zero errors");
    auto high=Approach(c,-6000,300);high.position=FromRunway(c.runway,-6000,300,ApproachPathHeight(c,-6000)+80);
    Check(Observe(c,mission,high)[Slot(ObservationIndex::GlidePathError)]>0.3F,"an aircraft above the plane sees a positive error");
    auto low=Approach(c,600,0);low.position=FromRunway(c.runway,600,0,1);
    auto lower=low;lower.position=FromRunway(c.runway,600,0,3);
    Check(Observe(c,mission,lower)[Slot(ObservationIndex::Height)]-Observe(c,mission,low)[Slot(ObservationIndex::Height)]>0.04F,
        "heights near the deck keep a fine resolution");
    aircraft.velocity.up=-100;aircraft.rollRate=5;aircraft.position=FromRunway(c.runway,-50000,9000,9000);
    auto const bounded=Observe(c,mission,aircraft);
    Check(bounded[Slot(ObservationIndex::VelocityUp)]==-1 && bounded[Slot(ObservationIndex::RollRate)]==1
        && bounded[Slot(ObservationIndex::Along)]==-1,"large finite values clip safely");
    for(float value : bounded) Check(std::isfinite(value) && std::abs(value)<=ObservationBound,"inputs are bounded");
    auto invalid=Approach(c);invalid.velocity.up=std::numeric_limits<double>::quiet_NaN();
    for(float value : Observe(c,mission,invalid)) Check(value==0,"an invalid state observes zeros");
    auto rolling=Approach(c,800,0);rolling.weightOnWheels=true;
    auto const ground=Observe(c,Flying(Phase::Rollout),rolling);
    Check(ground[RolloutObservation]==1 && ground[Slot(ObservationIndex::WeightOnWheels)]==1,"rollout and wheel contact are flagged");
}
/** @brief Full-difficulty spawns respect the configured envelopes on both sides of the centreline. */
void Spawns()
{
    auto const c=T6Config();
    std::mt19937 rng(17U);
    for(int sample=0;sample<200;++sample)
    {
        auto const runway=SampleRunway(c,rng);
        Check(runway.headingDegrees>=0 && runway.headingDegrees<360 && runway.origin.up>=14 && runway.origin.up<=16,
            "the runway pose stays inside its randomization ranges");
        for(auto const side : {ApproachSide::Left,ApproachSide::Right})
        {
            auto const aircraft=SampleApproachStart(c,runway,side,1.0,rng);
            auto const local=DeckCoordinates(runway,aircraft.position);
            double const distance=std::hypot(local.north,local.east);
            double const plane=runway.thresholdHeightMeters-local.north*std::tan(c.mission.glideDegrees*DegreesToRadians);
            Check(distance>=c.initial.distanceMeters.minimum-0.01 && distance<=c.initial.distanceMeters.maximum+0.01
                && (side==ApproachSide::Left ? local.east<0 : local.east>0)
                && std::abs(local.up-plane)<=c.initial.heightOffsetMeters+0.01
                && std::abs(aircraft.airspeed-c.mission.approachSpeed)<=c.initial.speedToleranceMps+1e-9,
                "approach starts respect distance, side, height and speed envelopes");
        }
        auto const rolling=SampleRolloutStart(c,runway,rng);
        auto const local=DeckCoordinates(runway,rolling.position);
        Check(local.north>=c.initial.rolloutStart.alongMeters.minimum && local.north<=c.initial.rolloutStart.alongMeters.maximum
            && rolling.airspeed>=c.initial.rolloutStart.speedMps.minimum,"rolling starts are on the runway");
    }
}

/** @brief Difficulty zero is a short aligned final; intermediate difficulties widen the envelope gradually. */
void CurriculumSpawns()
{
    auto const c=T6Config();
    auto const& first=c.training.curriculum.firstDistanceMeters;
    std::mt19937 rng(23U);
    double previousMaximum=0;
    for(double difficulty : {0.0,0.5,1.0})
    {
        double maximumDistance=0,maximumCross=0,maximumHeading=0;
        for(int sample=0;sample<200;++sample)
        {
            auto const runway=SampleRunway(c,rng);
            auto const aircraft=SampleApproachStart(c,runway,sample%2 ? ApproachSide::Right : ApproachSide::Left,difficulty,rng);
            auto const local=DeckCoordinates(runway,aircraft.position);
            double const distance=std::hypot(local.north,local.east);
            maximumDistance=std::max(maximumDistance,distance);
            maximumCross=std::max(maximumCross,std::abs(local.east));
            maximumHeading=std::max(maximumHeading,std::abs(WrapAngle(aircraft.heading-runway.headingDegrees*DegreesToRadians)));
            if(difficulty==0)
                Check(distance>=first.minimum-0.01 && distance<=first.maximum+0.01 && Near(local.east,0,1e-6)
                    && Near(local.up,ApproachPathHeight(c,local.north),1e-6)
                    && Near(WrapAngle(aircraft.heading-runway.headingDegrees*DegreesToRadians),0,1e-9),
                    "difficulty zero starts on the centreline and the glide plane, aligned with the runway");
        }
        Check(maximumDistance>previousMaximum,"a higher difficulty reaches farther starts");
        if(difficulty==1.0)
            Check(maximumDistance>c.initial.distanceMeters.maximum-500 && maximumCross>500
                && maximumHeading>10*DegreesToRadians,"the full difficulty covers the configured envelope");
        previousMaximum=maximumDistance;
    }
}

/** @brief Evaluation campaigns mix rolling starts with stratified approaches up to the curriculum frontier. */
void EvaluationStratification()
{
    constexpr std::size_t Episodes=20;
    constexpr double Share=0.15,Frontier=0.4;
    std::size_t rolling=0;
    double previous=0;
    for(std::size_t episode=0;episode<Episodes;++episode)
    {
        auto const start=EvaluationStart(episode,Episodes,Share,Frontier);
        if(start.rolling) { ++rolling;continue; }
        Check(start.difficulty>previous && start.difficulty<=Frontier+1e-12,
            "airborne evaluation difficulties grow from short finals up to the frontier");
        previous=start.difficulty;
    }
    Check(rolling==3 && EvaluationStart(0,Episodes,Share,Frontier).rolling
        && Near(EvaluationStart(Episodes-1,Episodes,Share,Frontier).difficulty,Frontier),
        "the campaign starts with the rolling share and ends at the curriculum frontier");
    Check(Near(EvaluationStart(3,Episodes,Share,Frontier).difficulty,Frontier/17),
        "the easiest airborne episode is a short final");
    // Curriculum level zero: every approach is the short aligned final.
    for(std::size_t episode=3;episode<Episodes;++episode)
        Check(EvaluationStart(episode,Episodes,Share,0.0).difficulty==0,"level zero evaluates short finals only");
    // Edge cases: no rolling share, a single episode and an almost all-rolling share.
    Check(!EvaluationStart(0,Episodes,0.0,1.0).rolling && Near(EvaluationStart(0,Episodes,0.0,1.0).difficulty,0.05),
        "without a rolling share every episode is airborne");
    Check(!EvaluationStart(0,1,Share,1.0).rolling && EvaluationStart(0,1,Share,1.0).difficulty==1.0,
        "a single-episode campaign flies the frontier");
    Check(!EvaluationStart(Episodes-1,Episodes,0.99,1.0).rolling
        && EvaluationStart(Episodes-2,Episodes,0.99,1.0).rolling,"at least one approach is always evaluated");
    Check(!EvaluationStart(0,0,Share,1.0).rolling,"an empty campaign yields the default airborne start");
}

/** @brief Display helpers keep the camera on the runway and zoom on distant aircraft. */
void DisplayReferences()
{
    auto c=T6Config();
    c.runway.origin={};c.runway.headingDegrees=0;
    auto const camera=DeckCoordinates(c.runway,RunwayCameraPosition(c.runway));
    Check(Near(camera.north,300) && Near(camera.up,8),"the runway camera sits 300 m past the threshold, 8 m high");
    auto shortRunway=c.runway;shortRunway.lengthMeters=200;
    Check(Near(DeckCoordinates(shortRunway,RunwayCameraPosition(shortRunway)).north,100),"a short runway uses its midpoint");
    auto const position=RunwayCameraPosition(c.runway);
    Check(RunwayCameraFieldOfView(position,FromRunway(c.runway,-200,0,100))
        >RunwayCameraFieldOfView(position,FromRunway(c.runway,-6000,0,400)),"distant aircraft get a narrower field of view");
}
} // namespace

int main()
{
    try
    {
        Configuration();
        RewardBalance();
        SignatureCompatibility();
        Geometry();
        TargetSpeed();
        AimPointOnRunway();
        ApproachDirection();
        FlareTrajectory();
        LandingQuality();
        CrashDiagnostics();
        Landing();
        FiniteSurfaceContacts();
        SignedLogCompression();
        ObservationConditioning();
        Spawns();
        CurriculumSpawns();
        EvaluationStratification();
        DisplayReferences();
    }
    catch(std::exception const& error) { std::cerr<<error.what()<<'\n'; return 1; }
    std::cout<<"Flight core regressions passed\n";
}
