#include "test_support.hpp"
#include "../Renderer/flight_display.hpp"
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
    double previousAlong{};
    for(std::size_t point=0;point<c.mission.straightSamples;++point)
    {
        auto const local=ToRunway(c.runway,plan.points[point]);
        Check(Near(local[1],0),"inbound racetrack straight lies on the localizer");
        if(point>0) Check(local[0]>previousAlong,"localizer-aligned racetrack straight points toward the threshold");
        previousAlong=local[0];
    }
    AircraftState ilsPresentation{};
    ilsPresentation.position=plan.points[0];
    ilsPresentation.heading=c.runway.headingDegrees*DegreesToRadians;
    auto const presentation=Guide(c,plan,MissionState{},ilsPresentation);
    Check(presentation.ilsValid
        && std::abs(presentation.localizer)<=c.mission.localizerCaptureDegrees*DegreesToRadians
        && std::abs(presentation.glideError)<=c.mission.glideCaptureDegrees*DegreesToRadians,
        "inbound racetrack straight crosses both LOC and glide capture windows");
    for (double h : {0.,90.,181.,359.})
    {
        auto runway=c.runway; runway.headingDegrees=h;
        auto const xy=ToRunway(runway,FromRunway(runway,-1234,57,99));
        Check(Near(xy[0],-1234) && Near(xy[1],57),"runway rotation/inverse");
    }
    MissionState s{};
    s.completedLaps=c.mission.requiredLaps;
    auto a=Approach(c);
    std::size_t const captureSteps=static_cast<std::size_t>(
        std::ceil(c.mission.captureHoldSeconds/c.mission.physicsDt))+1;
    auto const first=AdvanceMission(c,plan,s,a,a);
    Check(s.phase==Phase::Patrol && !first.terminated,"LOC hold precedes the approach switch");
    auto const g=Guide(c,plan,s,a);
    Check(g.ilsValid && Near(g.localizer,0) && Near(g.glideError,0),"on LOC/glide");
    Check(!s.capturedLocalizer && !s.capturedGlide,"captures require a stable hold");
    double captures=first.rewards[5]+first.rewards[6];
    for(std::size_t i=0;i<2*captureSteps+2;++i)
    { auto const stable=AdvanceMission(c,plan,s,a,a);captures+=stable.rewards[5]+stable.rewards[6]; }
    Check(s.capturedLocalizer && s.capturedGlide && Near(captures,c.reward.weights[5]+c.reward.weights[6]),"confirmed one-shot capture events");
    auto const again=AdvanceMission(c,plan,s,a,a);
    Check(again.rewards[5]==0 && again.rewards[6]==0,"captures must be one-shot");
    auto far=Approach(c,-c.mission.ilsRangeMeters-1);
    AdvanceMission(c,plan,s,a,far);
    Check(s.phase==Phase::Approach && s.capturedLocalizer && s.capturedGlide,
        "a transient ILS loss cannot cancel the approach");
    for(std::size_t step=1;step<captureSteps && !s.approachRecoveryPending;++step)
        AdvanceMission(c,plan,s,far,far);
    Check(s.phase==Phase::Approach && s.approachRecoveryPending
        && !s.capturedLocalizer && !s.capturedGlide && !Guide(c,plan,s,far).targetValid,
        "confirmed ILS loss first keeps a point-free approach recovery grace period");
    for(std::size_t step=0;step<captureSteps;++step) AdvanceMission(c,plan,s,far,far);
    Check(s.phase==Phase::Patrol && !s.capturedLocalizer && !s.capturedGlide,
        "an unrecovered ILS loss eventually returns guidance to the racetrack");
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

    MissionState escaped{};escaped.phase=Phase::Approach;
    auto const insideBoundary=Approach(c,-c.mission.maxDistanceMeters+1);
    auto const outsideBoundary=Approach(c,-c.mission.maxDistanceMeters-1);
    auto const rangeFailure=AdvanceMission(c,plan,escaped,insideBoundary,outsideBoundary);
    Check(rangeFailure.terminated && !rangeFailure.truncated && escaped.outcome==Outcome::Crash
        && rangeFailure.reward==c.reward.weights[static_cast<std::size_t>(Reward::Crash)],
        "leaving the recoverable distance is a penalized terminal failure");
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
    MissionState wheelTest{};wheelTest.phase=Phase::Approach;
    auto precise=after;precise.wheelDataValid=true;
    for(auto& wheel:precise.wheels) wheel={FromRunway(c.runway,302,0,0),true};
    AdvanceMission(c,plan,wheelTest,before,precise);
    Check(Near(wheelTest.touchdownFromThreshold,302),"touchdown is measured at the first physical wheel contact");
    wheelTest={};wheelTest.phase=Phase::Approach;
    precise.wheels[0].point=FromRunway(c.runway,302,c.runway.widthMeters,0);
    Check(AdvanceMission(c,plan,wheelTest,before,precise).terminated && !wheelTest.touchedDown,"all physical wheels must remain inside runway");
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
    for (int test=0; test<47; ++test)
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
        if (test==8) json["initial"]["runwayAlignedProbability"]=-0.01;
        if (test==9) json["initial"]["runwayAlignedProbability"]=1.01;
        if (test==10) json["initial"]["runwayAlignedProbability"]=true;
        if (test==11) json["initial"].erase("runwayAlignedProbability");
        if (test==12) json["initial"]["bearingDegrees"]={-20,20};
        if (test==13) json["mission"]["patrolLookaheadSeconds"]=0;
        if (test==14) json["initial"]["distanceMeters"]={5000,
            json["mission"]["ilsRangeMeters"].get<double>()+1};
        if (test==15) json["mission"]["racetrackCenterCross"]=0;
        if (test==16) json["training"]["ilsCaptureScale"]=0.99;
        if (test==17) json["training"]["ilsCaptureScale"]=100.0;
        if (test==18) json["training"].erase("ilsCaptureScale");
        if (test==19) json["mission"]["patrolHeadingToleranceDegrees"]=0;
        if (test==20) json["reward"].erase("glideToleranceDegrees");
        if (test==21) json["reward"].erase("slopeToleranceDegrees");
        if (test==22) json["reward"].erase("localizerCourseBlend");
        if (test==23) json["reward"].erase("localizerInterceptGain");
        if (test==24) json["reward"].erase("localizerInterceptLimitDegrees");
        if (test==25) json["normalization"]["localizerDegrees"]=35.0;
        if (test==26) json["training"]["curriculum"]["stage"]="invalid";
        if (test==27) json["training"]["curriculum"]["locReminderProbability"]=-0.1;
        if (test==28) json["training"]["curriculum"]["glideOffsetMeters"]=1e7;
        if (test==29) json["training"]["curriculum"]["gateHeadingDegrees"]=0;
        if (test==30) json["training"]["curriculum"]["gateRouteDegrees"]=50;
        if (test==31) json["training"]["curriculum"]["gateSlopeDegrees"]=10;
        if (test==32) json["mission"]["patrolEnabled"]=1;
        if (test==33) json["mission"].erase("patrolEnabled");
        if (test==34) json["training"]["curriculum"]["gateSpeedToleranceMps"]=0;
        if (test==35) json["training"]["curriculum"]["gateSpeedToleranceMps"]=16;
        if (test==36) json["training"]["curriculum"]["gateRollDegrees"]=0;
        if (test==37) json["training"]["curriculum"]["gateRateDegreesPerSecond"]=31;
        if (test==38) json["training"]["curriculum"]["captureMaxSeconds"]=1;
        if (test==39) json["training"]["curriculum"]["captureStart"]["speedToleranceMps"]=20;
        if (test==40) json["training"]["curriculum"]["captureStart"]["rollDegrees"]=31;
        if (test==41) json["training"]["curriculum"]["captureStart"]["pitchDegrees"]={4,1};
        if (test==42) json["training"]["curriculum"]["captureStart"]["slopeToleranceDegrees"]=10;
        if (test==43) json["training"]["curriculum"]["captureStart"]["localizerHeightOffsetMeters"]=10000;
        if (test==44) json["training"]["curriculum"].erase("captureStart");
        if (test==45) json["training"]["curriculum"]["captureStart"]["unknown"]=true;
        if (test==46)
        {
            json["initial"]["bearingDegrees"]={30,32};
            json["runwayRandomization"]["pitchAmplitudeDegrees"]={0,0};
            json["runwayRandomization"]["rollAmplitudeDegrees"]={0,0};
            json["training"]["curriculum"]["captureStart"]["localizerHeightOffsetMeters"]=
                15+4000*std::cos(32*DegreesToRadians)*std::tan(3*DegreesToRadians)-20+1;
        }
        { std::ofstream stream(path); stream<<json; }
        Rejects([&] { LoadConfig(path); },"invalid JSON accepted");
    }
    auto c=LoadConfig(PPO_FLIGHT_CONFIG);
    Check(!c.mission.patrolEnabled,"delivered scenario disables patrol without removing its settings");
    Check(Near(c.initial.runwayAlignedProbability,1.0/3.0),"balanced LOC/glide-start probability");
    Check(Near(c.training.ilsCaptureScale,2.0),"training-only ILS capture scale");
    Check(Near(c.mission.maxDistanceMeters,18000),"bounded recovery margin beyond ILS range");
    for(double probability : {std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()})
    {
        c.initial.runwayAlignedProbability=probability;
        Rejects([&] { Validate(c); },"non-finite spawn probability accepted");
    }
}

void CaptureLessonTermination(Config c)
{
    auto const plan=MakeFlightPlan(c);
    auto aircraft=Approach(c,-5000);
    aircraft.velocity=Add(Scale(RunwayForward(c.runway),c.mission.approachSpeed),
        Scale(RunwayNormal(c.runway),-c.mission.approachSpeed*std::tan(c.mission.glideDegrees*DegreesToRadians)));
    c.training.gateHoldSeconds=0.08;
    for(auto const stage : {CurriculumStage::Loc,CurriculumStage::Glide})
    {
        c.training.curriculumStage=stage;
        MissionState mission{};mission.phase=Phase::Approach;
        mission.capturedLocalizer=true;mission.capturedGlide=true;
        mission.rewardedLocalizerCapture=true;mission.rewardedGlideCapture=true;
        auto const initial=mission;
        for(int step=0;step<3;++step)
            Check(!AdvanceMission(c,plan,mission,aircraft,aircraft).terminated,"partial hold cannot finish a lesson");
        auto const partial=Observe(c,plan,mission,aircraft);
        Check(Near(partial[41],mission.captureGateSeconds/c.training.gateHoldSeconds,1e-6),
            "policy observes remaining stabilized completion through the existing hold slot");
        auto shaking=aircraft;shaking.pitchRate=2*c.training.gateRateDegreesPerSecond*DegreesToRadians;
        AdvanceMission(c,plan,mission,aircraft,shaking);
        Check(mission.captureGateSeconds==0,"a physical oscillation resets the continuous capture hold");
        StepResult result{};
        for(int step=0;step<20 && mission.outcome==Outcome::Flying;++step)
            result=AdvanceMission(c,plan,mission,aircraft,aircraft);
        Check(result.terminated && !result.truncated && mission.outcome==Outcome::CaptureCompleted
            && !mission.touchedDown && result.rewards[static_cast<std::size_t>(Reward::Stop)]==0
            && result.rewards[static_cast<std::size_t>(Reward::Touchdown)]==0,
            "stable LOC/Glide success ends its own task without fabricating a landing reward");
        auto const completed=AdvanceMission(c,plan,mission,aircraft,aircraft);
        Check(completed.terminated && completed.reward==0 && Observe(c,plan,mission,aircraft)[41]==1,
            "completed lesson cannot emit rewards or advance again");
        auto failure=initial;failure.captureGateSeconds=c.training.gateHoldSeconds-c.mission.physicsDt;
        auto contact=aircraft;contact.structuralContact=true;
        auto const crashed=AdvanceMission(c,plan,failure,aircraft,contact);
        Check(crashed.terminated && failure.outcome==Outcome::Crash,
            "a crash is authoritative even at the end of a capture hold");
        auto timeout=initial;timeout.steps=static_cast<std::uint64_t>(std::ceil(c.mission.maxSeconds/c.mission.physicsDt));
        auto const expired=AdvanceMission(c,plan,timeout,aircraft,aircraft);
        Check(expired.truncated && !expired.terminated && timeout.outcome==Outcome::Timeout,
            "unfinished capture at the time budget remains a bootstrap truncation");
        auto briefLoss=initial;briefLoss.captureGateSeconds=c.training.gateHoldSeconds-c.mission.physicsDt;
        auto edge=aircraft;auto const local=DeckCoordinates(c.runway,edge.position);
        edge.position=FromRunway(c.runway,local.north,
            -local.north*std::tan(1.1*c.mission.localizerCaptureDegrees*DegreesToRadians),local.up);
        auto const lost=AdvanceMission(c,plan,briefLoss,edge,edge);
        Check(!lost.terminated && briefLoss.capturedLocalizer && briefLoss.captureGateSeconds==0,
            "hysteresis cannot turn an out-of-tolerance snapshot into stabilized success");
    }
    c.training.curriculumStage=CurriculumStage::Joint;
    MissionState finalApproach{};finalApproach.phase=Phase::Approach;
    finalApproach.capturedLocalizer=true;finalApproach.capturedGlide=true;
    for(int step=0;step<30;++step)
        Check(!AdvanceMission(c,plan,finalApproach,aircraft,aircraft).terminated,
            "Joint must continue from captured beams through touchdown and stop");
    Check(finalApproach.captureGateSeconds==0,"Joint does not accumulate pedagogical completion");
}

void ApproachOnly(Config c)
{
    Check(!c.mission.patrolEnabled && MakeFlightPlan(c).count==0,
        "disabled patrol publishes no racetrack geometry");
    auto enabled=c;enabled.mission.patrolEnabled=true;
    auto const stalePlan=MakeFlightPlan(enabled);
    Check(stalePlan.count>1,"patrol geometry can be explicitly re-enabled");
    c.reward.weights[static_cast<std::size_t>(Reward::Path)]=100;
    auto const onBeam=Approach(c,-5000);
    auto offGlide=onBeam;offGlide.position.up+=500;
    auto const offLoc=Approach(c,-5000,5000);
    std::array<AircraftState,5> const samples{onBeam,offGlide,onBeam,offLoc,onBeam};
    std::size_t const holdSteps=static_cast<std::size_t>(
        std::ceil(c.mission.captureHoldSeconds/c.mission.physicsDt))+1;
    MissionState s{};
    s.pathQualified=true;s.rallyCorridorLatched=true;s.waypointsPassed=5;s.completedLaps=1;
    s.approachRecoveryPending=true;s.approachRecoverySeconds=c.mission.captureHoldSeconds;
    Check(!Guide(c,stalePlan,s,onBeam).targetValid,
        "a stale patrol phase and plan cannot expose a guidance point while disabled");
    double localizerBonus{},glideBonus{};
    for(int cycle=0;cycle<3;++cycle)
    {
        for(std::size_t sample=0;sample<samples.size();++sample)
        {
            auto const& aircraft=samples[sample];
            for(std::size_t step=0;step<4*holdSteps;++step)
            {
                auto const result=AdvanceMission(c,stalePlan,s,aircraft,aircraft);
                auto const guidance=Guide(c,stalePlan,s,aircraft);
                auto const observation=Observe(c,stalePlan,s,aircraft);
                localizerBonus+=result.rewards[static_cast<std::size_t>(Reward::EnterLocalizer)];
                glideBonus+=result.rewards[static_cast<std::size_t>(Reward::EnterGlide)];
                Check(s.phase==Phase::Approach && !s.approachRecoveryPending
                    && !s.pathQualified && s.waypointsPassed==0 && s.completedLaps==0
                    && !guidance.targetValid && guidance.mode!=GuidanceMode::Patrol
                    && guidance.target.north==aircraft.position.north
                    && guidance.target.east==aircraft.position.east
                    && guidance.target.up==aircraft.position.up
                    && result.rewards[static_cast<std::size_t>(Reward::Path)]==0
                    && observation[0]==0 && observation[1]==0 && observation[2]==0
                    && observation[32]==0 && observation[33]==1,
                    "approach-only cannot restore HIPP, its point or its reward across repeated beam losses");
                Check(!result.terminated && !result.truncated,"beam loss alone is not an episode boundary");
            }
            if(sample==3)
                Check(!s.capturedLocalizer && !s.capturedGlide,"LOC loss still releases both captures");
            else if(sample==1)
                Check(s.capturedLocalizer && !s.capturedGlide,"glide loss preserves LOC capture");
            else Check(s.capturedLocalizer && s.capturedGlide,"both beams can still be recaptured");
        }
    }
    Check(Near(localizerBonus,c.reward.weights[static_cast<std::size_t>(Reward::EnterLocalizer)])
        && Near(glideBonus,c.reward.weights[static_cast<std::size_t>(Reward::EnterGlide)]),
        "approach-only recaptures do not repeat entry bonuses");
    Landing(c); // Real landing, crash and timeout rules remain authoritative without patrol.
}
void OrderedNavigation(Config const& config)
{
    auto c=config;
    auto const plan=MakeFlightPlan(c);
    MissionState notArmed{};
    auto const near=Approach(c,-1000);
    AdvanceMission(c,plan,notArmed,near,near);
    Check(notArmed.phase==Phase::Patrol,"distance alone cannot skip prescribed racetrack laps");
    MissionState flight{};
    AircraftState a{};a.position=plan.points[0];a.airspeed=c.mission.patrolSpeed;
    for(std::size_t step=0;step<10000 && flight.completedLaps==0;++step)
    {
        auto const target=plan.points[flight.waypoint];auto const difference=Subtract(target,a.position);
        double const length=std::sqrt(Dot(difference,difference));
        auto next=a;
        next.position=Add(a.position,Scale(difference,std::min(10.0,length)/length));
        // Keep this path-validation fixture outside the approach capture gate;
        // the real aligned inbound branch intentionally gives LOC priority.
        next.heading=std::atan2(difference.east,difference.north)+Pi/2;
        AdvanceMission(c,plan,flight,a,next);a=next;
    }
    Check(flight.completedLaps==1,"one complete ordered physical circuit qualifies landing return");
    MissionState shortcut{};shortcut.pathQualified=true;shortcut.waypointsPassed=3;
    auto before=a,after=a;before.position=plan.points[0];after.position=plan.points[3];
    AdvanceMission(c,plan,shortcut,before,after);
    Check(!shortcut.pathQualified && shortcut.waypointsPassed==0 && shortcut.completedLaps==0,"teleport/shortcut invalidates lap progress");
    MissionState stopped{};a.position=plan.points[0];
    auto const inboundSegment=Subtract(plan.points[0],plan.points[plan.count-1]);
    a.heading=std::atan2(inboundSegment.east,inboundSegment.north);
    auto const stationary=AdvanceMission(c,plan,stopped,a,a);
    Check(stationary.rewards[0]<=0 && !stopped.pathQualified
        && stopped.waypointsPassed==0 && stopped.completedLaps==0,
        "stationary aircraft earns no path progression");
}
void AlignedStarts(Config c)
{
    std::array<double,4> const headings{0,90,181,359};
    for(double probability : {0.0,1.0,1.0/3.0})
    {
        c.initial.runwayAlignedProbability=probability;
        std::mt19937 rng(279),same(279);
        SpawnPlacementSampler placements{},samePlacements{};
        std::size_t alignedCount=0,left=0,right=0;
        std::array<bool,3> familyBlock{};
        double firstDistance=0,firstSpeed=0,firstRoll=0;
        bool variedDistance=false,variedSpeed=false,variedRoll=false;
        for(std::size_t sample=0;sample<1000;++sample)
        {
            auto motion=SampleRunway(c,rng),copy=SampleRunway(c,same);
            motion.base.headingDegrees=headings[sample%headings.size()];copy.base.headingDegrees=motion.base.headingDegrees;
            auto const runway=RunwayAt(motion,0),replicaRunway=RunwayAt(copy,0);
            auto const placement=placements.Next(probability,rng);
            auto const replicaPlacement=samePlacements.Next(probability,same);
            auto const aircraft=SampleAircraft(c,runway,placement,rng);
            auto const replica=SampleAircraft(c,replicaRunway,replicaPlacement,same);
            auto const local=DeckCoordinates(runway,aircraft.position);
            double const offset=WrapAngle(aircraft.heading-runway.headingDegrees*DegreesToRadians);
            bool const aligned=Near(local.east,0,1e-7) && Near(offset,0,1e-10);
            if(aligned)
            {
                ++alignedCount;
                double const expectedHeight=runway.thresholdHeightMeters-local.north*std::tan(c.mission.glideDegrees*DegreesToRadians);
                double const slope=std::atan2(aircraft.velocity.up,std::hypot(aircraft.velocity.north,aircraft.velocity.east));
                Check(local.north<0 && Near(local.up,expectedHeight,1e-7)
                    && Near(slope,-c.mission.glideDegrees*DegreesToRadians,1e-10)
                    && Dot(aircraft.velocity,RunwayForward(runway))>0,
                    "aligned starts are inbound on the exact LOC/glide geometry");
            }
            else
            {
                if(local.east<0) ++left;
                if(local.east>0) ++right;
                Check(local.east*offset<0 && std::cos(offset)>0,
                    "left/right starts point inward and never away from the runway");
                Check(local.up>=c.initial.heightMeters.minimum-1e-7 && local.up<=c.initial.heightMeters.maximum+1e-7,
                    "side starts keep the configured height envelope");
            }
            double const distance=std::hypot(local.north,local.east);
            Check(distance>=c.initial.distanceMeters.minimum-1e-7 && distance<=c.initial.distanceMeters.maximum+1e-7,
                "all spawn families keep the configured distance envelope");
            Check(placement==replicaPlacement && aircraft.position.north==replica.position.north
                && aircraft.heading==replica.heading && rng==same,
                "spawn mixture is reproducible for the same seed");
            if(probability==1.0/3.0)
            {
                familyBlock[static_cast<std::size_t>(placement)]=true;
                if(sample%familyBlock.size()==familyBlock.size()-1)
                {
                    Check(familyBlock[0] && familyBlock[1] && familyBlock[2],
                        "each balanced block contains left, LOC/glide and right exactly once");
                    familyBlock={};
                }
            }
            if(sample==0) { firstDistance=distance;firstSpeed=aircraft.airspeed;firstRoll=aircraft.roll; }
            else
            {
                variedDistance=variedDistance || !Near(distance,firstDistance);
                variedSpeed=variedSpeed || !Near(aircraft.airspeed,firstSpeed);
                variedRoll=variedRoll || !Near(aircraft.roll,firstRoll);
            }
        }
        Check(variedDistance && variedSpeed && variedRoll,"aligned starts must retain the other random parameters");
        if(probability==0) Check(alignedCount==0,"zero probability retains broad random starts");
        if(probability==1) Check(alignedCount==1000,"unit probability aligns every start on all runway headings");
        if(probability==1.0/3.0) Check(alignedCount>=333 && alignedCount<=334,
            "balanced scheduler preserves the configured LOC/glide share");
        if(probability<1) Check(std::max(left,right)-std::min(left,right)<=1,
            "side starts remain prefix-balanced across the runway");
    }
}

void CapturePriority(Config const& c)
{
    auto const plan=MakeFlightPlan(c);
    auto onBeam=Approach(c,-6000);
    MissionState crossing{};
    auto perpendicular=onBeam;perpendicular.heading+=Pi/2;
    std::size_t const holdSteps=static_cast<std::size_t>(std::ceil(c.mission.captureHoldSeconds/c.mission.physicsDt))+1;
    for(std::size_t step=0;step<holdSteps;++step) AdvanceMission(c,plan,crossing,perpendicular,perpendicular);
    Check(crossing.phase==Phase::Patrol && !crossing.capturedLocalizer,
        "crossing the localizer with an incompatible heading cannot capture it");

    MissionState completedPatrol{};
    completedPatrol.completedLaps=c.mission.requiredLaps;
    auto const outsideCapture=Approach(c,-6000,1000);
    AdvanceMission(c,plan,completedPatrol,outsideCapture,outsideCapture);
    Check(completedPatrol.phase==Phase::Patrol && !completedPatrol.capturedLocalizer
        && Guide(c,plan,completedPatrol,outsideCapture).mode==GuidanceMode::Patrol,
        "a completed racetrack cannot switch guidance before confirmed LOC capture");

    MissionState inbound{};
    double localizerReward{},glideReward{};
    for(std::size_t step=0;step<holdSteps;++step)
    {
        auto const result=AdvanceMission(c,plan,inbound,onBeam,onBeam);
        localizerReward+=result.rewards[static_cast<std::size_t>(Reward::EnterLocalizer)];
        glideReward+=result.rewards[static_cast<std::size_t>(Reward::EnterGlide)];
    }
    Check(inbound.phase==Phase::Approach && inbound.capturedLocalizer && !inbound.capturedGlide
        && Guide(c,plan,inbound,onBeam).mode==GuidanceMode::Localizer
        && !Guide(c,plan,inbound,onBeam).targetValid
        && Near(localizerReward,c.reward.weights[static_cast<std::size_t>(Reward::EnterLocalizer)]) && glideReward==0,
        "confirmed LOC capture removes all point guidance before a completed lap");
    for(std::size_t step=0;step<holdSteps;++step)
    {
        auto const result=AdvanceMission(c,plan,inbound,onBeam,onBeam);
        glideReward+=result.rewards[static_cast<std::size_t>(Reward::EnterGlide)];
    }
    Check(inbound.capturedGlide && Guide(c,plan,inbound,onBeam).mode==GuidanceMode::Glide
        && !Guide(c,plan,inbound,onBeam).targetValid
        && Near(glideReward,c.reward.weights[static_cast<std::size_t>(Reward::EnterGlide)]),
        "glide capture follows LOC capture and removes the competing threshold point");

    auto committedApproach=inbound;
    auto const pastThreshold=Approach(c,100);
    for(std::size_t step=0;step<holdSteps;++step)
        AdvanceMission(c,plan,committedApproach,pastThreshold,pastThreshold);
    Check(committedApproach.phase==Phase::Approach && committedApproach.capturedLocalizer
        && committedApproach.capturedGlide && !Guide(c,plan,committedApproach,pastThreshold).targetValid,
        "an approach committed across the threshold keeps both captures until touchdown");

    auto glideLoss=inbound;
    auto aboveGlide=onBeam;
    aboveGlide.position.up+=1000;
    AdvanceMission(c,plan,glideLoss,onBeam,aboveGlide);
    Check(glideLoss.phase==Phase::Approach && glideLoss.capturedLocalizer && glideLoss.capturedGlide,
        "a transient glide deviation cannot cancel the approach");
    for(std::size_t step=1;step<holdSteps && !glideLoss.approachRecoveryPending;++step)
        AdvanceMission(c,plan,glideLoss,aboveGlide,aboveGlide);
    Check(glideLoss.phase==Phase::Approach && glideLoss.approachRecoveryPending
        && glideLoss.approachRecoveryRequiresGlide && glideLoss.capturedLocalizer
        && !glideLoss.capturedGlide
        && !Guide(c,plan,glideLoss,aboveGlide).targetValid,
        "confirmed glide loss keeps a short point-free approach recovery window");
    auto recoveredBeforeHipp=glideLoss;
    for(std::size_t step=0;step<holdSteps;++step)
        AdvanceMission(c,plan,recoveredBeforeHipp,onBeam,onBeam);
    Check(recoveredBeforeHipp.phase==Phase::Approach && recoveredBeforeHipp.capturedLocalizer
        && recoveredBeforeHipp.capturedGlide
        && !recoveredBeforeHipp.approachRecoveryPending
        && !Guide(c,plan,recoveredBeforeHipp,onBeam).targetValid,
        "fresh glide capture during the grace period prevents a HIPP reactivation");
    for(std::size_t step=0;step<holdSteps;++step)
        AdvanceMission(c,plan,glideLoss,aboveGlide,aboveGlide);
    Check(glideLoss.phase==Phase::Patrol && !glideLoss.approachRecoveryPending,
        "unrecovered glide loss eventually returns to the racetrack");

    auto const outsideBeam=Approach(c,-6000,5000);
    inbound.pathQualified=true;
    inbound.waypointsPassed=5;
    AdvanceMission(c,plan,inbound,onBeam,outsideBeam);
    Check(inbound.capturedLocalizer && inbound.capturedGlide,
        "a transient ILS dropout does not release either capture");
    for(std::size_t step=1;step<holdSteps && !inbound.approachRecoveryPending;++step)
        AdvanceMission(c,plan,inbound,outsideBeam,outsideBeam);
    Check(inbound.phase==Phase::Approach && inbound.approachRecoveryPending
        && !inbound.capturedLocalizer && !inbound.capturedGlide
        && !Guide(c,plan,inbound,outsideBeam).targetValid,
        "sustained LOC loss does not immediately reactivate racetrack guidance");
    for(std::size_t step=0;step<holdSteps;++step)
        AdvanceMission(c,plan,inbound,outsideBeam,outsideBeam);
    Check(inbound.phase==Phase::Patrol && !inbound.capturedLocalizer && !inbound.capturedGlide
        && !inbound.pathQualified && inbound.waypointsPassed==0
        && Guide(c,plan,inbound,outsideBeam).mode==GuidanceMode::Patrol,
        "unrecovered LOC loss returns to racetrack reacquisition and clears partial lap credit");

    double repeatedCaptureReward{};
    for(std::size_t step=0;step<holdSteps;++step)
    {
        auto const result=AdvanceMission(c,plan,inbound,onBeam,onBeam);
        repeatedCaptureReward+=result.rewards[static_cast<std::size_t>(Reward::EnterLocalizer)]
            +result.rewards[static_cast<std::size_t>(Reward::EnterGlide)];
    }
    Check(inbound.capturedLocalizer && !inbound.capturedGlide
        && !Guide(c,plan,inbound,onBeam).targetValid,
        "LOC reacquisition cannot reactivate a runway or racetrack point");
    for(std::size_t step=0;step<holdSteps;++step)
    {
        auto const result=AdvanceMission(c,plan,inbound,onBeam,onBeam);
        repeatedCaptureReward+=result.rewards[static_cast<std::size_t>(Reward::EnterLocalizer)]
            +result.rewards[static_cast<std::size_t>(Reward::EnterGlide)];
    }
    auto const glideGuidance=Guide(c,plan,inbound,onBeam);
    auto const observation=Observe(c,plan,inbound,onBeam);
    Check(inbound.capturedLocalizer && inbound.capturedGlide && !glideGuidance.targetValid
        && observation[0]==0 && observation[1]==0 && observation[2]==0
        && repeatedCaptureReward==0,
        "LOC/glide recapture suppresses the threshold point without repeating entry bonuses");

    auto const flareAircraft=Approach(c,-50);
    auto const flareGuidance=Guide(c,plan,inbound,flareAircraft);
    auto rollout=inbound;rollout.phase=Phase::Rollout;
    auto const rolloutGuidance=Guide(c,plan,rollout,flareAircraft);
    Check(flareGuidance.mode==GuidanceMode::Flare && !flareGuidance.targetValid
        && flareGuidance.target.north==flareAircraft.position.north
        && flareGuidance.target.east==flareAircraft.position.east
        && rolloutGuidance.mode==GuidanceMode::Rollout && !rolloutGuidance.targetValid
        && rolloutGuidance.target.north==flareAircraft.position.north
        && rolloutGuidance.target.east==flareAircraft.position.east,
        "flare and rollout cannot reactivate a threshold guidance line");
}
void RandomizationAndMotion(Config const& c)
{
    std::mt19937 rng(91),same(91);
    SpawnPlacementSampler placements{},samePlacements{};
    bool differentPositions=false,differentAttitudes=false;
    AircraftState first{};
    for (std::size_t sample=0;sample<1000;++sample)
    {
        auto const motion=SampleRunway(c,rng),repeated=SampleRunway(c,same);
        auto const runway=RunwayAt(motion,0),copy=RunwayAt(repeated,0);
        auto const placement=placements.Next(c.initial.runwayAlignedProbability,rng);
        auto const replicaPlacement=samePlacements.Next(c.initial.runwayAlignedProbability,same);
        auto const aircraft=SampleAircraft(c,runway,placement,rng);
        auto const replica=SampleAircraft(c,copy,replicaPlacement,same);
        auto const local=DeckCoordinates(runway,aircraft.position);
        double const distance=std::hypot(local.north,local.east);
        Check(distance>=c.initial.distanceMeters.minimum-1e-7 && distance<=c.initial.distanceMeters.maximum+1e-7,"random spawn distance bounds");
        auto episode=c; episode.runway=runway;
        auto const plan=MakeFlightPlan(episode);
        double nearest=std::numeric_limits<double>::max();
        for(std::size_t i=0;i<plan.count;++i)
        {
            auto const delta=Subtract(aircraft.position,plan.points[i]);
            nearest=std::min(nearest,std::sqrt(Dot(delta,delta)));
        }
        Check(local.north<0 && nearest<3*c.normalization.path,
            "default starts stay on the approach side within bounded racetrack rally range");
        if(std::abs(local.east)<1e-7)
            Check(Near(local.up,c.runway.thresholdHeightMeters-local.north*std::tan(c.mission.glideDegrees*DegreesToRadians),1e-7),
                "centreline spawn is exactly on glide");
        else Check(local.up>=c.initial.heightMeters.minimum-1e-7 && local.up<=c.initial.heightMeters.maximum+1e-7,
            "left/right spawn height bounds");
        Check(aircraft.airspeed>=c.initial.speedMps.minimum && aircraft.airspeed<=c.initial.speedMps.maximum,"random spawn speed bounds");
        Check(aircraft.roll>=c.initial.rollDegrees.minimum*DegreesToRadians && aircraft.roll<=c.initial.rollDegrees.maximum*DegreesToRadians,"random roll bounds");
        Check(aircraft.pitch>=c.initial.pitchDegrees.minimum*DegreesToRadians && aircraft.pitch<=c.initial.pitchDegrees.maximum*DegreesToRadians,"random pitch bounds");
        Check(placement==replicaPlacement && aircraft.position.north==replica.position.north
            && aircraft.heading==replica.heading && aircraft.airspeed==replica.airspeed,
            "seeded reproducible randomization");
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
void DisplayReferences(Config const& config)
{
    for(double heading : {0.0,90.0,181.0,359.0})
    {
        auto c=config;
        c.runway.origin={1200,-700,18}; c.runway.headingDegrees=heading;
        c.runway.pitch=0.2*DegreesToRadians; c.runway.roll=-0.4*DegreesToRadians;
        c.runway.velocity={2,1,0.3}; c.runway.angularVelocity={0.0001,0.0002,0};
        auto const display=MakeIlsDisplayGeometry(c.runway,c.mission.ilsRangeMeters,
            c.mission.glideDegrees,c.mission.localizerCaptureDegrees,c.mission.glideCaptureDegrees);
        auto const locOrigin=DeckCoordinates(c.runway,display.localizerOrigin);
        auto const gsOrigin=DeckCoordinates(c.runway,display.glideOrigin);
        Check(Near(locOrigin.north,0) && Near(locOrigin.east,0) && locOrigin.up>0 && locOrigin.up<1,
            "LOC starts above threshold without deck depth fighting");
        Check(Near(gsOrigin.north,0) && Near(gsOrigin.east,0) && Near(gsOrigin.up,c.runway.thresholdHeightMeters),
            "GLIDE starts at configured threshold crossing height");
        auto const left=DeckCoordinates(c.runway,display.localizerLeft);
        auto const right=DeckCoordinates(c.runway,display.localizerRight);
        Check(Near(left.north,-c.mission.ilsRangeMeters) && Near(right.north,left.north)
            && Near(left.east,-right.east) && left.east<0,"LOC envelope is runway-aligned and symmetric");
        Check(Near(std::atan2(right.east,-right.north)/DegreesToRadians,c.mission.localizerCaptureDegrees),
            "LOC envelope follows configured capture angle");
        MissionState approach{}; approach.phase=Phase::Approach;
        AircraftState onGlide{}; onGlide.position=display.glideEnd; onGlide.pitch=8*DegreesToRadians;
        onGlide.velocity=Add(DeckVelocity(c.runway,onGlide.position),
            Add(Scale(RunwayForward(c.runway),100*std::cos(3*DegreesToRadians)),
                Scale(RunwayNormal(c.runway),-100*std::sin(3*DegreesToRadians))));
        auto const guidance=Guide(c,MakeFlightPlan(c),approach,onGlide);
        Check(Near(guidance.localizer,0) && Near(guidance.glideError,0),"displayed centreline agrees with real ILS guidance");
        Check(Near(DisplaySlopeDegrees(guidance),-3),"actual slope is deck-relative velocity, independent of aircraft pitch");
        for(auto const& edge : {display.glideLow,display.glideHigh})
        {
            auto const point=DeckCoordinates(c.runway,edge);
            double const angle=std::atan2(point.up-c.runway.thresholdHeightMeters,-point.north)/DegreesToRadians;
            Check(Near(std::abs(angle-c.mission.glideDegrees),c.mission.glideCaptureDegrees),"glide envelope follows configured capture angle");
        }
    }
    Guidance stopped{};
    Check(Near(DisplaySlopeDegrees(stopped),0),"stationary slope stays finite");
    stopped.relativeVelocity={0,100,100};
    Check(Near(DisplaySlopeDegrees(stopped),45),"climb uses full horizontal speed and positive slope");
}
}
int main()
{
    try
    {
        auto c=LoadConfig(PPO_FLIGHT_CONFIG);
        ApproachOnly(c);Configuration();CaptureLessonTermination(c);
        c.mission.patrolEnabled=true; // Keep all legacy patrol/recovery regressions active.
        Geometry(c);Landing(c);OrderedNavigation(c);AlignedStarts(c);CapturePriority(c);RandomizationAndMotion(c);DisplayReferences(c);
    }
    catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
    std::cout<<"Flight mission/configuration regressions passed\n";
}
