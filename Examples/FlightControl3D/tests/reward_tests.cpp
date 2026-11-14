#include "test_support.hpp"
#include <limits>
using namespace ppo::examples::flight;

namespace {
AircraftState At(Config const& c,double along,double cross,double height)
{
    AircraftState aircraft{};
    aircraft.position=FromRunway(c.runway,along,cross,height);
    aircraft.airspeed=c.mission.approachSpeed;
    aircraft.heading=c.runway.headingDegrees*DegreesToRadians;
    return aircraft;
}
AircraftState WithCourse(Config const& c,AircraftState aircraft,double courseDegrees)
{
    double const course=courseDegrees*DegreesToRadians;
    aircraft.velocity=Add(Scale(RunwayForward(c.runway),aircraft.airspeed*std::cos(course)),
        Scale(RunwayRight(c.runway),aircraft.airspeed*std::sin(course)));
    return aircraft;
}
StepResult ApproachStep(Config const& c,AircraftState const& before,AircraftState const& after)
{
    MissionState mission{};
    mission.phase=Phase::Approach;
    mission.capturedLocalizer=true;
    return AdvanceMission(c,MakeFlightPlan(c),mission,before,after);
}
double Channel(StepResult const& step,Reward reward)
{
    return step.rewards[static_cast<std::size_t>(reward)];
}
void Convergence(Config const& c)
{
    auto const far=WithCourse(c,At(c,-6000,2000,700),0);
    auto const closer=WithCourse(c,At(c,-6000,1800,600),0);
    auto const middle=WithCourse(c,At(c,-6000,1900,650),0);
    auto const improvement=ApproachStep(c,far,closer);
    auto const regression=ApproachStep(c,closer,far);
    MissionState mission{};mission.phase=Phase::Approach;mission.capturedLocalizer=true;
    auto const initial=ShapingPotentials(c,{},mission,far);
    auto const final=ShapingPotentials(c,{},mission,closer);
    double const gammaSub=std::pow(c.training.gamma,1.0/c.mission.actionSubsteps);
    for(auto channel : {Reward::Localizer,Reward::Glide})
    {
        auto const i=static_cast<std::size_t>(channel);
        Check(Near(Channel(improvement,channel),gammaSub*final[i]-initial[i],1e-10)
            && Near(Channel(regression,channel),gammaSub*initial[i]-final[i],1e-10)
            && Near(Channel(improvement,channel)+gammaSub*Channel(regression,channel),
                (gammaSub*gammaSub-1)*initial[i],1e-10),
            "discounted potential telescopes on out-and-back without farming");
    }
    auto const right=At(c,-6000,1000,700),left=At(c,-6000,-1000,700);
    auto const ri=ApproachStep(c,WithCourse(c,right,0),WithCourse(c,right,-5));
    auto const ro=ApproachStep(c,WithCourse(c,right,0),WithCourse(c,right,5));
    auto const li=ApproachStep(c,WithCourse(c,left,0),WithCourse(c,left,5));
    auto const lo=ApproachStep(c,WithCourse(c,left,0),WithCourse(c,left,-5));
    Check(Channel(ri,Reward::Localizer)>Channel(ro,Reward::Localizer)
        && Near(Channel(ri,Reward::Localizer),Channel(li,Reward::Localizer),1e-10)
        && Near(Channel(ro,Reward::Localizer),Channel(lo,Reward::Localizer),1e-10),
        "inward vs outward route rewards remain symmetric");
    auto glide=At(c,-6000,0,c.runway.thresholdHeightMeters
        +6000*std::tan(c.mission.glideDegrees*DegreesToRadians));
    auto down=WithCourse(c,glide,0),up=WithCourse(c,glide,0);
    down.velocity.up=-glide.airspeed*std::tan(c.mission.glideDegrees*DegreesToRadians);
    up.velocity.up=glide.airspeed*std::tan(c.mission.glideDegrees*DegreesToRadians);
    Check(Channel(ApproachStep(c,glide,down),Reward::Glide)
        >Channel(ApproachStep(c,glide,up),Reward::Glide),
        "Glide potential includes physical descent slope");
    auto locOnly=c;locOnly.training.curriculumStage=CurriculumStage::Loc;
    Check(Channel(ApproachStep(locOnly,far,closer),Reward::Glide)==0
        && Channel(ApproachStep(locOnly,far,closer),Reward::Localizer)!=0,
        "LOC disables glide potential");
    MissionState acquisition{};acquisition.phase=Phase::Approach;
    auto improvedSpeed=closer;improvedSpeed.airspeed+=1;
    Check(Channel(AdvanceMission(c,{},acquisition,far,improvedSpeed),Reward::Speed)!=0,
        "speed potential active before LOC capture");
    auto const outside=At(c,-c.mission.ilsRangeMeters-1,0,700);
    auto const inside=At(c,-c.mission.ilsRangeMeters+1,0,700);
    auto const edge=ApproachStep(c,outside,inside);
    Check(Near(Channel(edge,Reward::Glide),
        gammaSub*ShapingPotentials(c,{},mission,inside)[static_cast<std::size_t>(Reward::Glide)],1e-10),
        "entering the Glide envelope preserves a well-defined potential");
    auto half=c;half.mission.actionSubsteps*=2;
    double const gammaHalf=std::sqrt(gammaSub);
    Check(Near(Channel(ApproachStep(half,far,middle),Reward::Localizer)
        +gammaHalf*Channel(ApproachStep(half,middle,closer),Reward::Localizer),
        Channel(improvement,Reward::Localizer),1e-10),
        "substep subdivision preserves the gamma-discounted shaping");
}
void ProgressAndAmplitude(Config const& c)
{
    auto aligned=WithCourse(c,At(c,-6000,0,c.runway.thresholdHeightMeters
        +6000*std::tan(c.mission.glideDegrees*DegreesToRadians)),0);
    MissionState state{};state.phase=Phase::Approach;state.capturedLocalizer=true;
    double const gammaSub=std::pow(c.training.gamma,1.0/c.mission.actionSubsteps);
    auto const potentials=ShapingPotentials(c,{},state,aligned);
    for(auto channel:{Reward::Localizer,Reward::Glide,Reward::Speed,Reward::Slope})
    {
        auto const i=static_cast<std::size_t>(channel);
        Check(Near(Channel(ApproachStep(c,aligned,aligned),channel),
                (gammaSub-1)*potentials[i],1e-10),
            "stationary holding has the exact discounted baseline, not a positive recurrent reward");
    }
    auto prior=WithCourse(c,At(c,-6000,1000,600),0);
    auto receding=WithCourse(c,At(c,-6010,1000,600),0);
    auto const r=ApproachStep(c,prior,receding);
    auto const previous=ShapingPotentials(c,{},state,prior);
    auto const next=ShapingPotentials(c,{},state,receding);
    for(auto channel:{Reward::Localizer,Reward::Glide})
    {
        auto const i=static_cast<std::size_t>(channel);
        Check(Near(Channel(r,channel),gammaSub*next[i]-previous[i],1e-10),
            "receding must use full potential, not asymmetric gain clipping");
    }
    Check(c.reward.weights[7]==100 && c.reward.weights[8]==100 && c.reward.weights[9]==-100,
        "terminal event weights unchanged");
}
void SpeedAndSlopeShaping(Config const& c)
{
    auto bad=WithCourse(c,At(c,-6000,0,600),0);
    bad.airspeed=c.mission.approachSpeed-2*c.reward.speedTolerance;
    auto good=bad,wrong=bad;
    good.airspeed+=1;good.velocity.up=-bad.airspeed*0.10;
    wrong.airspeed-=1;wrong.velocity.up=bad.airspeed*0.10;
    for(auto channel:{Reward::Speed,Reward::Slope})
        Check(Channel(ApproachStep(c,bad,good),channel)
                >Channel(ApproachStep(c,bad,bad),channel)
            && Channel(ApproachStep(c,bad,wrong),channel)
                <Channel(ApproachStep(c,bad,bad),channel),
            "speed and slope potentials favor improved physical quality");
}
void LocalizerLessonHasNoVerticalObjective(Config c)
{
    c.mission.patrolEnabled=false;
    auto before=At(c,-6000,500,400);
    before.airspeed=c.mission.approachSpeed-5;
    before=WithCourse(c,before,0);
    auto after=before;
    after.position=FromRunway(c.runway,-6000,470.0,400);
    after.airspeed+=1;
    after=WithCourse(c,after,0);
    for(auto const stage : {CurriculumStage::Loc,CurriculumStage::Glide,
        CurriculumStage::Joint,CurriculumStage::JointMobile})
    {
        c.training.curriculumStage=stage;
        for(double const slopeDegrees : {-c.mission.glideDegrees,c.mission.glideDegrees})
        {
            auto changed=after;
            changed.velocity.up=std::hypot(changed.velocity.north,changed.velocity.east)
                *std::tan(slopeDegrees*DegreesToRadians);
            auto const result=ApproachStep(c,before,changed);
            if(stage==CurriculumStage::Loc)
                Check(Channel(result,Reward::Slope)==0 && Channel(result,Reward::Glide)==0,
                    "LOC never rewards or penalizes a vertical slope change");
            else
            {
                double const tolerance=c.reward.slopeToleranceDegrees;
                double const gammaSub=std::pow(c.training.gamma,1.0/c.mission.actionSubsteps);
                double const expected=c.reward.weights[static_cast<std::size_t>(Reward::Slope)]
                    *(gammaSub*tolerance/(tolerance+std::abs(slopeDegrees+c.mission.glideDegrees))
                        -tolerance/(tolerance+c.mission.glideDegrees));
                Check(Near(Channel(result,Reward::Slope),expected,1e-12),
                    "Glide and landing lessons preserve their signed slope shaping");
            }
            Check(Channel(result,Reward::Localizer)>0 && Channel(result,Reward::Speed)>0,
                "lateral interception and speed shaping remain active in every lesson");
        }
    }
    c.training.curriculumStage=CurriculumStage::Loc;
    after.structuralContact=true;
    auto const crash=ApproachStep(c,before,after);
    Check(crash.terminated && Channel(crash,Reward::Crash)
        ==c.reward.weights[static_cast<std::size_t>(Reward::Crash)],
        "LOC keeps physical crash safety without a vertical shaping objective");
}
void SecondaryGuidance(Config const& c)
{
    auto const plan=MakeFlightPlan(c);
    Check(ClosestPatrolWaypoint(plan,plan.points[0])==1 && ClosestPatrolWaypoint(plan,plan.points[1])==2,
        "rally from a waypoint targets the following segment endpoint");
    Check(ClosestPatrolWaypoint({}, {})==0,"empty flightplan recovery is bounded");
    MissionState rally{};
    AircraftState rallyAircraft{};
    rallyAircraft.position=plan.points[0];
    auto const rallyGuidance=Guide(c,plan,rally,rallyAircraft);
    Check(rallyGuidance.mode==GuidanceMode::Patrol && rallyGuidance.pathDistance==0
        && std::hypot(rallyGuidance.target.north-plan.points[0].north,
            rallyGuidance.target.east-plan.points[0].east)>0,
        "rally guidance projects onto the racetrack and anticipates forward like Blue patrol");
    rally.pathQualified=true;rally.waypoint=1;
    auto const followGuidance=Guide(c,plan,rally,rallyAircraft);
    Check(Near(followGuidance.target.north,plan.points[rally.waypoint].north)
        && Near(followGuidance.target.east,plan.points[rally.waypoint].east),
        "confirmed FOLLOW uses the fixed active waypoint like Blue patrol");
    MissionState acquiring{};
    rallyAircraft.airspeed=c.mission.patrolSpeed;
    AdvanceMission(c,plan,acquiring,rallyAircraft,rallyAircraft);
    Check(acquiring.rallyCorridorLatched && !acquiring.pathQualified,
        "one in-corridor sample only arms RALLY acquisition");
    AdvanceMission(c,plan,acquiring,rallyAircraft,rallyAircraft);
    auto const acquiredGuidance=Guide(c,plan,acquiring,rallyAircraft);
    Check(acquiring.pathQualified && acquiring.waypoint==1
        && Near(acquiredGuidance.target.north,plan.points[1].north)
        && Near(acquiredGuidance.target.east,plan.points[1].east),
        "confirmed RALLY acquisition exposes the next fixed point rather than the point under the aircraft");
    MissionState patrol{};
    auto const midpoint=Scale(Add(plan.points[0],plan.points[1]),0.5);
    auto before=At(c,0,0,600);before.position=Add(midpoint,Scale(RunwayRight(c.runway),-1500));
    auto after=before;after.position=Add(before.position,Scale(RunwayRight(c.runway),0.2));
    auto const toward=AdvanceMission(c,plan,patrol,before,after);
    auto const away=AdvanceMission(c,plan,patrol,after,before);
    Check(Channel(toward,Reward::Path)>Channel(away,Reward::Path)
        && Channel(away,Reward::Path)<0 && !patrol.pathQualified && patrol.completedLaps==0,
        "Blue-style patrol cost prefers recovery and penalizes retreat without qualifying a lap");

    MissionState missed{};missed.waypoint=1;missed.pathQualified=true;missed.waypointsPassed=5;
    auto const segment=Subtract(plan.points[1],plan.points[0]);
    auto const tangent=Scale(segment,1/std::sqrt(Dot(segment,segment)));
    before.position=Add(plan.points[1],Add(Scale(tangent,201),Scale(RunwayRight(c.runway),201)));
    after.position=Add(before.position,Scale(tangent,1));
    AdvanceMission(c,plan,missed,before,after);
    Check(missed.waypoint!=1 && !missed.pathQualified && missed.waypointsPassed==0 && missed.completedLaps==0,
        "a missed gate advances guidance through reacquisition without crediting a shortcut");

    auto const outside=At(c,-4000,5000,600),intercepting=At(c,-4000,4500,550);
    auto const fallback=ApproachStep(c,outside,intercepting);
    Check(Channel(fallback,Reward::Path)==0 && Channel(fallback,Reward::Localizer)>0
        && Channel(fallback,Reward::Glide)>0,
        "approach uses only dense LOC/glide shaping without a competing path reward");
    MissionState approach{};approach.phase=Phase::Approach;
    auto const afterThreshold=At(c,1000,5000,600);
    auto const acquisition=Guide(c,plan,approach,outside);
    auto const overflight=Guide(c,plan,approach,afterThreshold);
    Check(!acquisition.targetValid && !overflight.targetValid
        && acquisition.target.north==outside.position.north
        && overflight.target.north==afterThreshold.position.north,
        "approach has no racetrack, intercept or runway point before or after overflight");
}
void MovingFrame(Config c)
{
    auto const previous=c.runway;
    c.runway.origin={100,200,20};c.runway.headingDegrees+=12;c.runway.pitch=0.01;c.runway.roll=-0.01;
    auto const before=At(c,-6000,2000,700);
    auto aircraftBefore=before;aircraftBefore.position=FromRunway(previous,-6000,2000,700);
    MissionState mission{};mission.phase=Phase::Approach;
    auto const reward=AdvanceMission(c,MakeFlightPlan(c),mission,aircraftBefore,before,&previous);
    MissionState prior{};prior.phase=Phase::Approach;
    auto const beforePotential=ShapingPotentials(c,{},prior,aircraftBefore,&previous);
    double const gammaSub=std::pow(c.training.gamma,1.0/c.mission.actionSubsteps);
    for(auto channel:{Reward::Localizer,Reward::Glide})
        Check(Near(Channel(reward,channel),(gammaSub-1)*beforePotential[static_cast<std::size_t>(channel)],1e-10),
            "rigid runway motion alone yields only the discounted potential baseline");
    c.reward.localizerCourseBlend=-0.01;
    Rejects([&] {Validate(c);},"negative LOC course blend rejected");
    c.reward.localizerCourseBlend=1.01;
    Rejects([&] {Validate(c);},"LOC course blend above one rejected");
    c.reward.localizerCourseBlend=0.5;
    c.reward.localizerInterceptGain=0.99;
    Rejects([&] {Validate(c);},"LOC intercept gain below one rejected");
    c.reward.localizerInterceptGain=4.01;
    Rejects([&] {Validate(c);},"excessive LOC intercept gain rejected");
    c.reward.localizerInterceptGain=2;
    c.reward.localizerInterceptLimitDegrees=c.initial.bearingDegrees.maximum
        +std::max(std::abs(c.initial.headingOffsetDegrees.minimum),std::abs(c.initial.headingOffsetDegrees.maximum));
    Rejects([&] {Validate(c);},"LOC intercept limit must exceed every initial course");
    c.reward.localizerInterceptLimitDegrees=90;
    Rejects([&] {Validate(c);},"LOC intercept limit below ninety degrees required");
}
void StabilizedCaptureBonuses(Config c)
{
    auto aligned=WithCourse(c,At(c,-5000,0,c.runway.thresholdHeightMeters
        +5000*std::tan(c.mission.glideDegrees*DegreesToRadians)),0);
    aligned.velocity.up=-c.mission.approachSpeed*std::tan(c.mission.glideDegrees*DegreesToRadians);
    auto crossing=WithCourse(c,aligned,20);
    auto const loc=static_cast<std::size_t>(Reward::EnterLocalizer);
    auto const glide=static_cast<std::size_t>(Reward::EnterGlide);
    for(auto const stage : {CurriculumStage::Loc,CurriculumStage::Glide})
    {
        c.training.curriculumStage=stage;
        MissionState mission{};mission.phase=Phase::Approach;
        for(int step=0;step<120;++step)
        {
            auto const result=AdvanceMission(c,{},mission,crossing,crossing);
            Check(mission.outcome==Outcome::Flying && result.rewards[loc]==0 && result.rewards[glide]==0,
                "geometric beam crossings do not earn capture-lesson success bonuses");
        }
        Check(mission.capturedLocalizer && mission.capturedGlide,"transient pass really acquired the geometric beams");
        for(int step=0;step<50;++step) AdvanceMission(c,{},mission,aligned,aligned);
        auto unstable=aligned;unstable.rollRate=2*c.training.gateRateDegreesPerSecond*DegreesToRadians;
        auto const interrupted=AdvanceMission(c,{},mission,aligned,unstable);
        Check(mission.captureGateSeconds==0 && interrupted.rewards[loc]==0 && interrupted.rewards[glide]==0,
            "an interrupted stabilization hold earns no success bonus");
        double localizerBonus{},glideBonus{};
        for(int step=0;step<2000 && mission.outcome==Outcome::Flying;++step)
        {
            auto const result=AdvanceMission(c,{},mission,aligned,aligned);
            localizerBonus+=result.rewards[loc];glideBonus+=result.rewards[glide];
            if(mission.outcome==Outcome::Flying)
                Check(result.rewards[loc]==0 && result.rewards[glide]==0,"partial stabilization remains unrewarded");
        }
        Check(mission.outcome==Outcome::CaptureCompleted && Near(localizerBonus,c.reward.weights[loc])
            && Near(glideBonus,stage==CurriculumStage::Glide ? c.reward.weights[glide] : 0),
            "stabilized completion pays the existing lesson bonuses exactly once");
        Check(AdvanceMission(c,{},mission,aligned,aligned).reward==0,"terminal capture bonus never repeats");
        MissionState timeout{};timeout.phase=Phase::Approach;
        timeout.capturedLocalizer=timeout.capturedGlide=true;
        timeout.captureGateSeconds=c.training.gateHoldSeconds/2;
        timeout.steps=static_cast<std::uint64_t>(std::ceil(c.mission.maxSeconds/c.mission.physicsDt));
        auto const expired=AdvanceMission(c,{},timeout,aligned,aligned);
        Check(expired.truncated && expired.rewards[loc]==0 && expired.rewards[glide]==0,
            "timeout cannot collect an unfinished stabilized success bonus");
        MissionState failure{};failure.phase=Phase::Approach;
        failure.capturedLocalizer=failure.capturedGlide=true;
        failure.captureGateSeconds=c.training.gateHoldSeconds-c.mission.physicsDt;
        auto crashed=aligned;crashed.structuralContact=true;
        auto const rejected=AdvanceMission(c,{},failure,aligned,crashed);
        Check(failure.outcome==Outcome::Crash && rejected.rewards[loc]==0 && rejected.rewards[glide]==0,
            "crash remains authoritative at the last stabilized-capture substep");
    }
    for(auto const stage : {CurriculumStage::Joint,CurriculumStage::JointMobile})
    {
        c.training.curriculumStage=stage;
        MissionState mission{};mission.phase=Phase::Approach;
        double localizerBonus{},glideBonus{};
        for(int step=0;step<120;++step)
        {
            auto const result=AdvanceMission(c,{},mission,crossing,crossing);
            localizerBonus+=result.rewards[loc];glideBonus+=result.rewards[glide];
        }
        Check(mission.outcome==Outcome::Flying && Near(localizerBonus,c.reward.weights[loc])
            && Near(glideBonus,c.reward.weights[glide]),"both full landing lessons retain their geometric entry events");
    }
}

void TrueSnapshotCycle(Config c)
{
    c.mission.patrolEnabled=false;
    auto aircraft=WithCourse(c,At(c,-6000,0,400),0);
    aircraft.velocity.up=-aircraft.airspeed*std::tan(c.mission.glideDegrees*DegreesToRadians);
    auto const firstDeck=c.runway;
    auto moved=c;
    moved.runway.pitch=0.02;moved.runway.roll=-0.01;
    moved.runway.angularVelocity={0.001,0.002,0.003};moved.runway.velocity={1,2,0.4};
    auto next=aircraft;next.velocity.up+=0.3;
    MissionState forward{},reverse{};forward.phase=reverse.phase=Phase::Approach;
    auto const outward=AdvanceMission(moved,{},forward,aircraft,next,&firstDeck);
    auto const inward=AdvanceMission(c,{},reverse,next,aircraft,&moved.runway);
    auto const initialPotential=ShapingPotentials(c,{},forward,aircraft,&firstDeck);
    double const gammaSub=std::pow(c.training.gamma,1.0/c.mission.actionSubsteps);
    for(auto channel:{Reward::Speed,Reward::Slope})
        Check(Near(Channel(outward,channel)+gammaSub*Channel(inward,channel),
                (gammaSub*gammaSub-1)*initialPotential[static_cast<std::size_t>(channel)],1e-10),
            "true state and runway snapshots telescope with physical discounting");
}
/** Kinematic probe for a constant-speed coordinated turn, with SI gravity. */
AircraftState CoordinatedArc(Config const& c,double bankDegrees,double seconds)
{
    double const speed=c.mission.approachSpeed;
    double const bank=bankDegrees*DegreesToRadians;
    double const rate=9.80665*std::tan(bank)/speed;
    double const course=rate*seconds;
    double const forward=std::abs(rate)>1e-12 ? speed*std::sin(course)/rate : speed*seconds;
    double const cross=std::abs(rate)>1e-12 ? speed*(1-std::cos(course))/rate : 0;
    auto aircraft=WithCourse(c,At(c,-5000+forward,cross,300),course/DegreesToRadians);
    aircraft.roll=bank;aircraft.pitch=2*DegreesToRadians;
    aircraft.heading=c.runway.headingDegrees*DegreesToRadians+course;
    aircraft.rollRate=-rate*std::sin(aircraft.pitch);
    aircraft.pitchRate=rate*std::sin(bank)*std::cos(aircraft.pitch);
    aircraft.yawRate=rate*std::cos(bank)*std::cos(aircraft.pitch);
    return aircraft;
}
Outcome ArcCaptureOutcome(Config const& c,double bankDegrees,double& successBonus)
{
    MissionState mission{};mission.phase=Phase::Approach;
    auto before=CoordinatedArc(c,bankDegrees,0);
    for(int step=1;step<=12000 && mission.outcome==Outcome::Flying;++step)
    {
        auto const after=CoordinatedArc(c,bankDegrees,step*c.mission.physicsDt);
        auto const result=AdvanceMission(c,{},mission,before,after);
        successBonus+=Channel(result,Reward::EnterLocalizer);
        before=after;
    }
    return mission.outcome;
}
void PotentialTerminalAndTimeout(Config c)
{
    c.training.curriculumStage=CurriculumStage::Loc;
    c.mission.patrolEnabled=false;
    auto before=WithCourse(c,At(c,-6000,500,350),0);
    MissionState state{};state.phase=Phase::Approach;
    auto const phi=ShapingPotentials(c,{},state,before);
    auto crashed=before;crashed.structuralContact=true;
    auto crash=AdvanceMission(c,{},state,before,crashed);
    Check(crash.terminated && state.outcome==Outcome::Crash
        && Near(Channel(crash,Reward::Localizer),-phi[static_cast<std::size_t>(Reward::Localizer)],1e-10)
        && Channel(crash,Reward::Crash)==c.reward.weights[static_cast<std::size_t>(Reward::Crash)],
        "terminal crash has zero next potential and retains its physical penalty");
    MissionState timed{};timed.phase=Phase::Approach;
    timed.steps=static_cast<std::uint64_t>(std::ceil(c.mission.maxSeconds/c.mission.physicsDt));
    auto const timeout=AdvanceMission(c,{},timed,before,before);
    double const gammaSub=std::pow(c.training.gamma,1.0/c.mission.actionSubsteps);
    Check(timeout.truncated && !timeout.terminated && timed.outcome==Outcome::Timeout
        && Near(Channel(timeout,Reward::Localizer),
            (gammaSub-1)*phi[static_cast<std::size_t>(Reward::Localizer)],1e-10),
        "truncated timeout preserves next-state potential for critic bootstrap");
}
void LocalizerTrackingRejectsBriefTurn(Config c)
{
    c.training.curriculumStage=CurriculumStage::Loc;
    c.mission.patrolEnabled=false;c.mission.maxSeconds=120;
    c.runway.velocity={};c.runway.angularVelocity={};
    for(double const heading:{0.0,90.0,270.0})
    {
        c.runway.headingDegrees=heading;
        for(double const bank:{-7.0,7.0})
        {
            auto legacy=c;legacy.training.gateHoldSeconds=2;
            double legacyBonus{},trackingBonus{};
            Check(ArcCaptureOutcome(legacy,bank,legacyBonus)==Outcome::CaptureCompleted
                && Near(legacyBonus,c.reward.weights[static_cast<std::size_t>(Reward::EnterLocalizer)]),
                "a turning arc reproduces the former two-second success loophole");
            Check(ArcCaptureOutcome(c,bank,trackingBonus)==Outcome::Timeout && trackingBonus==0,
                "delivered LOC tracking cannot complete on a brief left or right arc");
        }
        double straightBonus{};
        Check(ArcCaptureOutcome(c,0,straightBonus)==Outcome::CaptureCompleted
            && Near(straightBonus,c.reward.weights[static_cast<std::size_t>(Reward::EnterLocalizer)]),
            "straight forward LOC tracking completes without imposing a descent slope");
    }
}
} // namespace
int main()
{
    try
    {
        auto config=LoadConfig(PPO_FLIGHT_CONFIG);
        PotentialTerminalAndTimeout(config);
        LocalizerTrackingRejectsBriefTurn(config);
        LocalizerLessonHasNoVerticalObjective(config);
        StabilizedCaptureBonuses(config);
        TrueSnapshotCycle(config);
        config.mission.patrolEnabled=true; // Exercise the preserved optional patrol path reward.
        Convergence(config);ProgressAndAmplitude(config);SpeedAndSlopeShaping(config);SecondaryGuidance(config);MovingFrame(config);
    }
    catch(std::exception const& error) {std::cerr<<error.what()<<'\n';return 1;}
    std::cout<<"Flight guidance reward regressions passed\n";
}
