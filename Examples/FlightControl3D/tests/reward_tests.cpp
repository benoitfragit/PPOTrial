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
    auto far=WithCourse(c,At(c,-6000,2000,700),0);
    auto closer=WithCourse(c,At(c,-6000,1999.8,699.8),0);
    auto const improvement=ApproachStep(c,far,closer);
    auto const regression=ApproachStep(c,closer,far);
    for(auto channel : {Reward::Localizer,Reward::Glide})
        Check(Channel(improvement,channel)>0 && Channel(regression,channel)<0
            && Near(Channel(improvement,channel)+Channel(regression,channel),0,1e-12)
            && Channel(ApproachStep(c,far,far),channel)==0,
            "signed delta Q gives positive/negative/zero without recurrence");
    auto const right=At(c,-6000,1000,700),left=At(c,-6000,-1000,700);
    auto const ri=ApproachStep(c,WithCourse(c,right,0),WithCourse(c,right,-5));
    auto const ro=ApproachStep(c,WithCourse(c,right,0),WithCourse(c,right,5));
    auto const li=ApproachStep(c,WithCourse(c,left,0),WithCourse(c,left,5));
    auto const lo=ApproachStep(c,WithCourse(c,left,0),WithCourse(c,left,-5));
    Check(Channel(ri,Reward::Localizer)>0 && Channel(ro,Reward::Localizer)<0
        && Near(Channel(ri,Reward::Localizer),Channel(li,Reward::Localizer),1e-10)
        && Near(Channel(ro,Reward::Localizer),Channel(lo,Reward::Localizer),1e-10),
        "inward vs outward route rewards are symmetric");
    auto glide=At(c,-6000,0,c.runway.thresholdHeightMeters
        +6000*std::tan(c.mission.glideDegrees*DegreesToRadians));
    auto down=WithCourse(c,glide,0),up=WithCourse(c,glide,0);
    down.velocity.up=-glide.airspeed*std::tan(c.mission.glideDegrees*DegreesToRadians);
    up.velocity.up=glide.airspeed*std::tan(c.mission.glideDegrees*DegreesToRadians);
    Check(Channel(ApproachStep(c,glide,down),Reward::Glide)>0
        && Channel(ApproachStep(c,glide,up),Reward::Glide)<0,
        "glide includes physical descent slope, not just location");
    auto locOnly=c;locOnly.training.curriculumStage=CurriculumStage::Loc;
    Check(Channel(ApproachStep(locOnly,far,closer),Reward::Glide)==0
        && Channel(ApproachStep(locOnly,far,closer),Reward::Localizer)>0,
        "LOC gate disables glide shaping only");
    MissionState acquisition{};acquisition.phase=Phase::Approach;
    auto improvedSpeed=closer;
    improvedSpeed.airspeed+=1;
    auto const ungated=AdvanceMission(c,MakeFlightPlan(c),acquisition,far,improvedSpeed);
    Check(Channel(ungated,Reward::Speed)!=0,
        "speed shaping is active before LOC capture");
    auto const outside=At(c,-c.mission.ilsRangeMeters-1,0,700);
    auto const inside=At(c,-c.mission.ilsRangeMeters+1,0,700);
    Check(Channel(ApproachStep(c,outside,inside),Reward::Glide)==0,
        "glide envelope transition does not fabricate delta Q");
    auto half=c;half.mission.physicsDt/=2;
    auto const middle=WithCourse(c,At(c,-6000,1999.9,699.9),0);
    Check(Near(Channel(ApproachStep(half,far,middle),Reward::Localizer)
        +Channel(ApproachStep(half,middle,closer),Reward::Localizer),
        Channel(improvement,Reward::Localizer),1e-12),
        "substep subdivision preserves additive quality difference");
}
void ProgressAndAmplitude(Config const& c)
{
    auto aligned=WithCourse(c,At(c,-6000,0,c.runway.thresholdHeightMeters
        +6000*std::tan(c.mission.glideDegrees*DegreesToRadians)),0);
    Check(Channel(ApproachStep(c,aligned,aligned),Reward::Localizer)==0
        && Channel(ApproachStep(c,aligned,aligned),Reward::Glide)==0,
        "perfect stationary beams cannot accumulate time reward");
    auto const prior=WithCourse(c,At(c,-6000,1000,600),0);
    auto const receding=WithCourse(c,At(c,-6010,1000,600),0);
    auto const r=ApproachStep(c,prior,receding);
    Check(Channel(r,Reward::Localizer)<=0 && Channel(r,Reward::Glide)<=0,
        "receding cannot farm angular quality");
    Check(c.reward.weights[7]==100 && c.reward.weights[8]==100 && c.reward.weights[9]==-100,
        "terminal outcomes unchanged");
}
void SpeedAndSlopeShaping(Config const& c)
{
    auto bad=WithCourse(c,At(c,-6000,0,600),0);
    bad.airspeed=c.mission.approachSpeed-2*c.reward.speedTolerance;
    auto good=bad,wrong=bad;
    good.airspeed+=1;good.velocity.up=-bad.airspeed*0.10;
    wrong.airspeed-=1;wrong.velocity.up=bad.airspeed*0.10;
    for(auto channel:{Reward::Speed,Reward::Slope})
        Check(Channel(ApproachStep(c,bad,good),channel)>0
            && Channel(ApproachStep(c,bad,wrong),channel)<0
            && Channel(ApproachStep(c,bad,bad),channel)==0,
            "speed and slope improve/recede with no constant baseline");
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

    auto const outside=At(c,-4000,5000,600),intercepting=At(c,-4000,4999.8,599.8);
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
    Check(Near(Channel(reward,Reward::Localizer),0,1e-12) && Near(Channel(reward,Reward::Glide),0,1e-12),
        "rigid runway motion alone does not create progress or convergence");
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
    for(auto channel:{Reward::Speed,Reward::Slope})
        Check(Near(Channel(outward,channel)+Channel(inward,channel),0,1e-11),
            "true state and runway snapshots telescope across a closed moving-deck cycle");
}
}
int main()
{
    try
    {
        auto config=LoadConfig(PPO_FLIGHT_CONFIG);
        TrueSnapshotCycle(config);
        config.mission.patrolEnabled=true; // Exercise the preserved optional patrol path reward.
        Convergence(config);ProgressAndAmplitude(config);SpeedAndSlopeShaping(config);SecondaryGuidance(config);MovingFrame(config);
    }
    catch(std::exception const& error) {std::cerr<<error.what()<<'\n';return 1;}
    std::cout<<"Flight guidance reward regressions passed\n";
}
