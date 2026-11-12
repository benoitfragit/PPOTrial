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
    auto const far=At(c,-6000,2000,700);
    auto const closer=At(c,-6000,1999.8,699.98);
    MissionState mission{};mission.phase=Phase::Approach;
    auto const geometry=Guide(c,MakeFlightPlan(c),mission,far);
    Check(geometry.ilsValid && std::abs(geometry.localizer)>c.reward.localizerToleranceDegrees*DegreesToRadians
        && std::abs(geometry.glideError)>c.reward.glideToleranceDegrees*DegreesToRadians,
        "fixture starts outside both old zero-reward tolerances");
    auto const improvement=ApproachStep(c,far,closer),regression=ApproachStep(c,closer,far);
    auto const stationary=ApproachStep(c,far,far);
    for(auto const channel : {Reward::Localizer,Reward::Glide})
    {
        Check(Channel(improvement,channel)>0 && Channel(regression,channel)<0,
            "closing onto either beam earns shaping outside the old tolerances; moving away loses it");
        Check(Near(Channel(improvement,channel)+Channel(regression,channel),0,1e-12),
            "a closed error excursion cannot farm convergence");
        Check(Channel(stationary,channel)==0,"stationary geometry earns no repeated ILS shaping");
    }
    auto unitLocalizerScale=c;
    unitLocalizerScale.reward.localizerConvergenceScale=1;
    auto const unitScaleImprovement=ApproachStep(unitLocalizerScale,far,closer);
    Check(Near(Channel(improvement,Reward::Localizer),
            c.reward.localizerConvergenceScale*Channel(unitScaleImprovement,Reward::Localizer),1e-12)
        && Near(Channel(improvement,Reward::Glide),Channel(unitScaleImprovement,Reward::Glide),1e-12),
        "LOC corrective potential is strengthened without scaling glide");
    Check(Channel(improvement,Reward::EnterLocalizer)==0 && Channel(improvement,Reward::EnterGlide)==0,
        "far convergence does not claim a capture");

    auto geometricOnly=c;
    geometricOnly.reward.localizerCourseBlend=0;
    auto atAngle=[&](double degrees)
    { return At(geometricOnly,-6000,6000*std::tan(degrees*DegreesToRadians),700); };
    auto const nearCorrection=ApproachStep(geometricOnly,atAngle(10),atAngle(9.9));
    auto const farCorrection=ApproachStep(geometricOnly,atAngle(25),atAngle(24.9));
    Check(Channel(nearCorrection,Reward::Localizer)>0
        && Near(Channel(nearCorrection,Reward::Localizer),Channel(farCorrection,Reward::Localizer),1e-10),
        "linear LOC geometry keeps an equal corrective gradient across the front-side sector");

    auto const rightState=At(c,-6000,1000,700);
    auto const rightNeutral=WithCourse(c,rightState,0);
    auto const rightInward=ApproachStep(c,rightNeutral,WithCourse(c,rightState,-5));
    auto const rightOutward=ApproachStep(c,rightNeutral,WithCourse(c,rightState,5));
    auto const leftState=At(c,-6000,-1000,700);
    auto const leftNeutral=WithCourse(c,leftState,0);
    auto const leftInward=ApproachStep(c,leftNeutral,WithCourse(c,leftState,5));
    auto const leftOutward=ApproachStep(c,leftNeutral,WithCourse(c,leftState,-5));
    Check(Channel(rightInward,Reward::Localizer)>0 && Channel(rightOutward,Reward::Localizer)<0
        && Channel(leftInward,Reward::Localizer)>0 && Channel(leftOutward,Reward::Localizer)<0
        && Near(Channel(rightInward,Reward::Localizer),Channel(leftInward,Reward::Localizer),1e-10)
        && Near(Channel(rightOutward,Reward::Localizer),Channel(leftOutward,Reward::Localizer),1e-10),
        "mirrored LOC states reward the corresponding inward course and penalize the outward course");

    MissionState observing{};
    auto const plan=MakeFlightPlan(c);
    auto observationAtAngle=[&](double degrees)
    {
        double const radians=degrees*DegreesToRadians;
        return Observe(c,plan,observing,At(c,-6000,6000*std::tan(radians),700))[22];
    };
    double const right45=observationAtAngle(45),right60=observationAtAngle(60);
    double const left45=observationAtAngle(-45),left60=observationAtAngle(-60);
    Check(Near(right45,std::sin(45*DegreesToRadians))
        && Near(right60,std::sin(60*DegreesToRadians))
        && Near(left45,-right45) && Near(left60,-right60)
        && std::abs(right60)>std::abs(right45) && std::abs(right60)<1,
        "signed LOC observation preserves side and angular magnitude beyond the radio sector");

    MissionState patrol{};
    auto const patrolImprovement=AdvanceMission(c,plan,patrol,far,closer);
    Check(patrol.phase==Phase::Patrol && Channel(patrolImprovement,Reward::Localizer)>0
        && Channel(patrolImprovement,Reward::Glide)==0
        && Channel(patrolImprovement,Reward::EnterLocalizer)==0
        && Channel(patrolImprovement,Reward::EnterGlide)==0,
        "dense LOC guides acquisition while glide stays silent before LOC capture");
    auto const outside=At(c,-4000,5000,700),outsideCloser=At(c,-4000,4999.8,699.98);
    MissionState outsidePatrol{};
    auto const denseShaping=AdvanceMission(c,plan,outsidePatrol,outside,outsideCloser);
    Check(!Guide(c,plan,outsidePatrol,outsideCloser).ilsValid
        && Channel(denseShaping,Reward::Localizer)>0 && Channel(denseShaping,Reward::Glide)==0,
        "dense LOC guides patrol acquisition outside the signal sector");
    auto const outsideObservation=Observe(c,plan,outsidePatrol,outsideCloser);
    Check(outsideObservation[22]!=0 && outsideObservation[23]!=0 && outsideObservation[24]==0,
        "out-of-sector geometric LOC/glide errors remain observable with ILS valid false");
    MissionState directApproach{};directApproach.phase=Phase::Approach;
    auto const approachShaping=AdvanceMission(c,plan,directApproach,outside,outsideCloser);
    Check(!directApproach.capturedLocalizer && Channel(approachShaping,Reward::Glide)>0,
        "approach learns glide correction before radio LOC capture");
    auto const beyondRange=At(c,-16000,1000,1000),nearerBeyondRange=At(c,-15999.8,999.8,1000);
    MissionState beyondRangePatrol{};
    auto const farGeometry=AdvanceMission(c,plan,beyondRangePatrol,beyondRange,nearerBeyondRange);
    auto const farObservation=Observe(c,plan,beyondRangePatrol,nearerBeyondRange);
    Check(!Guide(c,plan,beyondRangePatrol,nearerBeyondRange).ilsValid
        && Channel(farGeometry,Reward::Localizer)>0 && Channel(farGeometry,Reward::Glide)==0
        && Channel(farGeometry,Reward::Speed)==0 && Channel(farGeometry,Reward::Slope)==0
        && farObservation[22]!=0 && farObservation[23]==0,
        "far LOC remains visible while glide, speed and slope stay silent");
    auto const outsideBoundary=At(c,-c.mission.ilsRangeMeters-1,0,100);
    auto const insideBoundary=At(c,-c.mission.ilsRangeMeters+1,0,100);
    auto const envelopeEntry=ApproachStep(c,outsideBoundary,insideBoundary);
    double const maximumEntrySustain=c.reward.weights[static_cast<std::size_t>(Reward::Localizer)]
        *c.mission.physicsDt*c.reward.localizerSustainScale;
    Check(Channel(envelopeEntry,Reward::Localizer)>0
        && Channel(envelopeEntry,Reward::Localizer)<=maximumEntrySustain+1e-12,
        "crossing the radio-range boundary neither gates LOC nor creates a corrective spike");

    // Slow improvement followed by one fast regression must also cancel.
    auto previous=far;double localizer{},glide{};
    for(int step=1;step<=100;++step)
    {
        auto const next=At(c,-6000,2000-0.2*step,700-0.02*step);
        auto const reward=ApproachStep(c,previous,next);
        localizer+=Channel(reward,Reward::Localizer);glide+=Channel(reward,Reward::Glide);previous=next;
    }
    auto const returnStep=ApproachStep(c,previous,far);
    Check(Near(localizer+Channel(returnStep,Reward::Localizer),0,1e-12)
        && Near(glide+Channel(returnStep,Reward::Glide),0,1e-12),"convergence is not clipped asymmetrically");

    auto refined=c;refined.mission.physicsDt=c.mission.physicsDt/2;
    auto const middle=At(c,-6000,1999.9,699.99);
    auto const first=ApproachStep(refined,far,middle),second=ApproachStep(refined,middle,closer);
    Check(Near(Channel(first,Reward::Localizer)+Channel(second,Reward::Localizer),Channel(improvement,Reward::Localizer),1e-12)
        && Near(Channel(first,Reward::Glide)+Channel(second,Reward::Glide),Channel(improvement,Reward::Glide),1e-12),
        "the convergence amount is independent of physics subdivision");
}
void ProgressAndAmplitude(Config const& c)
{
    double const along=-6000,advance=c.mission.approachSpeed*c.mission.physicsDt;
    double const slope=std::tan(c.mission.glideDegrees*DegreesToRadians);
    auto const before=At(c,along,0,c.runway.thresholdHeightMeters-along*slope);
    auto const forward=At(c,along+advance,0,c.runway.thresholdHeightMeters-(along+advance)*slope);
    auto const backward=At(c,along-advance,0,c.runway.thresholdHeightMeters-(along-advance)*slope);
    auto const correct=ApproachStep(c,before,forward),wrong=ApproachStep(c,before,backward);
    for(auto const channel : {Reward::Localizer,Reward::Glide})
    {
        auto const weight=c.reward.weights[static_cast<std::size_t>(channel)];
        double const sustainScale=channel==Reward::Localizer ? c.reward.localizerSustainScale : 1.0;
        Check(Near(Channel(correct,channel),weight*c.mission.physicsDt*sustainScale,1e-10),
            "on-beam progress uses the channel sustain scale");
        Check(Channel(wrong,channel)<0,"following the beams away from the runway is penalized");
        Check(weight>=100*c.reward.weights[static_cast<std::size_t>(Reward::Path)],"ILS strongly outweighs the secondary path");
    }
    // Late approaches should still receive guidance, with a decayed magnitude.
    MissionState late{};late.phase=Phase::Approach;late.capturedLocalizer=true;
    late.steps=static_cast<std::uint64_t>(std::round(c.reward.ilsDecaySeconds/c.mission.physicsDt));
    auto const delayed=AdvanceMission(c,MakeFlightPlan(c),late,before,forward);
    double const decay=std::exp(-static_cast<double>(late.steps-1)*c.mission.physicsDt/c.reward.ilsDecaySeconds);
    for(auto const channel : {Reward::Localizer,Reward::Glide})
        Check(Near(Channel(delayed,channel),Channel(correct,channel)*decay,1e-10),
            "ILS reward follows configured elapsed-time decay");
    // A closed ILS-error excursion must not earn convergence just because
    // the return occurs after the progress reward has decayed.
    auto const offBeam=At(c,-6000,2000,700),nearBeam=At(c,-6000,1000,500);
    auto const earlyImprovement=ApproachStep(c,offBeam,nearBeam);
    MissionState lateReversal{};lateReversal.phase=Phase::Approach;lateReversal.capturedLocalizer=true;
    lateReversal.steps=static_cast<std::uint64_t>(std::round(c.reward.ilsDecaySeconds/c.mission.physicsDt));
    auto const reversed=AdvanceMission(c,MakeFlightPlan(c),lateReversal,nearBeam,offBeam);
    for(auto const channel : {Reward::Localizer,Reward::Glide})
    {
        Check(Channel(earlyImprovement,channel)>0 && Channel(reversed,channel)<0,
            "delayed reversal still penalizes LOC and glide regression");
        Check(Near(Channel(earlyImprovement,channel)+Channel(reversed,channel),0,1e-12),
            "time decay cannot create reward from a closed ILS error excursion");
    }
    // Angular errors can improve simply because range increases near the threshold.
    auto const offset=At(c,-100,30,40),receding=At(c,-100-advance,30,40);
    auto const retreat=ApproachStep(c,offset,receding);
    Check(Channel(retreat,Reward::Localizer)<=0 && Channel(retreat,Reward::Glide)<=0,
        "artificial angular improvement while retreating cannot earn reward");

    // With continuous in-sector ILS, only progress decays: signed convergence
    // telescopes exactly, including delayed oscillations.
    double const decayingSeconds=c.reward.ilsDecaySeconds+c.mission.physicsDt;
    double const maximumLocalizer=c.reward.weights[1]*(
        c.reward.localizerSustainScale*decayingSeconds
        +c.reward.localizerConvergenceScale*c.reward.convergenceSeconds);
    double const maximumGlide=c.reward.weights[2]*(decayingSeconds+c.reward.convergenceSeconds);
    double const maximumSecondary=(c.reward.weights[3]+c.reward.weights[4])*
        (decayingSeconds+c.reward.convergenceSeconds);
    double const captures=c.reward.weights[5]+c.reward.weights[6];
    Check(maximumLocalizer+maximumGlide+maximumSecondary+captures<c.reward.weights[7],
        "continuous ILS shaping plus captures remains below touchdown bonus");
    double const localizerWeight=c.reward.weights[static_cast<std::size_t>(Reward::Localizer)];
    double const speedWeight=c.reward.weights[static_cast<std::size_t>(Reward::Speed)];
    double const slopeWeight=c.reward.weights[static_cast<std::size_t>(Reward::Slope)];
    Check(Near(localizerWeight,0.30) && Near(speedWeight,0.01)
        && Near(slopeWeight,0.01) && Near(localizerWeight,30*speedWeight)
        && speedWeight>c.reward.weights[static_cast<std::size_t>(Reward::Path)],
        "LOC dominates speed and slope while secondary flight guidance remains useful");
    Check(c.initial.approach && c.reward.weights[7]==100 && c.reward.weights[8]==100 && c.reward.weights[9]==-100
        && Near(c.reward.localizerCourseBlend,0.5) && Near(c.reward.localizerInterceptGain,2)
        && Near(c.reward.localizerInterceptLimitDegrees,60) && c.training.hiddenWidth==128,
        "default mission uses intercept-aware LOC, the larger network and preserved outcome amplitudes");
}
void SpeedAndSlopeShaping(Config const& c)
{
    auto unstable=At(c,-6000,0,600);
    unstable.airspeed=c.mission.approachSpeed-2*c.reward.speedTolerance;
    unstable.velocity=Add(Scale(RunwayForward(c.runway),unstable.airspeed),
        Scale(RunwayNormal(c.runway),-0.30*unstable.airspeed));
    auto improved=unstable;improved.airspeed+=1;
    improved.velocity=Add(Scale(RunwayForward(c.runway),improved.airspeed),
        Scale(RunwayNormal(c.runway),-0.25*improved.airspeed));
    auto worsened=unstable;worsened.airspeed-=1;
    worsened.velocity=Add(Scale(RunwayForward(c.runway),worsened.airspeed),
        Scale(RunwayNormal(c.runway),-0.35*worsened.airspeed));
    auto const steady=ApproachStep(c,unstable,unstable);
    auto const better=ApproachStep(c,unstable,improved);
    auto const worse=ApproachStep(c,unstable,worsened);
    for(auto channel : {Reward::Speed,Reward::Slope})
        Check(Channel(better,channel)>Channel(steady,channel)
            && Channel(steady,channel)>Channel(worse,channel)
            && Channel(steady,channel)>0,
            "speed/slope reward stays dense outside tolerance and discriminates correction from regression");

    MissionState uncaptured{};
    uncaptured.phase=Phase::Approach;
    auto incompatible=improved;
    incompatible.heading+=Pi;
    auto const beforeCapture=AdvanceMission(c,MakeFlightPlan(c),uncaptured,unstable,incompatible);
    Check(!uncaptured.capturedLocalizer
        && Channel(beforeCapture,Reward::Speed)==0 && Channel(beforeCapture,Reward::Slope)==0,
        "speed and slope stay silent before confirmed LOC capture");

    MissionState lost{};
    lost.phase=Phase::Approach;
    lost.capturedLocalizer=true;
    lost.localizerHoldSeconds=c.mission.captureHoldSeconds;
    lost.localizerLossSeconds=c.mission.captureHoldSeconds-c.mission.physicsDt/2;
    auto const afterLoss=AdvanceMission(c,MakeFlightPlan(c),lost,unstable,incompatible);
    Check(!lost.capturedLocalizer
        && Channel(afterLoss,Reward::Speed)==0 && Channel(afterLoss,Reward::Slope)==0,
        "confirmed LOC loss immediately disables speed and slope reward");
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
    c.reward.convergenceSeconds=-1;Rejects([&] {Validate(c);},"negative convergence scale rejected");
    c.reward.convergenceSeconds=std::numeric_limits<double>::infinity();
    Rejects([&] {Validate(c);},"non-finite convergence scale rejected");
    c.reward.convergenceSeconds=10;
    c.reward.ilsDecaySeconds=0;Rejects([&] {Validate(c);},"zero ILS decay rejected");
    c.reward.ilsDecaySeconds=std::numeric_limits<double>::infinity();
    Rejects([&] {Validate(c);},"non-finite ILS decay rejected");
    c.reward.ilsDecaySeconds=140;
    c.reward.localizerConvergenceScale=0;
    Rejects([&] {Validate(c);},"zero LOC corrective scale rejected");
    c.reward.localizerConvergenceScale=std::numeric_limits<double>::infinity();
    Rejects([&] {Validate(c);},"non-finite LOC corrective scale rejected");
    c.reward.localizerConvergenceScale=5;
    c.reward.localizerSustainScale=-0.01;
    Rejects([&] {Validate(c);},"negative LOC sustain scale rejected");
    c.reward.localizerSustainScale=1.01;
    Rejects([&] {Validate(c);},"LOC sustain scale above one rejected");
    c.reward.localizerSustainScale=0.25;
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
    c.reward.localizerInterceptLimitDegrees=c.initial.bearingDegrees.maximum+3;
    Rejects([&] {Validate(c);},"LOC intercept limit must exceed every initial course");
    c.reward.localizerInterceptLimitDegrees=90;
    Rejects([&] {Validate(c);},"LOC intercept limit below ninety degrees required");
}
}
int main()
{
    try
    {
        auto const config=LoadConfig(PPO_FLIGHT_CONFIG);
        Convergence(config);ProgressAndAmplitude(config);SpeedAndSlopeShaping(config);SecondaryGuidance(config);MovingFrame(config);
    }
    catch(std::exception const& error) {std::cerr<<error.what()<<'\n';return 1;}
    std::cout<<"Flight guidance reward regressions passed\n";
}
