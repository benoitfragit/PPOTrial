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
StepResult ApproachStep(Config const& c,AircraftState const& before,AircraftState const& after)
{
    MissionState mission{};mission.phase=Phase::Approach;
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
    Check(Channel(improvement,Reward::EnterLocalizer)==0 && Channel(improvement,Reward::EnterGlide)==0,
        "far convergence does not claim a capture");

    MissionState patrol{};
    auto const plan=MakeFlightPlan(c);
    auto const patrolImprovement=AdvanceMission(c,plan,patrol,far,closer);
    Check(patrol.phase==Phase::Patrol && Channel(patrolImprovement,Reward::Localizer)>0
        && Channel(patrolImprovement,Reward::Glide)>0
        && Channel(patrolImprovement,Reward::EnterLocalizer)==0
        && Channel(patrolImprovement,Reward::EnterGlide)==0,
        "visible raw ILS shapes LOC and glide before capture while patrol remains active");
    auto const outside=At(c,-4000,5000,700),outsideCloser=At(c,-4000,4999.8,699.98);
    MissionState outsidePatrol{};
    auto const denseShaping=AdvanceMission(c,plan,outsidePatrol,outside,outsideCloser);
    Check(!Guide(c,plan,outsidePatrol,outsideCloser).ilsValid
        && Channel(denseShaping,Reward::Localizer)>0 && Channel(denseShaping,Reward::Glide)>0,
        "dense shaping guides acquisition outside the physical signal sector");
    auto const beyondRange=At(c,-16000,0,1000),nearerBeyondRange=At(c,-15999.8,0,1000);
    MissionState beyondRangePatrol{};
    auto const noGeometry=AdvanceMission(c,plan,beyondRangePatrol,beyondRange,nearerBeyondRange);
    Check(Channel(noGeometry,Reward::Localizer)==0 && Channel(noGeometry,Reward::Glide)==0,
        "ILS shaping remains bounded to the configured front-side range");

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
        Check(Near(Channel(correct,channel),weight*c.mission.physicsDt,1e-10),"on-beam progress is time scaled");
        Check(Channel(wrong,channel)<0,"following the beams away from the runway is penalized");
        Check(weight>=100*c.reward.weights[static_cast<std::size_t>(Reward::Path)],"ILS strongly outweighs the secondary path");
    }
    // Late approaches should still receive guidance, with a decayed magnitude.
    MissionState late{};late.phase=Phase::Approach;
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
    MissionState lateReversal{};lateReversal.phase=Phase::Approach;
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
    double const ilsWeights=c.reward.weights[1]+c.reward.weights[2];
    double const maximumIls=ilsWeights*(c.reward.ilsDecaySeconds+c.mission.physicsDt+c.reward.convergenceSeconds);
    double const maximumSecondary=(c.reward.weights[3]+c.reward.weights[4])*c.mission.maxSeconds;
    double const captures=c.reward.weights[5]+c.reward.weights[6];
    Check(maximumIls+maximumSecondary+captures<c.reward.weights[7],
        "continuous ILS shaping plus captures remains below touchdown bonus");
    Check(!c.initial.approach && c.reward.weights[7]==100 && c.reward.weights[8]==100 && c.reward.weights[9]==-100,
        "default mission keeps the racetrack as fallback and preserves outcome amplitudes");
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
    Check(Channel(fallback,Reward::Path)>0 && Channel(fallback,Reward::Localizer)>0
        && Channel(fallback,Reward::Glide)>0,
        "outside the radio sector, dense ILS shaping complements the geometric intercept path");
    MissionState approach{};approach.phase=Phase::Approach;
    auto const target=DeckCoordinates(c.runway,Guide(c,plan,approach,outside).target);
    auto const afterThreshold=At(c,1000,5000,600);
    auto const overflightTarget=DeckCoordinates(c.runway,Guide(c,plan,approach,afterThreshold).target);
    Check(Near(target.north,-c.mission.switchDistanceMeters) && Near(target.east,0)
        && target.up>c.runway.thresholdHeightMeters
        && Near(overflightTarget.north,target.north) && Near(overflightTarget.east,target.east)
        && Near(overflightTarget.up,target.up),
        "uncaptured approach target stays at the fixed LOC/glide entry before and after overflight");
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
}
}
int main()
{
    try
    {
        auto const config=LoadConfig(PPO_FLIGHT_CONFIG);
        Convergence(config);ProgressAndAmplitude(config);SecondaryGuidance(config);MovingFrame(config);
    }
    catch(std::exception const& error) {std::cerr<<error.what()<<'\n';return 1;}
    std::cout<<"Flight guidance reward regressions passed\n";
}
