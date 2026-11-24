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
#include "flight_aircraft_performance.hpp"
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

/** @brief Former piecewise T-6 flap lift: 0.02 per degree to 10, 0.01 to 20, then 0.005 (reference copy). */
double FormerT6FlapLift(double degrees)
{
    if(degrees<=10) return 0.02*degrees;
    if(degrees<=20) return 0.2+0.01*(degrees-10);
    return 0.3+0.005*(degrees-20);
}

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
    Check(StallCalibratedSpeed(T6Performance,0.33)>38 && StallCalibratedSpeed(T6Performance,0.33)<39,
        "the pinned mass, wing area and flap lift imply a 38-39 m/s stall CAS");
    // The T-6 flap table reproduces the former piecewise law: 0.02, 0.01 then 0.005 per degree.
    for(double degrees : {0.0,4.0,10.0,15.0,20.0,27.0,30.0})
        Check(Near(FlapLiftIncrement(T6Performance,degrees),FormerT6FlapLift(degrees),1e-12),
            "the T-6 flap lift table is unchanged");
    Check(Near(FlapLiftIncrement(Pc7Performance,10),0.092,1e-12) && StallCalibratedSpeed(Pc7Performance,0.33)>31
        && StallCalibratedSpeed(Pc7Performance,0.33)<34,"the PC-7 lift implies a 31-34 m/s stall CAS with approach flaps");
    Check(FindAircraftPerformance("c172p")==nullptr && FindAircraftPerformance("pc7")==&SupportedAircraft[1],
        "only the T-6 and the PC-7 are supported");
    auto const pc7=LoadConfig(std::filesystem::path(PPO_FLIGHT_PC7_CONFIG));
    Check(pc7.aircraft=="pc7" && pc7.performance.name=="pc7" && pc7.checkpointSignature!=c.checkpointSignature,
        "the PC-7 scenario is valid and has its own checkpoint contract");

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
    for(auto const* field : {"throttle","elevatorTrim","approachFlap","rolloutFlap","propeller","mixture"})
    {
        changed=signature;changed["initial"][field]=changed["initial"][field].get<double>()-0.05;
        Check(!CompatibleTrainingSignature(changed.dump(),c.checkpointSignature),
            "fixed controls and aircraft configuration are part of the contract");
    }
    changed=signature;changed["initial"]["unknownControl"]=1.0;
    Check(!CompatibleTrainingSignature(changed.dump(),c.checkpointSignature),
        "an unlisted initial field is incompatible by default");
    changed=signature;changed["initial"]["rolloutStart"]["speedMps"]={10,20};changed["initial"]["rollDegrees"]=3.0;
    Check(CompatibleTrainingSignature(changed.dump(),c.checkpointSignature),
        "rollout and attitude spawn ranges keep a compatible warm start");
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
        Check(Guide(c,approach,aircraft).targetSpeed>=1.3*StallCalibratedSpeed(T6Performance,flap),
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

/** @brief Short final points at the aim point; near it the track converges onto the centreline. */
void ApproachDirection()
{
    auto c=T6Config();
    c.runway.origin={150,-200,25};c.runway.headingDegrees=123;
    double const aimAlong=AimPointAlongMeters(c);
    for(double cross : {-300.0,0.0,250.0}) for(double along : {-9000.0,-3000.0,-1000.0})
    {
        auto const aircraft=Approach(c,along,cross);
        auto const guidance=Guide(c,Flying(),aircraft);
        if(aimAlong-along<=InterceptLookaheadMeters)
        {
            auto const toAim=UnitDirection(Subtract(AimPoint(c),aircraft.position));
            Check(guidance.targetValid && Near(Dot(guidance.targetDirection,toAim),1,1e-9)
                && Near(Length(Subtract(guidance.target,AimPoint(c))),0,1e-6),
                "on short final the commanded direction points straight at the aim point");
        }
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

/** @brief Angle between the commanded horizontal track and the runway axis, positive to the right, in degrees. */
double CommandTrackDegrees(Config const& c,Guidance const& guidance)
{
    auto const direction=RunwayAxes(c.runway,guidance.targetDirection);
    return std::atan2(direction.east,direction.north)/DegreesToRadians;
}

/** @brief Far out the command intercepts the extended centreline at a bounded angle and keeps the glide plane. */
void CentrelineIntercept()
{
    auto c=T6Config();
    c.runway.origin={150,-200,25};c.runway.headingDegrees=123;
    double const aimAlong=AimPointAlongMeters(c);
    double const along=-9000;
    auto const onAxis=Guide(c,Flying(),Approach(c,along));
    Check(Near(CommandTrackDegrees(c,onAxis),0,1e-9)
        && Near(Length(Subtract(onAxis.target,FromRunway(c.runway,along+InterceptLookaheadMeters,0,0))),0,1e-6),
        "on the extended centreline the command follows the runway axis toward a point 2 km ahead");
    for(double cross : {-2900.0,-250.0,-5.0,5.0,250.0,2900.0})
    {
        auto const aircraft=Approach(c,along,cross);
        auto const guidance=Guide(c,Flying(),aircraft);
        double const expected=std::max(-MaximumInterceptAngleDegrees,std::min(MaximumInterceptAngleDegrees,
            std::atan2(-cross,InterceptLookaheadMeters)/DegreesToRadians));
        Check(Near(CommandTrackDegrees(c,guidance),expected,1e-9) && CommandTrackDegrees(c,guidance)*cross<0,
            "off the centreline the command turns toward it, at most at the intercept angle");
        Check(Near(CommandTrackDegrees(c,Guide(c,Flying(),Approach(c,along,-cross))),-expected,1e-9),
            "left and right offsets are intercepted symmetrically");
        double const height=DeckCoordinates(c.runway,aircraft.position).up;
        Check(Near(guidance.targetSlope,-std::atan(height/std::hypot(aimAlong-along,cross)),1e-9),
            "the commanded slope still descends toward the aim point on the glide plane");
    }
    // The look-ahead switches from 2 km to the aim point without a jump in the command.
    double const switchAlong=aimAlong-InterceptLookaheadMeters;
    auto const before=Guide(c,Flying(),Approach(c,switchAlong-0.01,300));
    auto const after=Guide(c,Flying(),Approach(c,switchAlong+0.01,300));
    Check(Near(Dot(before.targetDirection,after.targetDirection),1,1e-9),
        "the direction is continuous where the look-ahead starts shrinking toward the aim point");
}

/** @brief Following the command from the widest curriculum start reaches the threshold aligned with the runway. */
void CentrelineInterceptTrajectory()
{
    auto c=T6Config();
    c.runway.origin={0,0,15};c.runway.headingDegrees=0;
    constexpr double Speed=56;
    constexpr double Dt=0.1;
    double const distance=c.initial.distanceMeters.maximum;
    double const bearing=c.initial.bearingDegrees.maximum*DegreesToRadians;
    for(double const side : {-1.0,1.0})
    {
        double const startCross=side*distance*std::sin(bearing);
        auto aircraft=Approach(c,-distance*std::cos(bearing),startCross);
        double previousCross=std::abs(startCross);
        bool reachedThreshold=false;
        for(int step=0;step<10000;++step)
        {
            auto const guidance=Guide(c,Flying(),aircraft);
            auto const local=DeckCoordinates(c.runway,aircraft.position);
            Check(std::abs(local.east)<=previousCross+1e-9 && local.east*startCross>=0,
                "the lateral offset decreases toward the centreline without overshooting it");
            previousCross=std::abs(local.east);
            if(local.north>=0)
            {
                Check(std::abs(local.east)<5 && std::abs(CommandTrackDegrees(c,guidance))<1,
                    "the aircraft crosses the threshold on the centreline, aligned with the runway");
                reachedThreshold=true;break;
            }
            aircraft.velocity=Scale(guidance.targetDirection,Speed);
            aircraft.position=Add(aircraft.position,Scale(aircraft.velocity,Dt));
        }
        Check(reachedThreshold,"the intercept reaches the threshold in finite time");
    }
}

/**
 * @brief Downwind, base and final gates of a pattern at a given height, as in a curriculum stage.
 *
 * The final gate lies on the extended centreline 7 km out, below the glide plane, so the
 * approach captures the plane from below with the lateral offset already removed.
 */
Route PatternRoute(double side,double height)
{
    Route route{};
    route.start={500,side*2500,height};
    route.gates[0]={-3000,side*2500,height};
    route.gates[1]={-6000,side*1500,height};
    route.gates[2]={-7000,0,height};
    route.count=3;
    return route;
}

/** @brief A gate is passed within its radius, or beyond its plane inside the corridor. */
void GatePassage()
{
    Route route{};
    route.start={0,0,300};
    route.gates[0]={-3000,0,300};
    route.gates[1]={-3000,2000,300};
    route.count=2;
    Check(GatePassed(route,0,-2800,150) && !GatePassed(route,0,-2500,0),
        "the first gate is passed within its radius, not before it");
    Check(GatePassed(route,0,-3100,GateCorridorMeters-1) && !GatePassed(route,0,-3100,GateCorridorMeters+1),
        "beyond the plane of the inbound leg, only the corridor passes the gate");
    Check(!GatePassed(route,0,-2900,GateCorridorMeters-1),"before the plane only the radius counts");
    Check(GatePassed(route,1,-3000+GateCorridorMeters-1,2001) && !GatePassed(route,1,-3000,1500),
        "a later gate uses the leg from the previous gate");
}

/** @brief The command flies the route gate by gate, then the unchanged approach. */
void RouteGuidance()
{
    auto c=T6Config();
    c.runway.origin={150,-200,25};c.runway.headingDegrees=123;
    auto mission=Flying();
    mission.route=PatternRoute(1,300);
    auto aircraft=Approach(c,0,2500);aircraft.position=FromRunway(c.runway,0,2500,250);
    // Flying downwind toward the gate, so the command is not limited by the turn lead.
    aircraft.velocity=Scale(Scale(RunwayForward(c.runway),-1),c.mission.approachSpeed);
    auto const guidance=Guide(c,mission,aircraft);
    auto const direction=RunwayAxes(c.runway,guidance.targetDirection);
    Check(direction.north<0 && Near(direction.east,0,1e-9) && direction.up>0
        && Near(Length(Subtract(guidance.target,FromRunway(c.runway,-3000,2500,300))),0,1e-6),
        "on downwind the command flies toward the first gate and climbs to its height");
    Check(Near(guidance.targetSlope,std::atan(50.0/3000.0),1e-9),"the gate slope reaches the gate height");
    mission.nextGate=mission.route.count;
    auto direct=Flying();
    auto const after=Guide(c,mission,aircraft);
    auto const reference=Guide(c,direct,aircraft);
    Check(after.targetDirection.north==reference.targetDirection.north && after.targetDirection.east==reference.targetDirection.east
        && after.targetDirection.up==reference.targetDirection.up && after.targetSlope==reference.targetSlope,
        "once the route is complete the command is the direct approach, bit for bit");
    mission.nextGate=0;
    Check(Guide(c,Flying(Phase::Rollout),aircraft).targetSlope==0,"the route never applies on the ground");
}

/** @brief Regression: route turn and downwind commands are invariant under runway rotation. */
void RouteGuidanceUsesRunwayAxes()
{
    auto c = T6Config();
    c.runway.origin = {150, -200, 25};
    double const lead = MaximumGateLeadDegrees * DegreesToRadians;
    for (double const heading : {0.0, 90.0, 123.0, 270.0})
    {
        c.runway.headingDegrees = heading;
        for (double const side : {-1.0, 1.0})
        {
            auto mission = Flying();
            mission.route.start = {-11000.0, 0.0, 600.0};
            mission.route.gates[0] = {-11000.0, side * 1100.0, 600.0};
            mission.route.count = 1;

            auto aircraft = Approach(c, -11000.0);
            aircraft.position = FromRunway(c.runway, -11000.0, 0.0, 600.0);
            aircraft.velocity = Scale(RunwayForward(c.runway), c.mission.approachSpeed);
            auto const turn = RunwayAxes(c.runway, Guide(c, mission, aircraft).targetDirection);
            Check(Near(turn.north, std::cos(lead), 1e-9)
                && Near(turn.east, side * std::sin(lead), 1e-9)
                && Near(turn.up, 0.0, 1e-9),
                "a route turn leads the runway-relative track by 10 degrees on the gate side");

            mission.route.gates[0] = {-12000.0, 0.0, 600.0};
            aircraft.velocity = Scale(RunwayForward(c.runway), -c.mission.approachSpeed);
            auto const downwind = RunwayAxes(c.runway, Guide(c, mission, aircraft).targetDirection);
            Check(Near(downwind.north, -1.0, 1e-9)
                && Near(downwind.east, 0.0, 1e-9)
                && Near(downwind.up, 0.0, 1e-9),
                "a downwind route follows the reverse runway axis at every runway heading");
        }
    }
}

/** @brief Gates advance in the mission; touching down before the last gate is a crash. */
void RouteMission()
{
    auto c=T6Config();
    c.runway.origin={0,0,15};c.runway.headingDegrees=0;
    auto mission=Flying();
    mission.route=PatternRoute(-1,300);
    auto before=Approach(c,-2900,-2500);before.position=FromRunway(c.runway,-2900,-2500,300);
    auto after=before;after.position=FromRunway(c.runway,-3001,-2500,300);
    AdvanceMission(c,mission,before,after);
    Check(mission.nextGate==1 && mission.outcome==Outcome::Flying,"passing a gate advances the route");
    auto landing=Flying();
    landing.route=PatternRoute(-1,300);
    auto wheels=Approach(c,600,0);wheels.position=FromRunway(c.runway,600,0,1.5);wheels.weightOnWheels=true;
    CrashDiagnostic diagnostic{};
    auto const step=AdvanceMission(c,landing,wheels,wheels,&diagnostic);
    Check(step.terminated && landing.outcome==Outcome::Crash && !landing.touchedDown
        && diagnostic.conditions[static_cast<std::size_t>(CrashReason::RouteIncomplete)].failed,
        "landing with gates remaining is a shortcut and a crash");
}

/**
 * @brief Largest heading error of a 30-degree bank follower in the route turns, in degrees.
 *
 * Measured: the command leads the track by at most MaximumGateLeadDegrees in every turn of
 * the patterns and of the holding lap.
 */
constexpr double MaximumRouteTurnErrorDegrees = 40.0;

/**
 * @brief Flies the command at approach speed with a turn rate bounded by a 30-degree bank,
 *        passing gates as AdvanceMission does.
 * @param maximumErrorDegrees Largest heading error between the track and the command once the
 *     first gate is passed.
 * @return True when the aircraft crosses the threshold aligned after the last gate.
 */
bool FlyRouteAtBank(Config const& c,MissionState mission,AircraftState aircraft,double& maximumErrorDegrees)
{
    constexpr double Speed=56;
    constexpr double Dt=0.1;
    double const turnRate=StandardGravity*std::tan(30*DegreesToRadians)/Speed;
    auto const horizontal=RunwayAxes(c.runway,aircraft.velocity);
    double track=std::atan2(horizontal.east,horizontal.north);
    maximumErrorDegrees=0;
    for(int step=0;step<30000;++step)
    {
        auto const local=DeckCoordinates(c.runway,aircraft.position);
        while(mission.nextGate<mission.route.count && GatePassed(mission.route,mission.nextGate,local.north,local.east))
            ++mission.nextGate;
        auto const guidance=Guide(c,mission,aircraft);
        double const command=CommandTrackDegrees(c,guidance)*DegreesToRadians;
        double const error=WrapAngle(command-track);
        if(mission.nextGate>0 && mission.nextGate<mission.route.count)
            maximumErrorDegrees=std::max(maximumErrorDegrees,std::abs(error)/DegreesToRadians);
        if(local.north>=0 && mission.nextGate==mission.route.count && std::abs(local.east)<200)
            return std::abs(local.east)<10 && std::abs(command/DegreesToRadians)<2;
        track+=std::clamp(error,-turnRate*Dt,turnRate*Dt);
        double const climb=Speed*std::sin(guidance.targetSlope);
        double const level=Speed*std::cos(guidance.targetSlope);
        aircraft.velocity=Add(Add(Scale(RunwayForward(c.runway),level*std::cos(track)),
            Scale(RunwayRight(c.runway),level*std::sin(track))),Scale(RunwayNormal(c.runway),climb));
        aircraft.position=Add(aircraft.position,Scale(aircraft.velocity,Dt));
    }
    return false;
}

/** @brief Holding laps and patterns flown at a 30-degree bank keep a small heading error in the turns. */
void RouteTurnsAtBank()
{
    auto c=T6Config();
    c.runway.origin={0,0,15};c.runway.headingDegrees=0;
    std::mt19937 rng(91U);
    for(auto const& stage : c.training.curriculum.stages)
    {
        if(stage.kind==StageKind::Approach) continue;
        for(auto const side : {ApproachSide::Left,ApproachSide::Right})
        {
            auto const start=SampleStageStart(c,c.runway,stage,side,rng);
            auto mission=Flying();
            mission.route=start.route;
            double maximumError=0;
            Check(FlyRouteAtBank(c,mission,start.aircraft,maximumError),
                "a route flown at a 30-degree bank reaches the threshold aligned");
            std::cout<<"[ROUTE_BANK] "<<stage.name<<" max_error_deg="<<maximumError<<"\n";
            Check(maximumError<=MaximumRouteTurnErrorDegrees,
                "a 30-degree bank follows the route turns with a small heading error");
        }
    }
}

/** @brief A moving deck translates along its heading and heaves sinusoidally, with a matching velocity. */
void MovingDeckPose()
{
    auto c=T6Config();
    c.runway.origin={100,-50,15};c.runway.headingDegrees=30;
    auto const still=MovedDeck(c.runway,DeckMotionSettings{},1.0,12.0);
    Check(still.origin.north==c.runway.origin.north && still.origin.east==c.runway.origin.east
        && still.origin.up==c.runway.origin.up && Length(still.velocity)==0,"a static deck never moves");
    DeckMotionSettings const motion{6.0,0.6,9.0};
    double const phase=0.7;
    for(double const seconds : {0.0,1.3,4.5,20.0})
    {
        auto const deck=MovedDeck(c.runway,motion,phase,seconds);
        auto const offset=DeckCoordinates(c.runway,deck.origin);
        double const angle=2*Pi*seconds/9.0+phase;
        Check(Near(offset.north,6.0*seconds,1e-9) && Near(offset.east,0,1e-9)
            && Near(offset.up,0.6*std::sin(angle),1e-9),"the deck advances along its heading and heaves");
        auto const velocity=RunwayAxes(c.runway,deck.velocity);
        Check(Near(velocity.north,6.0,1e-9) && Near(velocity.east,0,1e-9)
            && Near(velocity.up,0.6*2*Pi/9.0*std::cos(angle),1e-9),"the deck velocity is the pose derivative");
        // A finite difference of the pose agrees with the reported velocity.
        constexpr double Step=1e-4;
        auto const next=MovedDeck(c.runway,motion,phase,seconds+Step);
        auto const numeric=Scale(Subtract(next.origin,deck.origin),1/Step);
        Check(Length(Subtract(numeric,deck.velocity))<1e-3,"the pose and the velocity are consistent");
    }
}

/** @brief On a moving deck speeds, touchdown sink, ground roll and stop are measured relative to the deck. */
void MovingDeckMission()
{
    auto c=T6Config();
    c.runway.landingMarginMeters=0;
    c.runway.origin={0,0,15};c.runway.headingDegrees=0;
    c.runway.velocity={6,0,0.4};
    auto aircraft=Approach(c,-1000);
    auto const guidance=Guide(c,Flying(),aircraft);
    auto const world=RunwayAxes(c.runway,aircraft.velocity);
    Check(Near(guidance.relativeVelocity.north,world.north-6,1e-9) && Near(guidance.relativeVelocity.up,world.up-0.4,1e-9),
        "guidance velocities are deck-relative");
    // A rising deck meets a descending aircraft harder: the sink adds the heave rate.
    auto before=Approach(c,300);before.position=FromRunway(c.runway,300,0,1.8);before.velocity.up=-1;
    auto after=before;after.weightOnWheels=true;after.velocity.up=0;
    auto landing=Flying();
    AdvanceMission(c,landing,before,after);
    Check(landing.touchedDown && Near(landing.touchdownSinkMps,1.4,1e-9),"the touchdown sink is relative to the heaving deck");
    // An aircraft carried by the deck rolls nowhere and stops, although it moves at 6 m/s.
    c.runway.velocity={6,0,0};
    auto mission=Flying(Phase::Rollout);mission.touchedDown=true;mission.contactAttempted=true;
    auto carried=Approach(c,600);carried.weightOnWheels=true;carried.velocity=c.runway.velocity;
    auto const origin=c.runway.origin;
    std::size_t const holdSteps=static_cast<std::size_t>(c.mission.stoppedHoldSeconds/c.mission.physicsDt)+2;
    for(std::size_t step=1;step<=holdSteps && mission.outcome==Outcome::Flying;++step)
    {
        auto previous=carried;previous.position=FromRunway(c.runway,600,0,1.8);
        c.runway.origin=Add(origin,Scale(c.runway.velocity,c.mission.physicsDt*static_cast<double>(step)));
        carried.position=FromRunway(c.runway,600,0,1.8);
        AdvanceMission(c,mission,previous,carried);
    }
    Check(mission.outcome==Outcome::Stopped && Near(mission.stopFromThreshold,600,1e-6) && mission.groundRoll<1e-6,
        "an aircraft at rest on the moving deck adds no ground roll and stops");
}

/** @brief The explicit previous potentials replace Phi(s) in the shaping, and match it on a static deck. */
void PreviousPotentials()
{
    auto const c=T6Config();
    auto const before=Approach(c,-3000);
    auto after=before;after.position=Add(before.position,Scale(before.velocity,c.mission.physicsDt));
    auto first=Flying();auto second=Flying();auto third=Flying();
    auto const implicit=AdvanceMission(c,first,before,after);
    auto const explicitStep=AdvanceMission(c,second,before,after,ShapingPotentials(c,second,before));
    std::array<double,RewardCount> const zero{};
    auto const fromZero=AdvanceMission(c,third,before,after,zero);
    for(std::size_t channel=0;channel<RewardCount;++channel)
        Check(implicit.rewards[channel]==explicitStep.rewards[channel],"explicit potentials of s give the same rewards");
    auto const phi=ShapingPotentials(c,Flying(),before);
    double const path=fromZero.rewards[static_cast<std::size_t>(Reward::Path)]-implicit.rewards[static_cast<std::size_t>(Reward::Path)];
    Check(Near(path,phi[static_cast<std::size_t>(Reward::Path)],1e-12),"the shaping subtracts the given Phi(s)");
}

/** @brief Stages may move the deck; the motion is validated and stays outside the checkpoint contract. */
void DeckMotionStages()
{
    auto const c=T6Config();
    auto const& stages=c.training.curriculum.stages;
    Check(stages.size()>=6 && stages[stages.size()-2].name=="swell" && stages.back().name=="moving-deck",
        "the curriculum ends with the swell and moving-deck stages");
    Check(!stages[stages.size()-2].deckMotion.speedMps && stages[stages.size()-2].deckMotion.swellAmplitudeMeters>0
        && stages.back().deckMotion.speedMps>0 && stages.back().deckMotion.swellAmplitudeMeters>0,
        "the swell stage heaves a stationary deck, the last stage also moves it forward");
    for(std::size_t stage=0;stage+2<stages.size();++stage)
        Check(!stages[stage].deckMotion.Moving(),"the earlier stages keep a static deck");
    std::ifstream source(PPO_FLIGHT_CONFIG);nlohmann::json json;source>>json;
    // The deck motion lives in the training curriculum, outside the checkpoint signature.
    auto withoutMotion=json;
    for(auto& stage : withoutMotion["training"]["curriculum"]["stages"]) stage.erase("deckMotion");
    TemporaryDirectory directory("flight-deck-");
    auto const path=directory.path/"scenario.json";
    {std::ofstream output(path);output<<withoutMotion;}
    auto const staticDeck=LoadConfig(path);
    Check(staticDeck.checkpointSignature==c.checkpointSignature && !staticDeck.training.curriculum.stages.back().deckMotion.Moving(),
        "a stage without deckMotion is static and the motion does not change the checkpoint contract");
    auto const reject=[&](char const* key,double value,char const* message)
    {
        auto invalid=json;
        invalid["training"]["curriculum"]["stages"].back()["deckMotion"][key]=value;
        {std::ofstream output(path);output<<invalid;}
        Rejects([&] { LoadConfig(path); },message);
    };
    reject("speedMps",-1.0,"a negative deck speed is rejected");
    reject("speedMps",40.0,"an implausible deck speed is rejected");
    reject("swellAmplitudeMeters",-0.1,"a negative swell is rejected");
    reject("swellAmplitudeMeters",14.5,"a swell that sinks the deck below the sea is rejected");
    reject("swellPeriodSeconds",0.5,"a swell period below the physical range is rejected");
    auto unknown=json;
    unknown["training"]["curriculum"]["stages"].back()["deckMotion"]["pitchDegrees"]=1.0;
    {std::ofstream output(path);output<<unknown;}
    Rejects([&] { LoadConfig(path); },"unknown deck motion fields are rejected");
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
/** @brief Places the three nominal wheel points, without native contact, at a height above the deck. */
void PlaceWheels(Config const& c,AircraftState& aircraft,double height)
{
    aircraft.wheels={WheelContact{FromRunway(c.runway,502,0,height),false},
        WheelContact{FromRunway(c.runway,499,-1.5,height),false},WheelContact{FromRunway(c.runway,499,1.5,height),false}};
}

/** @brief With the gear not locked down, the airframe reaching the deck is a crash contact. */
void BellyContact()
{
    auto c=T6Config();
    c.runway.origin={0,0,15};c.runway.headingDegrees=0;
    auto aircraft=Approach(c,500);
    aircraft.position=FromRunway(c.runway,500,0,1.5);
    aircraft.wheelDataValid=true;
    PlaceWheels(c,aircraft,0.4);
    aircraft.gearPosition=0;
    Check(!ContactDetected(c,aircraft),"a retracted gear above the deck is not a contact");
    PlaceWheels(c,aircraft,-0.01);
    Check(ContactDetected(c,aircraft),"the airframe reaching the deck with the gear retracted is a contact");
    aircraft.gearPosition=0.5*c.mission.gearDownMinimum;
    Check(ContactDetected(c,aircraft),"a gear still in transit cannot carry the aircraft");
    auto mission=Flying();
    CrashDiagnostic diagnostic{};
    auto const step=AdvanceMission(c,mission,aircraft,aircraft,&diagnostic);
    Check(step.terminated && mission.outcome==Outcome::Crash
        && diagnostic.conditions[static_cast<std::size_t>(CrashReason::Gear)].failed,
        "a belly contact crashes and names the gear");
    aircraft.gearPosition=1;
    Check(!ContactDetected(c,aircraft),"with the gear locked down only the native wheel contacts count");
}

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

/** @brief The curriculum scale runs over the approach levels, then one unit step per stage. */
void CurriculumScale()
{
    auto const c=T6Config();
    auto const& curriculum=c.training.curriculum;
    Check(curriculum.stages.size()>=4 && CurriculumLevels(curriculum)==curriculum.levels+curriculum.stages.size(),
        "the scenario curriculum continues with its stages");
    double const steps=static_cast<double>(CurriculumLevels(curriculum)-1);
    double const approachSteps=static_cast<double>(curriculum.levels-1);
    for(double level=0;level<=approachSteps;++level)
    {
        auto const position=ResolveCurriculum(curriculum,level/steps);
        Check(!position.staged && Near(position.approachDifficulty,level/approachSteps,1e-12),
            "the approach levels keep their approach difficulty");
    }
    for(std::size_t stage=0;stage<curriculum.stages.size();++stage)
    {
        double const level=approachSteps+static_cast<double>(stage)+1;
        auto const atLevel=ResolveCurriculum(curriculum,level/steps);
        auto const justAbove=ResolveCurriculum(curriculum,(level-0.999)/steps);
        Check(atLevel.staged && atLevel.stage==stage && justAbove.staged && justAbove.stage==stage,
            "each stage covers one unit step of the scale, up to its own level");
    }
    Check(ResolveCurriculum(curriculum,1.0).stage==curriculum.stages.size()-1,"difficulty one is the last stage");
    auto direct=curriculum;direct.stages.clear();
    for(double difficulty : {0.0,0.37,1.0})
    {
        auto const position=ResolveCurriculum(direct,difficulty);
        Check(!position.staged && Near(position.approachDifficulty,difficulty,1e-12),
            "without stages the curriculum difficulty is the approach difficulty, as before");
    }
}

/**
 * @brief Follows the command kinematically at approach speed, passing gates as AdvanceMission does.
 * @return True when the aircraft crosses the threshold after the last gate, within 10 m of the
 *     centreline and 2 degrees of the runway heading.
 */
bool FollowRouteToThreshold(Config const& c,MissionState mission,AircraftState aircraft)
{
    constexpr double Speed=56;
    constexpr double Dt=0.1;
    for(int step=0;step<30000;++step)
    {
        auto const guidance=Guide(c,mission,aircraft);
        auto const local=DeckCoordinates(c.runway,aircraft.position);
        while(mission.nextGate<mission.route.count && GatePassed(mission.route,mission.nextGate,local.north,local.east))
            ++mission.nextGate;
        if(local.north>=0 && mission.nextGate==mission.route.count && std::abs(local.east)<200)
            return std::abs(local.east)<10 && std::abs(CommandTrackDegrees(c,Guide(c,mission,aircraft)))<2;
        aircraft.velocity=Scale(guidance.targetDirection,Speed);
        aircraft.position=Add(aircraft.position,Scale(aircraft.velocity,Dt));
    }
    return false;
}

/** @brief Stage starts respect their envelope, and their routes lead to an aligned threshold crossing. */
void StageStarts()
{
    auto c=T6Config();
    c.runway.origin={0,0,15};c.runway.headingDegrees=0;
    std::mt19937 rng(57U);
    for(auto const& stage : c.training.curriculum.stages)
    {
        for(int sample=0;sample<40;++sample)
        {
            auto const side=sample%2 ? ApproachSide::Right : ApproachSide::Left;
            double const sideSign=side==ApproachSide::Right ? 1.0 : -1.0;
            auto const start=SampleStageStart(c,c.runway,stage,side,rng);
            auto const local=DeckCoordinates(c.runway,start.aircraft.position);
            double const relativeHeading=WrapAngle(start.aircraft.heading)/DegreesToRadians;
            Check(IsFinite(start.aircraft) && start.aircraft.gearPosition==1 && start.route.count<=MaximumGates
                && std::hypot(local.north,local.east)<c.mission.maxDistanceMeters && std::abs(local.east)<5000
                && local.up>c.mission.flareHeightMeters && local.up<2000,
                "a stage start is finite, gear down, inside the envelope and the observed ranges");
            if(stage.kind==StageKind::Approach)
            {
                double const distance=std::hypot(local.north,local.east);
                double const above=local.up-ApproachPathHeight(c,local.north);
                // The runway heading is zero here: a turn against the side sign heads for the centreline.
                double const towardCentreline=-sideSign*relativeHeading;
                Check(towardCentreline>=stage.headingOffsetDegrees.minimum-1e-6
                    && towardCentreline<=stage.headingOffsetDegrees.maximum+1e-6,
                    "an approach stage heading offset is positive toward the extended centreline");
                Check(start.route.count==0 && distance>=stage.distanceMeters.minimum-0.01
                    && distance<=stage.distanceMeters.maximum+0.01 && local.east*sideSign>=-1e-9
                    && above>=stage.heightMeters.minimum-1e-6 && above<=stage.heightMeters.maximum+1e-6,
                    "an approach stage draws its distance, side and height above the glide plane");
                continue;
            }
            auto const& last=start.route.gates[start.route.count-1];
            Check(last.crossMeters==0 && local.up>=stage.heightMeters.minimum-1e-6
                && local.up<=stage.heightMeters.maximum+1e-6 && local.east*sideSign>=-1e-9,
                "a route stage starts on its side, at its height, and ends on the extended centreline");
            if(stage.kind==StageKind::Pattern)
            {
                Check(start.route.count==3 && std::abs(std::abs(relativeHeading)-180)<=std::max(-stage.headingOffsetDegrees.minimum,stage.headingOffsetDegrees.maximum)+1e-6
                    && start.route.gates[0].crossMeters*sideSign>0,"a pattern starts downwind, opposite the runway");
            }
            else
            {
                auto const& gates = start.route.gates;
                double const fix = stage.fixAlongMeters;
                double const far = fix - stage.outboundMeters;
                double const width = sideSign * stage.holdingWidthMeters;
                double const radius = stage.holdingWidthMeters / 2.0;
                Check(start.route.count == 11 && gates[0].alongMeters == fix
                    && gates[10].alongMeters == fix && gates[10].crossMeters == 0.0
                    && gates[4].alongMeters == fix && gates[4].crossMeters == width
                    && gates[5].alongMeters == far && gates[5].crossMeters == width
                    && gates[9].alongMeters == far && gates[9].crossMeters == 0.0,
                    "holding straight legs meet both semicircles at their tangent endpoints");
                for (std::size_t segment = 0; segment <= 4; ++segment)
                {
                    double const angle = Pi * static_cast<double>(segment) / 4.0;
                    auto const& first = gates[segment];
                    auto const& second = gates[5 + segment];
                    Check(Near(first.alongMeters, fix + radius * std::sin(angle), 1e-8)
                        && Near(first.crossMeters, width * (1.0 - std::cos(angle)) / 2.0, 1e-8)
                        && Near(second.alongMeters, far - radius * std::sin(angle), 1e-8)
                        && Near(second.crossMeters, width * (1.0 + std::cos(angle)) / 2.0, 1e-8)
                        && Near(first.heightMeters, local.up, 1e-8)
                        && Near(second.heightMeters, local.up, 1e-8),
                        "holding gates sample both constant-height semicircles on the selected side");
                }
            }
            if(sample<4)
            {
                auto mission=Flying();
                mission.route=start.route;
                Check(FollowRouteToThreshold(c,mission,start.aircraft),
                    "following the command along the stage route crosses the threshold aligned");
            }
        }
    }
}

/** @brief Bank angle of the feasibility follower, in degrees: a firm but ordinary approach turn. */
constexpr double FeasibilityBankDegrees = 30.0;
/**
 * @brief Largest centreline offset at the threshold of a landable start, in metres: half the
 *        lateral half-width of the landing surface, leaving room for the flare.
 *
 * Measured: the corners of the approach levels 0 to 5, which the T-6 policy lands, cross the
 * threshold within 22 m and 4.3 degrees.
 */
constexpr double LandableThresholdCrossMeters = 30.0;
/** @brief Largest track error at the threshold of a landable start, in degrees (a quarter of the touchdown limit). */
constexpr double LandableThresholdTrackDegrees = 5.0;

/** @brief Lateral state of the follower when it crosses the threshold after the last gate. */
struct ThresholdCrossing final
{
    bool reached{};
    double crossMeters{};
    double trackDegrees{};
};

/**
 * @brief Flies the command at approach speed with a turn rate bounded by FeasibilityBankDegrees,
 *        passing gates as AdvanceMission does.
 * @return Centreline offset and track error at the first threshold crossing after the last gate;
 *     not reached within the step budget otherwise.
 */
ThresholdCrossing FlyToThreshold(Config const& c,MissionState mission,AircraftState aircraft)
{
    double const speed=c.mission.approachSpeed;
    constexpr double Dt=0.1;
    constexpr int MaximumSteps=30000;
    double const turnRate=StandardGravity*std::tan(FeasibilityBankDegrees*DegreesToRadians)/speed;
    auto const horizontal=RunwayAxes(c.runway,aircraft.velocity);
    double track=std::atan2(horizontal.east,horizontal.north);
    double previousAlong=DeckCoordinates(c.runway,aircraft.position).north;
    for(int step=0;step<MaximumSteps;++step)
    {
        auto const local=DeckCoordinates(c.runway,aircraft.position);
        while(mission.nextGate<mission.route.count && GatePassed(mission.route,mission.nextGate,local.north,local.east))
            ++mission.nextGate;
        // A pattern starts abeam the runway: only a crossing from before the threshold counts.
        bool const crossed=previousAlong<0 && local.north>=0;
        if(crossed && mission.nextGate==mission.route.count)
            return {true,local.east,WrapAngle(track)/DegreesToRadians};
        previousAlong=local.north;
        auto const guidance=Guide(c,mission,aircraft);
        double const error=WrapAngle(CommandTrackDegrees(c,guidance)*DegreesToRadians-track);
        track+=std::clamp(error,-turnRate*Dt,turnRate*Dt);
        double const climb=speed*std::sin(guidance.targetSlope);
        double const level=speed*std::cos(guidance.targetSlope);
        aircraft.velocity=Add(Add(Scale(RunwayForward(c.runway),level*std::cos(track)),
            Scale(RunwayRight(c.runway),level*std::sin(track))),Scale(RunwayNormal(c.runway),climb));
        aircraft.position=Add(aircraft.position,Scale(aircraft.velocity,Dt));
    }
    return {};
}

/** @brief True when a threshold crossing leaves a normal flare and touchdown. */
bool Landable(ThresholdCrossing const& crossing) noexcept
{
    return crossing.reached && std::abs(crossing.crossMeters)<=LandableThresholdCrossMeters
        && std::abs(crossing.trackDegrees)<=LandableThresholdTrackDegrees;
}

/** @brief Range reduced to one of its bounds. */
Range Corner(Range const& range,bool upper)
{
    double const bound=upper ? range.maximum : range.minimum;
    return {bound,bound};
}

/**
 * @brief Flies one start corner, logging and counting it as a failure when it is not landable.
 * @return Centreline offset magnitude at the threshold, in metres (infinite when not reached).
 */
double MeasureCorner(Config const& c,char const* scenario,std::string const& level,
    MissionState const& mission,AircraftState const& aircraft,std::size_t& failures)
{
    auto const crossing=FlyToThreshold(c,mission,aircraft);
    if(!Landable(crossing))
    {
        ++failures;
        auto const local=DeckCoordinates(c.runway,aircraft.position);
        std::cout<<"[FEASIBILITY_FAIL] "<<scenario<<" "<<level<<" along="<<local.north<<" cross="<<local.east
            <<" heading_deg="<<WrapAngle(aircraft.heading)/DegreesToRadians<<" reached="<<crossing.reached
            <<" threshold_cross_m="<<crossing.crossMeters<<" threshold_track_deg="<<crossing.trackDegrees<<"\n";
    }
    if(!crossing.reached)
        return std::numeric_limits<double>::infinity();
    return std::abs(crossing.crossMeters);
}

/** @brief Corners of the full approach envelope at every approach level. */
void ApproachLevelCorners(Config const& base,char const* scenario,std::size_t& failures)
{
    std::mt19937 rng(73U);
    auto const& curriculum=base.training.curriculum;
    double worst=0;
    for(std::size_t level=0;level<curriculum.levels;++level)
    {
        double difficulty=0;
        if(curriculum.levels>1)
            difficulty=static_cast<double>(level)/static_cast<double>(curriculum.levels-1);
        for(int corner=0;corner<8;++corner)
        {
            auto c=base;
            bool const far=corner&1;
            c.training.curriculum.firstDistanceMeters=Corner(base.training.curriculum.firstDistanceMeters,far);
            c.initial.distanceMeters=Corner(base.initial.distanceMeters,far);
            c.initial.bearingDegrees=Corner(base.initial.bearingDegrees,corner&2);
            c.initial.headingOffsetDegrees=Corner(base.initial.headingOffsetDegrees,corner&4);
            for(auto const side : {ApproachSide::Left,ApproachSide::Right})
            {
                auto const aircraft=SampleApproachStart(c,c.runway,side,difficulty,rng);
                worst=std::max(worst,MeasureCorner(c,scenario,"approach level "+std::to_string(level),
                    Flying(),aircraft,failures));
            }
        }
    }
    std::cout<<"[FEASIBILITY] "<<scenario<<" approach levels worst_threshold_cross_m="<<worst<<"\n";
}

/** @brief Corners of one stage envelope: distance or start position, bearing or offset, heading. */
void StageCorners(Config const& c,char const* scenario,StageSettings const& stage,std::size_t& failures)
{
    std::mt19937 rng(79U);
    double worst=0;
    for(int corner=0;corner<8;++corner)
    {
        auto reduced=stage;
        reduced.distanceMeters=Corner(stage.distanceMeters,corner&1);
        reduced.startAlongMeters=Corner(stage.startAlongMeters,corner&1);
        reduced.bearingDegrees=Corner(stage.bearingDegrees,corner&2);
        reduced.lateralMeters=Corner(stage.lateralMeters,corner&2);
        reduced.headingOffsetDegrees=Corner(stage.headingOffsetDegrees,corner&4);
        reduced.heightMeters=Corner(stage.heightMeters,false);
        for(auto const side : {ApproachSide::Left,ApproachSide::Right})
        {
            auto const start=SampleStageStart(c,c.runway,reduced,side,rng);
            auto mission=Flying();
            mission.route=start.route;
            worst=std::max(worst,MeasureCorner(c,scenario,stage.name,mission,start.aircraft,failures));
        }
    }
    std::cout<<"[FEASIBILITY] "<<scenario<<" "<<stage.name<<" worst_threshold_cross_m="<<worst<<"\n";
}

/**
 * @brief Every curriculum level is landable: each corner of its start envelope, flown along the
 *        command at a FeasibilityBankDegrees bank, crosses the threshold within the Landable
 *        tolerances. An infeasible corner caps the stop rate below the promotion threshold and
 *        teaches the policy to give up instead of turning toward the runway.
 */
void CurriculumFeasibility()
{
    std::size_t failures=0;
    for(auto const& [scenario,path] : {std::pair<char const*,char const*>{"t6",PPO_FLIGHT_CONFIG},
        std::pair<char const*,char const*>{"pc7",PPO_FLIGHT_PC7_CONFIG}})
    {
        auto c=LoadConfig(std::filesystem::path(path));
        c.runway.origin={0,0,15};c.runway.headingDegrees=0;
        ApproachLevelCorners(c,scenario,failures);
        for(auto const& stage : c.training.curriculum.stages)
            StageCorners(c,scenario,stage,failures);
    }
    Check(failures==0,"every curriculum start corner crosses the threshold aligned enough to land");
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
        CentrelineIntercept();
        CentrelineInterceptTrajectory();
        GatePassage();
        RouteGuidance();
        RouteGuidanceUsesRunwayAxes();
        RouteMission();
        RouteTurnsAtBank();
        MovingDeckPose();
        MovingDeckMission();
        PreviousPotentials();
        DeckMotionStages();
        FlareTrajectory();
        LandingQuality();
        CrashDiagnostics();
        Landing();
        FiniteSurfaceContacts();
        BellyContact();
        SignedLogCompression();
        ObservationConditioning();
        Spawns();
        CurriculumSpawns();
        CurriculumScale();
        StageStarts();
        CurriculumFeasibility();
        EvaluationStratification();
        DisplayReferences();
    }
    catch(std::exception const& error) { std::cerr<<error.what()<<'\n'; return 1; }
    std::cout<<"Flight core regressions passed\n";
}
