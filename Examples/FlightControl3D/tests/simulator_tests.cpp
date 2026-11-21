/**
 * @file simulator_tests.cpp
 * @brief Regressions on the real JSBSim T-6: command filter, phase controls, resets,
 *        native contacts, touchdown and stop, nose-wheel steering and action-level shaping.
 *
 * Protects: the JSBSim adapter conventions (units, frames, signs of the rudder and the
 * nose wheel), the deck and sea ground callback, the rolling-start reset on the wheels and
 * the discounting of potentials inside one action, including truncated actions.
 *
 * Read next: Simulator/flight_environment.cpp, Simulator/jsbsim_aircraft.cpp.
 */
#include "test_support.hpp"
#include "flight_mission.hpp"
#include "flight_reward.hpp"
#include "flight_policy.hpp"
#include "flight_trim.hpp"
#include <utility>
using namespace ppo::examples::flight;

namespace {
Config T6Config()
{
    return LoadConfig(std::filesystem::path(PPO_FLIGHT_CONFIG));
}
/** @brief Trimmed approach controls, gear down, with an optional brake command. */
Controls Neutral(Config const& c,double brake=0)
{
    Controls controls{};
    controls.elevator=c.initial.elevatorTrim;
    controls.throttle=c.initial.throttle;
    controls.brake=brake;
    controls.gear=1;
    return controls;
}
std::size_t ControlObservation(std::size_t axis) { return ObservationSize-ContinuousActions+axis; }

/** @brief Check first-order command filtering and its observable reset state. */
void FilteredControls()
{
    auto const c=T6Config();
    FlightEnvironment env(c,DefaultDataRoot(),778U);
    auto observation=env.Reset(778U);
    // An airborne start begins on JSBSim's trim of its own speed and flight path, close to
    // the action reference trimmed on the nominal 3-degree approach.
    double const trim=env.Mission().filteredControls.elevator;
    double const throttle=env.Mission().filteredControls.throttle;
    Check(Near(trim,c.initial.elevatorTrim,0.05) && Near(throttle,c.initial.throttle,0.15)
        && Near(observation[ControlObservation(0)],trim,1e-6) && Near(observation[ControlObservation(1)],0)
        && Near(observation[ControlObservation(3)],throttle,1e-6),"reset exposes the trimmed applied controls");
    Controls target{};target.elevator=0.3;target.aileron=-0.6;target.rudder=0.2;target.throttle=1.0;
    double const actionSeconds=c.StepSeconds();
    double const rudderAuthority=c.mission.airborneRudderAuthority;
    for(int step=1;step<=2;++step)
    {
        auto const result=env.Step(target);
        Check(!result.terminated && !result.truncated,"the filter test remains in flight");
        double const alpha=-std::expm1(-step*actionSeconds/c.mission.controlFilterTimeConstantSeconds);
        auto const applied=env.Mission().filteredControls;
        observation=env.ObservationNow();
        Check(Near(applied.elevator,trim+(0.3-trim)*alpha) && Near(applied.aileron,-0.6*alpha)
            && Near(applied.rudder,0.2*rudderAuthority*alpha)
            && Near(applied.throttle,throttle+(1.0-throttle)*alpha),
            "continuous commands follow the physics-step first-order response");
        Check(Near(observation[ControlObservation(0)],applied.elevator,1e-6)
            && Near(observation[ControlObservation(1)],applied.aileron,1e-6)
            && Near(observation[ControlObservation(2)],applied.rudder,1e-6)
            && Near(observation[ControlObservation(3)],applied.throttle,1e-6),
            "the policy sees the commands actually applied to JSBSim");
    }
    auto invalid=target;invalid.aileron=std::numeric_limits<double>::quiet_NaN();
    auto const before=env.Mission().filteredControls;
    Rejects([&] { env.Step(invalid); },"a non-finite command is rejected");
    Check(Near(before.aileron,env.Mission().filteredControls.aileron),"a rejected command cannot advance the filter");
    observation=env.Reset(778U);
    Check(Near(observation[ControlObservation(0)],trim,1e-6) && Near(observation[ControlObservation(1)],0)
        && Near(observation[ControlObservation(3)],throttle,1e-6),
        "a new episode does not inherit the prior filter state");
}

/** @brief Training starts follow the curriculum difficulty and rolling share; rudder authority is reduced only in flight. */
void TrainingStartsAndRudderAuthority()
{
    auto c=T6Config();
    c.training.curriculum.rolloutStartShare=0.5;
    auto const& first=c.training.curriculum.firstDistanceMeters;
    double const alpha=-std::expm1(-c.StepSeconds()/c.mission.controlFilterTimeConstantSeconds);
    auto rudder=Neutral(c);rudder.rudder=0.8;
    FlightEnvironment environment(c,DefaultDataRoot(),123U);
    environment.SetDifficulty(0);
    bool sawRolling=false,sawAirborne=false;
    for(int episode=0;episode<12;++episode)
    {
        environment.Reset();
        auto const guidance=environment.CurrentGuidance();
        if(environment.StartedAirborne())
        {
            Check(environment.Mission().phase==Phase::Approach && !environment.Aircraft().weightOnWheels
                && guidance.distance>=first.minimum-1 && guidance.distance<=first.maximum+1 && std::abs(guidance.cross)<1,
                "difficulty zero training starts are short aligned finals");
            if(!sawAirborne)
            {
                environment.Step(rudder);
                Check(Near(environment.Mission().filteredControls.rudder,0.8*c.mission.airborneRudderAuthority*alpha),
                    "airborne rudder scaling precedes filtering");
            }
            sawAirborne=true;
            continue;
        }
        Check(environment.Mission().phase==Phase::Rollout && environment.Mission().touchedDown
            && environment.Aircraft().weightOnWheels && !environment.Aircraft().structuralContact
            && environment.ObservationNow()[RolloutObservation]>0.5F,
            "a rolling start uses native wheel contact without a touchdown reward");
        for(auto const& wheel:environment.Aircraft().wheels) Check(wheel.onGround,"all rollout wheels touch the runway");
        Check(environment.Mission().filteredControls.elevator==0 && environment.Mission().filteredControls.throttle==0,
            "a rolling start begins at idle with a neutral elevator");
        if(!sawRolling)
        {
            environment.Step(rudder);
            Check(Near(environment.Mission().filteredControls.rudder,0.8*alpha),"rudder authority remains full during rollout");
        }
        sawRolling=true;
    }
    Check(sawRolling && sawAirborne,"training mixes rolling and airborne starts");
    environment.Reset(4342U);
    Check(environment.StartedAirborne() && environment.CurrentGuidance().distance>=c.initial.distanceMeters.minimum-1,
        "a seeded episode defaults to the full task whatever the curriculum level");
    // Seeded evaluation episodes follow their explicit start, independently of the training stream.
    environment.SetDifficulty(1);
    environment.Reset(4342U,EpisodeStart{false,0.0});
    auto const shortFinal=environment.CurrentGuidance();
    Check(environment.StartedAirborne() && shortFinal.distance>=first.minimum-1 && shortFinal.distance<=first.maximum+1
        && std::abs(shortFinal.cross)<1,"a seeded difficulty-zero start is a short aligned final");
    environment.Reset();
    environment.Reset(4342U,EpisodeStart{false,0.0});
    Check(Near(environment.CurrentGuidance().along,shortFinal.along,1e-9),
        "a seeded start is reproducible after training episodes");
    environment.Reset(4343U,EpisodeStart{true,0.0});
    Check(!environment.StartedAirborne() && environment.Mission().phase==Phase::Rollout
        && environment.Mission().touchedDown && environment.Aircraft().weightOnWheels,
        "a seeded rolling start begins on the runway without a touchdown reward");
}
/** @brief Opposite aileron, rudder and throttle commands produce ordered native responses. */
void ControlProbes()
{
    auto const config=T6Config();
    for(std::uint32_t seed : {42U,4342U,4343U})
    {
        std::array<double,3> rolls{};
        constexpr std::array<double,3> commands{-0.15,0.0,0.15};
        for(std::size_t trial=0;trial<commands.size();++trial)
        {
            FlightEnvironment env(config,DefaultDataRoot(),seed);env.Reset(seed);
            auto controls=Neutral(config);controls.aileron=commands[trial];
            for(int step=0;step<30 && env.Mission().outcome==Outcome::Flying;++step) env.Step(controls);
            Check(env.Mission().outcome==Outcome::Flying && IsFinite(env.Aircraft()),"the aileron probe stays in flight");
            rolls[trial]=env.Aircraft().roll;
        }
        Check(rolls[0]<rolls[1] && rolls[1]<rolls[2],"opposite aileron commands produce ordered roll responses");
    }
    constexpr std::array<double,3> rudderCommands{-0.21,0.0,0.21};
    std::array<double,3> headingChanges{};
    for(std::size_t trial=0;trial<rudderCommands.size();++trial)
    {
        FlightEnvironment env(config,DefaultDataRoot(),4342U);env.Reset(4342U);
        auto const initial=env.Aircraft();
        auto controls=Neutral(config);controls.rudder=rudderCommands[trial];
        for(int step=0;step<50 && env.Mission().outcome==Outcome::Flying;++step) env.Step(controls);
        Check(env.Mission().outcome==Outcome::Flying && IsFinite(env.Aircraft()),"the rudder probe stays in flight");
        headingChanges[trial]=WrapAngle(env.Aircraft().heading-initial.heading);
    }
    Check(headingChanges[0]>headingChanges[1] && headingChanges[1]>headingChanges[2],
        "opposite rudder commands produce opposite yaw responses");
    std::array<double,2> speeds{};
    for(std::size_t trial=0;trial<speeds.size();++trial)
    {
        FlightEnvironment env(config,DefaultDataRoot(),4342U);env.Reset(4342U);
        auto controls=Neutral(config);controls.throttle=trial==0 ? 0.0 : 1.0;
        for(int step=0;step<100 && env.Mission().outcome==Outcome::Flying;++step) env.Step(controls);
        Check(env.Mission().outcome==Outcome::Flying && IsFinite(env.Aircraft()),"the propulsion probe stays in flight");
        speeds[trial]=env.Aircraft().airspeed;
    }
    Check(speeds[1]>speeds[0]+1.0,"throttle changes the native T-6 speed");
}

/** @brief Native contacts use the finite deck; the sea surrounds it. */
void FiniteSurfaceContactAuthority()
{
    auto c=T6Config();
    c.runway.landingMarginMeters=0;
    for(double const heading:{0.0,90.0,181.0,359.0})
    {
        c.runway.headingDegrees=heading;c.runway.origin={0,0,15};
        JsbsimAircraft fdm(c,DefaultDataRoot());fdm.SetRunway(c.runway);
        Check(Near(fdm.ProbeGround(FromRunway(c.runway,500,0,10)).height,10,1e-6),"JSBSim uses the deck plane");
        auto const offDeck=FromRunway(c.runway,500,c.runway.widthMeters+100,10);
        Check(Near(fdm.ProbeGround(offDeck).height,offDeck.up-c.runwayRandomization.seaHeightMeters,1e-6),
            "the finite deck falls back to the sea outside the runway");
        AircraftState initial{};
        initial.position=FromRunway(c.runway,500,c.runway.widthMeters+50,0.1);
        initial.airspeed=56;initial.heading=heading*DegreesToRadians;initial.gearPosition=1;
        initial.velocity=Scale(RunwayForward(c.runway),56);
        auto before=fdm.Reset(initial);
        Controls controls=Neutral(c);controls.flap=c.initial.approachFlap;
        auto after=fdm.Advance(controls);
        Check(after.wheelDataValid && !after.weightOnWheels && !after.structuralContact,
            "a native aircraft crossing the off-deck plane remains airborne");
        MissionState mission{};
        Check(!AdvanceMission(c,mission,before,after).terminated && !mission.contactAttempted,
            "the mission cannot invent contact on the infinite runway extension");
        initial.position.up=c.runwayRandomization.seaHeightMeters+6;initial.airspeed=3;initial.velocity={0,0,-1};
        before=fdm.Reset(initial);mission={};
        controls.throttle=0;
        for(int step=0;step<500 && mission.outcome==Outcome::Flying;++step)
        {
            after=fdm.Advance(controls);
            AdvanceMission(c,mission,before,after);before=after;
        }
        Check(mission.outcome==Outcome::Crash && (after.structuralContact || after.weightOnWheels),
            "a real native impact with the sea remains a terminal crash");
    }
}

void PhaseControlMapping()
{
    auto const c=T6Config();
    Controls requested{};requested.elevator=0.1;requested.aileron=-0.2;requested.rudder=0.3;
    requested.throttle=c.initial.throttle;requested.brake=0.8;requested.flap=0.99;
    auto const approach=ApplyPhaseControls(requested,Phase::Approach,c.initial);
    auto const rollout=ApplyPhaseControls(requested,Phase::Rollout,c.initial);
    Check(approach.brake==0 && rollout.brake==0.8,"brakes are disabled in flight and enabled during rollout");
    Check(Near(approach.flap,c.initial.approachFlap) && Near(rollout.flap,c.initial.rolloutFlap),"flaps are phase-controlled");
    Check(Near(approach.aileron,requested.aileron) && Near(approach.throttle,requested.throttle),
        "continuous controls survive phase mapping");
}

/** @brief Compares a reused T-6 with a fresh simulator after a terminal contact. */
void CheckResetAfterContact(JsbsimAircraft& reused,Config const& config)
{
    JsbsimAircraft fresh(config,DefaultDataRoot());fresh.SetRunway(config.runway);
    AircraftState initial{};
    initial.position=FromRunway(config.runway,-6000,400,500);
    initial.airspeed=65;initial.heading=0;initial.gearPosition=1;
    initial.pitch=2*DegreesToRadians;initial.velocity={65,0,0};
    auto actual=reused.Reset(initial);
    auto expected=fresh.Reset(initial);
    Controls controls=Neutral(config);controls.flap=config.initial.approachFlap;
    for(int step=0;step<=100;++step)
    {
        Check(IsFinite(actual) && IsFinite(expected) && !actual.weightOnWheels && !actual.structuralContact
            && Near(actual.position.north,expected.position.north,1e-5) && Near(actual.position.up,expected.position.up,1e-5)
            && Near(actual.velocity.north,expected.velocity.north,1e-5) && Near(actual.velocity.up,expected.velocity.up,1e-5)
            && Near(actual.roll,expected.roll,1e-6) && Near(actual.pitch,expected.pitch,1e-6)
            && Near(actual.airspeed,expected.airspeed,1e-5) && Near(actual.flapPosition,expected.flapPosition,1e-6),
            "a reset after a ground episode must not retain any previous state");
        if(step<100) {actual=reused.Advance(controls);expected=fresh.Advance(controls);}
    }
}

/** @brief Native T-6 wheels can touch safely and brake to a Stop; a hard touchdown crashes. */
void NativeTouchdownAndStop()
{
    auto config=T6Config();
    config.runway.origin={0,0,15};config.runway.headingDegrees=0;config.initial.throttle=0;
    JsbsimAircraft fdm(config,DefaultDataRoot());fdm.SetRunway(config.runway);
    AircraftState initial{};initial.position=FromRunway(config.runway,100,0,3);
    initial.airspeed=config.mission.flareSpeed;initial.heading=0;initial.gearPosition=1;
    initial.velocity={initial.airspeed,0,-1};initial.pitch=1.5*DegreesToRadians;
    auto const measured=fdm.Reset(initial);
    double lowestWheel=100;
    for(auto const& wheel:measured.wheels) lowestWheel=std::min(lowestWheel,DeckCoordinates(config.runway,wheel.point).up);
    initial.position.up+=0.02-lowestWheel;
    auto before=fdm.Reset(initial);
    Check(before.wheelDataValid && !before.weightOnWheels && !before.structuralContact,
        "the contact regression starts with native wheels above the deck");
    MissionState mission{};
    double touchdownReward=0,stopReward=0;
    for(int step=0;step<12000 && mission.outcome==Outcome::Flying;++step)
    {
        auto const controls=ApplyPhaseControls(Neutral(config,1),mission.phase,config.initial);
        auto const after=fdm.Advance(controls);
        auto const result=AdvanceMission(config,mission,before,after);
        touchdownReward+=result.rewards[static_cast<std::size_t>(Reward::Touchdown)];
        stopReward+=result.rewards[static_cast<std::size_t>(Reward::Stop)];
        Check(IsFinite(after),"native touchdown and braking remain finite");
        before=after;
    }
    Check(mission.firstWheelContact && mission.touchedDown && mission.touchdownSinkMps<config.mission.maxTouchdownSink,
        "native JSBSim contact satisfies the touchdown limits");
    Check(mission.outcome==Outcome::Stopped && mission.touchdownQuality>0.9
        && Near(touchdownReward,Weight(config.reward,Reward::Touchdown)*mission.touchdownQuality)
        && Near(stopReward,Weight(config.reward,Reward::Stop)),"native braking reaches Stop and pays both landing events");
    CheckResetAfterContact(fdm,config);

    initial.position=FromRunway(config.runway,100,0,5);
    initial.velocity={initial.airspeed,0,-15};
    before=fdm.Reset(initial);
    mission={};
    Controls idle{};idle.flap=config.initial.approachFlap;idle.gear=1;idle.gear=1;
    for(int step=0;step<500 && mission.outcome==Outcome::Flying;++step)
    {
        auto const after=fdm.Advance(idle);
        AdvanceMission(config,mission,before,after);
        before=after;
    }
    Check(mission.outcome==Outcome::Crash,"a hard T6 touchdown produces a terminal crash");
    CheckResetAfterContact(fdm,config);
}

/**
 * @brief Rolls a seeded landed T-6 under one constant raw policy output until the episode ends.
 * @param brakeRaw Raw brake action; every other raw action is neutral.
 * @return Final mission state.
 */
MissionState RollWithRawBrake(double brakeRaw)
{
    auto c=T6Config();
    // A landing at the flare speed near the end of the touchdown zone.
    c.initial.rolloutStart.speedMps={c.mission.flareSpeed,c.mission.flareSpeed};
    c.initial.rolloutStart.alongMeters={900,900};
    c.initial.rolloutStart.crossMeters={0,0};
    FlightEnvironment environment(c,DefaultDataRoot(),31U);
    environment.Reset(31U,EpisodeStart{true,0.0});
    PolicyAction action{};
    action.brakeActive=true;
    action.raw[BrakeAction]=brakeRaw;
    while(environment.Mission().outcome==Outcome::Flying) environment.Step(ToControls(action,c.initial));
    return environment.Mission();
}

/** @brief A neutral policy output brakes the native T-6 to a stop; released brakes overrun the runway. */
void NeutralRolloutBrakesToStop()
{
    auto const neutral=RollWithRawBrake(0.0);
    Check(neutral.outcome==Outcome::Stopped && neutral.stopFromThreshold<T6Config().runway.lengthMeters,
        "neutral ground actions stop the T-6 on the runway");
    // The previous trained mean (-0.19) clamped to no brake at all: it now brakes.
    Check(RollWithRawBrake(-0.19).outcome==Outcome::Stopped,"a slightly negative brake output still stops");
    Check(RollWithRawBrake(-1.0).outcome==Outcome::Crash,"without brakes the T-6 rolls off the runway end");
}

/**
 * @brief Rolls a seeded landed T-6 at taxi speed with a constant rudder command.
 * @param rudder Rudder pedal command in [-1, 1].
 * @return Heading change relative to the runway after two seconds, in radians.
 */
double TaxiHeadingChange(Config c,double rudder)
{
    // At taxi speed the rudder has almost no aerodynamic authority: only the
    // nose wheel can turn the aircraft.
    constexpr double TaxiSpeedMps=8.0;
    c.initial.rolloutStart.speedMps={TaxiSpeedMps,TaxiSpeedMps};
    c.initial.rolloutStart.alongMeters={900,900};
    c.initial.rolloutStart.crossMeters={0,0};
    FlightEnvironment environment(c,DefaultDataRoot(),37U);
    environment.Reset(37U,EpisodeStart{true,0.0});
    double const initialHeading=environment.Aircraft().heading;
    auto controls=Neutral(c);
    controls.rudder=rudder;
    for(int step=0;step<20 && environment.Mission().outcome==Outcome::Flying;++step) environment.Step(controls);
    Check(environment.Mission().outcome==Outcome::Flying,"the taxi steering probe stays on the runway");
    return WrapAngle(environment.Aircraft().heading-initialHeading);
}

/** @brief Rudder pedals steer the native nose wheel, so the ground track can be corrected at low speed. */
void NoseWheelSteering()
{
    // Same convention as the airborne rudder probe: a negative command yaws the nose right.
    double const noseRight=TaxiHeadingChange(T6Config(),-0.5);
    double const straight=TaxiHeadingChange(T6Config(),0.0);
    double const noseLeft=TaxiHeadingChange(T6Config(),0.5);
    std::cout<<"[STEERING_PROBE] nose_right="<<noseRight/DegreesToRadians<<" straight="<<straight/DegreesToRadians
        <<" nose_left="<<noseLeft/DegreesToRadians<<" deg\n";
    // The rudder alone turns the T-6 by less than one degree here; the nose wheel turns it much more.
    constexpr double MinimumTurnRadians=3.0*DegreesToRadians;
    Check(noseRight>straight+MinimumTurnRadians && noseLeft<straight-MinimumTurnRadians,
        "rudder pedals steer the nose wheel in the direction of the aerodynamic yaw at taxi speed");
}

/** @brief Every shaping channel of a real 0.1 s action telescopes, including a partial timeout. */
void CheckActionPotentials(StepResult const& step,std::array<double,RewardCount> const& before,
    std::array<double,RewardCount> const& after,double gamma,char const* message)
{
    for(std::size_t i=0;i<RewardCount;++i)
        if(IsPotentialChannel(static_cast<Reward>(i)))
            Check(Near(step.rewards[i],gamma*after[i]-before[i],1e-7),message);
}
void ActionLevelPotential()
{
    auto c=T6Config();
    for(std::uint32_t seed : {92U,93U,94U})
    {
        FlightEnvironment env(c,DefaultDataRoot(),seed);env.Reset(seed);
        auto const before=ShapingPotentials(env.Settings(),env.Mission(),env.Aircraft());
        auto const step=env.Step(Neutral(c));
        Check(!step.terminated && !step.truncated,"the potential probe remains airborne");
        auto const after=ShapingPotentials(env.Settings(),env.Mission(),env.Aircraft());
        CheckActionPotentials(step,before,after,c.training.gamma,"a PPO action telescopes the discounted substep potentials");
        double sum=0;for(double reward:step.rewards) sum+=reward;
        Check(Near(sum,step.reward,1e-10),"an action preserves the reward channel sum");
    }
    // Timeout after 0.13 s: the second PPO action executes only 3 substeps.
    c.mission.maxSeconds=0.13;
    FlightEnvironment partial(c,DefaultDataRoot(),95U);partial.Reset(95U);
    Check(!partial.Step(Neutral(c)).truncated,"the first 0.1 s action precedes the partial timeout");
    auto const pre=ShapingPotentials(partial.Settings(),partial.Mission(),partial.Aircraft());
    auto const last=partial.Step(Neutral(c));
    auto const post=ShapingPotentials(partial.Settings(),partial.Mission(),partial.Aircraft());
    Check(last.truncated && !last.terminated && partial.Mission().outcome==Outcome::Timeout,
        "the second action is the three-substep timeout");
    CheckActionPotentials(last,pre,post,c.training.gamma,"a partial timeout keeps gamma-correct potentials");
    Rejects([&] { partial.Step(Neutral(c)); },"a step after the episode boundary is rejected");
}
/** @brief Native reset preserves sampled attitude, speed and course, and leaves no state behind. */
void IndependentResetAttitude()
{
    auto c=T6Config();
    std::mt19937 rng(183U);
    JsbsimAircraft fdm(c,DefaultDataRoot());
    for(double heading : {0.,90.,181.,359.})
    {
        auto runway=SampleRunway(c,rng);runway.headingDegrees=heading;
        fdm.SetRunway(runway);
        for(double roll : {0.,5.}) for(double pitch : {-3.,3.}) for(double slope : {0.,1.5})
        {
            c.initial.rollDegrees=roll;c.initial.pitchDegrees={pitch,pitch};c.initial.slopeToleranceDegrees=slope;
            auto const requested=SampleApproachStart(c,runway,ApproachSide::Left,1.0,rng);
            auto const actual=fdm.Reset(requested);
            // JSBSim keeps the requested climb rate and course in the aircraft's
            // own tangent frame; the world frame is tangent at the runway
            // reference, so Earth curvature tilts them slightly at 7-14 km.
            constexpr double CurvatureVerticalSpeedToleranceMps=0.15;
            constexpr double CurvatureCourseToleranceRadians=0.003;
            double const requestedCourse=std::atan2(requested.velocity.east,requested.velocity.north);
            double const actualCourse=std::atan2(actual.velocity.east,actual.velocity.north);
            Check(Near(actual.roll,requested.roll,1e-8) && Near(actual.pitch,requested.pitch,1e-8)
                && Near(WrapAngle(actual.heading-requested.heading),0,1e-8)
                && Near(actual.airspeed,requested.airspeed,1e-5)
                && Near(actual.velocity.up,requested.velocity.up,CurvatureVerticalSpeedToleranceMps)
                && Near(WrapAngle(actualCourse-requestedCourse),0,CurvatureCourseToleranceRadians),
                "the native reset preserves attitude, calibrated speed, climb rate and course");
            Check(Near(actual.position.north,requested.position.north,0.02) && Near(actual.position.up,requested.position.up,0.02),
                "the JSBSim geodetic conversion preserves the local position");
            Controls manoeuvre=Neutral(c);manoeuvre.elevator=0.2;manoeuvre.aileron=-0.2;manoeuvre.rudder=0.1;
            for(int step=0;step<20;++step) fdm.Advance(manoeuvre);
            auto const repeated=fdm.Reset(requested);
            Check(Near(repeated.roll,actual.roll,1e-8) && Near(repeated.pitch,actual.pitch,1e-8)
                && Near(repeated.velocity.north,actual.velocity.north,1e-7) && Near(repeated.velocity.up,actual.velocity.up,1e-7),
                "a previous manoeuvre cannot contaminate the next reset");
        }
    }
    auto invalid=SampleApproachStart(c,SampleRunway(c,rng),ApproachSide::Left,1.0,rng);
    invalid.velocity.up=1000;
    Rejects([&] {fdm.Reset(invalid);},"an impossible initial climb fails explicitly");
}
} // namespace

/** @brief A retracted start extends the gear only on request, then keeps it latched down. */
void GearExtensionLatch()
{
    auto const c=T6Config();
    FlightEnvironment env(c,DefaultDataRoot(),31U);
    EpisodeStart start{};start.difficulty=0;start.gearUp=true;
    auto observation=env.Reset(31U,start);
    Check(Near(env.Aircraft().gearPosition,0) && !env.Mission().gearDownCommanded
        && observation[Slot(ObservationIndex::GearPosition)]==0 && observation[GearCommandObservation]==0,
        "a gear-up start begins retracted with no extension commanded");
    Controls hold=Neutral(c);hold.gear=0;
    for(int action=0;action<20;++action) env.Step(hold);
    Check(Near(env.Aircraft().gearPosition,0) && !env.Mission().gearDownCommanded,
        "without a request the gear stays retracted");
    Controls extend=hold;extend.gear=1;
    env.Step(extend);
    Check(env.Mission().gearDownCommanded && env.ObservationNow()[GearCommandObservation]==1,
        "the first request latches the extension");
    double previous=env.Aircraft().gearPosition;
    bool locked=false;
    for(int action=0;action<300 && !locked;++action)
    {
        auto const result=env.Step(hold);
        Check(!result.terminated && !result.truncated,"the gear extension test stays airborne");
        double const position=env.Aircraft().gearPosition;
        Check(position>=previous-1e-9 && env.Mission().gearDownCommanded,
            "a latched gear never retracts, even when the lever request stops");
        previous=position;
        locked=position>=c.mission.gearDownMinimum;
    }
    Check(locked && Near(env.ObservationNow()[Slot(ObservationIndex::GearPosition)],env.Aircraft().gearPosition,1e-6),
        "the native gear reaches the down-and-locked position and is observed");
    auto const gearDown=env.Reset(32U);
    Check(Near(env.Aircraft().gearPosition,1) && env.Mission().gearDownCommanded && gearDown[GearCommandObservation]==1,
        "a default start keeps the gear down and latched");
}

/** @brief Touching the runway with the gear retracted is a crash, whatever the sink rate. */
void GearUpTouchdownCrashes()
{
    auto config=T6Config();
    config.runway.origin={0,0,15};config.runway.headingDegrees=0;
    JsbsimAircraft fdm(config,DefaultDataRoot());fdm.SetRunway(config.runway);
    AircraftState initial{};initial.position=FromRunway(config.runway,100,0,2);
    initial.airspeed=config.mission.flareSpeed;initial.heading=0;initial.gearPosition=0;
    initial.velocity={initial.airspeed,0,-0.5};initial.pitch=1.5*DegreesToRadians;
    auto before=fdm.Reset(initial);
    Check(Near(before.gearPosition,0),"the native gear starts retracted");
    MissionState mission{};mission.gearDownCommanded=false;
    Controls controls=Neutral(config);controls.gear=0;controls.throttle=0;controls.flap=config.initial.approachFlap;
    CrashDiagnostic diagnostic{};
    for(int step=0;step<3000 && mission.outcome==Outcome::Flying;++step)
    {
        auto const after=fdm.Advance(controls);
        AdvanceMission(config,mission,before,after,&diagnostic);
        before=after;
    }
    Check(mission.outcome==Outcome::Crash && !mission.touchedDown,"a gear-up contact is a terminal crash");
    auto const& gear=diagnostic.conditions[static_cast<std::size_t>(CrashReason::Gear)];
    auto const& structure=diagnostic.conditions[static_cast<std::size_t>(CrashReason::StructuralContact)];
    Check(gear.failed || structure.failed,"the crash names the retracted gear or the belly contact");
}

/** @brief Seeded episodes resolve the curriculum: full task is the last stage, with its route, gear draw and trim. */
void StageEpisodes()
{
    auto const c=T6Config();
    FlightEnvironment env(c,DefaultDataRoot(),41U);
    auto const& last=c.training.curriculum.stages.back();
    int gearUp=0;
    constexpr int Episodes=12;
    for(std::uint32_t seed=600U;seed<600U+Episodes;++seed)
    {
        EpisodeStart start{};start.difficulty=1.0;
        env.Reset(seed,start);
        auto const& mission=env.Mission();
        Check(last.kind==StageKind::Holding && mission.route.count==5 && mission.nextGate==0,
            "the full task flies the holding lap of the last stage");
        Check(mission.gearDownCommanded==(env.Aircraft().gearPosition>0.5),"the gear position matches its latch");
        gearUp+=mission.gearDownCommanded ? 0 : 1;
        // Level flight needs more power than the 3-degree descent of the action reference.
        Check(mission.filteredControls.throttle>c.initial.throttle,"a level start begins on its own trim");
        auto const local=DeckCoordinates(env.Settings().runway,env.Aircraft().position);
        Check(local.north<=last.startAlongMeters.maximum+1 && local.up>last.heightMeters.minimum-5,
            "the holding start is inbound before the fix, at the holding height");
    }
    Check(gearUp>0 && gearUp<Episodes,"the stage draws gear-up and gear-down starts");
    EpisodeStart first{};first.difficulty=0;
    env.Reset(700U,first);
    Check(env.Mission().route.count==0 && env.Mission().gearDownCommanded,"the first level is a direct approach, gear down");
}

/** @brief Heading change of an airborne aircraft after a few seconds of constant rudder, in radians. */
double AirborneHeadingChange(Config const& c,double rudder)
{
    FlightEnvironment environment(c,DefaultDataRoot(),43U);
    EpisodeStart start{};start.difficulty=0;
    environment.Reset(43U,start);
    double const initialHeading=environment.Aircraft().heading;
    auto controls=Neutral(c);
    controls.rudder=rudder;
    for(int step=0;step<20;++step) environment.Step(controls);
    Check(environment.Mission().outcome==Outcome::Flying,"the airborne rudder probe stays in flight");
    return WrapAngle(environment.Aircraft().heading-initialHeading);
}

/** @brief The PC-7 loads, trims by simulated flight to its action reference, steers and extends its gear. */
void Pc7Aircraft()
{
    auto const c=LoadConfig(std::filesystem::path(PPO_FLIGHT_PC7_CONFIG));
    Check(c.aircraft=="pc7" && !c.performance.jsbsimTrim && c.performance.name=="pc7",
        "the PC-7 scenario resolves the PC-7 performance");
    JsbsimAircraft aircraft(c,DefaultDataRoot());
    TrimCondition nominal{};
    nominal.calibratedSpeed=c.mission.approachSpeed;
    nominal.flightPathRadians=-c.mission.glideDegrees*DegreesToRadians;
    nominal.flap=c.initial.approachFlap;
    nominal.heightMeters=300;
    Check(!aircraft.Trim(nominal).converged,"FGTrim is never run on the PC-7 turboprop");
    auto const flown=FlyTrim(aircraft,c,nominal);
    std::cout<<"[FLIGHT_TRIM] aircraft=pc7 method=flight elevator="<<flown.elevator<<" throttle="<<flown.throttle<<'\n';
    Check(flown.converged && Near(flown.elevator,c.initial.elevatorTrim,0.005) && Near(flown.throttle,c.initial.throttle,0.01),
        "the PC-7 action reference is its trimmed 3-degree approach");
    // Same steering convention as the T-6: the nose wheel turns the way the rudder yaws.
    double const airborne=AirborneHeadingChange(c,-0.5)-AirborneHeadingChange(c,0.5);
    double const taxi=TaxiHeadingChange(c,-0.5)-TaxiHeadingChange(c,0.5);
    std::cout<<"[STEERING_PROBE] aircraft=pc7 airborne="<<airborne/DegreesToRadians<<" taxi="<<taxi/DegreesToRadians<<" deg\n";
    Check(airborne>0 && taxi>3.0*DegreesToRadians,"PC-7 pedals yaw and steer the nose wheel the same way");
    // A retracted start extends and locks the PC-7 gear once commanded.
    FlightEnvironment env(c,DefaultDataRoot(),47U);
    EpisodeStart start{};start.difficulty=0;start.gearUp=true;
    env.Reset(47U,start);
    Check(Near(env.Aircraft().gearPosition,0),"the PC-7 gear starts retracted");
    auto extend=Neutral(c);
    bool locked=false;
    for(int action=0;action<300 && !locked;++action)
    {
        env.Step(extend);
        locked=env.Aircraft().gearPosition>=c.mission.gearDownMinimum;
    }
    Check(locked && env.Mission().outcome==Outcome::Flying,"the PC-7 gear extends and locks in flight");
}

/** @brief JSBSim's trim of the nominal approach reproduces the action reference of the scenario. */
void ApproachTrimMatchesActionReference()
{
    auto const c=T6Config();
    JsbsimAircraft aircraft(c,DefaultDataRoot());
    TrimCondition nominal{};
    nominal.calibratedSpeed=c.mission.approachSpeed;
    nominal.flightPathRadians=-c.mission.glideDegrees*DegreesToRadians;
    nominal.flap=c.initial.approachFlap;
    nominal.heightMeters=300;
    auto const trim=aircraft.Trim(nominal);
    std::cout<<"[FLIGHT_TRIM] elevator="<<trim.elevator<<" throttle="<<trim.throttle
        <<" pitch_deg="<<trim.pitch/DegreesToRadians<<" converged="<<trim.converged<<'\n';
    Check(trim.converged,"JSBSim trims the nominal 3-degree approach");
    Check(Near(trim.elevator,c.initial.elevatorTrim,0.002) && Near(trim.throttle,c.initial.throttle,0.002),
        "the scenario action reference is the trimmed 3-degree approach");
    // The trim leaves no pitch-trim command behind: the next episode flies on the elevator only.
    std::mt19937 rng(1U);
    auto const reset=aircraft.Reset(SampleApproachStart(c,c.runway,ApproachSide::Left,0.0,rng));
    Check(IsFinite(reset),"a reset after a trim restores a normal episode start");
    // The trim of a condition does not depend on what the simulator flew before.
    Controls manoeuvre=Neutral(c);manoeuvre.elevator=0.3;manoeuvre.aileron=-0.4;manoeuvre.throttle=1;
    for(int step=0;step<300;++step) aircraft.Advance(manoeuvre);
    auto const again=aircraft.Trim(nominal);
    Check(again.converged && again.elevator==trim.elevator && again.throttle==trim.throttle && again.pitch==trim.pitch,
        "a trim is reproduced bit for bit after any previous flight");
}

int main()
{
    try
    {
        FilteredControls();
        TrainingStartsAndRudderAuthority();
        ControlProbes();
        FiniteSurfaceContactAuthority();
        PhaseControlMapping();
        NativeTouchdownAndStop();
        NeutralRolloutBrakesToStop();
        NoseWheelSteering();
        ActionLevelPotential();
        IndependentResetAttitude();
        ApproachTrimMatchesActionReference();
        GearExtensionLatch();
        GearUpTouchdownCrashes();
        StageEpisodes();
        Pc7Aircraft();
    }
    catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
    std::cout<<"Native JSBSim regressions passed\n";
}
