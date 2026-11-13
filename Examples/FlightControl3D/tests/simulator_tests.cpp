#include "test_support.hpp"
using namespace ppo::examples::flight;

namespace {
void FiniteSurfaceContactAuthority(Config c)
{
    c.training.curriculumStage=CurriculumStage::Joint;c.initial.gearDown=false;
    c.runwayRandomization.seaHeightMeters=0;
    for(double const heading:{0.0,90.0,181.0,359.0})
    {
        RunwayMotion motion{};motion.base=c.runway;
        motion.base.headingDegrees=heading;motion.base.origin={0,0,15};
        motion.base.pitch=0;motion.base.roll=0;motion.seaHeight=0;motion.period=10;
        c.runway=RunwayAt(motion,0);
        JsbsimAircraft fdm(c,DefaultDataRoot());fdm.SetRunwayMotion(motion);
        AircraftState initial{};
        initial.position=FromRunway(c.runway,500,c.runway.widthMeters+50,0.1);
        initial.airspeed=56;initial.heading=heading*DegreesToRadians;
        initial.velocity=Scale(RunwayForward(c.runway),56);
        auto before=fdm.Reset(initial,Phase::Approach);
        auto after=fdm.Advance({0,0,0,c.initial.throttle,0,false,c.initial.approachFlap});
        Check(after.wheelDataValid && !after.weightOnWheels && !after.structuralContact
            && fdm.ProbeGround(after.position).height>10,
            "native aircraft crossing the off-deck tangent plane remains physically airborne");
        MissionState mission{};mission.phase=Phase::Approach;
        Check(!AdvanceMission(c,{},mission,before,after).terminated && !mission.contactAttempted,
            "mission cannot invent native contact on the infinite runway extension");

        initial.position.up=motion.seaHeight+3;initial.airspeed=3;initial.velocity={0,0,-1};
        before=fdm.Reset(initial,Phase::Approach);mission={};mission.phase=Phase::Approach;
        for(int step=0;step<500 && mission.outcome==Outcome::Flying;++step)
        {
            after=fdm.Advance({0,0,0,0,0,false,c.initial.approachFlap});
            AdvanceMission(c,{},mission,before,after);before=after;
        }
        Check(mission.outcome==Outcome::Crash && (after.structuralContact || after.weightOnWheels),
            "real native impact with the sea remains a terminal crash");
    }
}
void PropulsionAndPhaseReset(Config c)
{
    c.initial.distanceMeters={6000,6000};c.initial.speedMps={56,56};
    c.initial.rollDegrees={0,0};c.initial.pitchDegrees={2,2};c.initial.slopeDegrees={-3,-3};
    c.initial.headingOffsetDegrees={0,0};
    c.runwayRandomization.speedMps={0,0};c.runwayRandomization.heaveAmplitudeMeters={0,0};
    c.runwayRandomization.pitchAmplitudeDegrees={0,0};c.runwayRandomization.rollAmplitudeDegrees={0,0};
    std::mt19937 rng(91U);
    auto const motion=SampleRunway(c,rng);
    auto const runway=RunwayAt(motion,0);
    auto const initial=SampleAircraft(c,runway,SpawnPlacement::LocalizerGlide,rng);
    JsbsimAircraft fdm(c,DefaultDataRoot());fdm.SetRunwayMotion(motion);
    std::array<double,3> speeds{};
    for(std::size_t run=0;run<speeds.size();++run)
    {
        fdm.Reset(initial);
        double const throttle=run==1 ? 1.0 : 0.0;
        AircraftState after{};
        for(int step=0;step<500;++step)
            after=fdm.Advance({0,0,0,throttle,0,false,c.initial.flap});
        Check(IsFinite(after),"native powered flight stays finite after reset");
        speeds[run]=after.airspeed;
    }
    Check(speeds[1]>speeds[0]+5 && Near(speeds[0],speeds[2],1e-7),
        "native throttle changes airspeed after reset and no propulsion state leaks between episodes");
    for(auto phase:{Phase::Patrol,Phase::Approach,Phase::Rollout})
    {
        auto const requested=ApplyPhaseControls({},phase,c.initial);
        auto const reset=fdm.Reset(initial,phase);
        auto const after=fdm.Advance(requested);
        Check(Near(reset.flapPosition,requested.flap,1e-10)
            && Near(after.flapPosition,requested.flap,1e-10),
            "native flap actuator starts at its phase detent without a first-action transient");
    }
}
void PhaseControlMapping(Config const& c)
{
    Controls const requested{0.1,-0.2,0.3,c.initial.throttle,0.8,true,0.99};
    auto const patrol=ApplyPhaseControls(requested,Phase::Patrol,c.initial);
    auto const approach=ApplyPhaseControls(requested,Phase::Approach,c.initial);
    auto const rollout=ApplyPhaseControls(requested,Phase::Rollout,c.initial);
    Check(patrol.brake==0 && approach.brake==0 && rollout.brake==0.8,
        "brakes remain disabled in flight and enabled during rollout");
    Check(Near(patrol.flap,c.initial.flap) && Near(approach.flap,c.initial.approachFlap)
        && Near(rollout.flap,c.initial.rolloutFlap),"flaps are phase-controlled");
    Check(patrol.gearDown && approach.gearDown && rollout.gearDown
        && Near(approach.aileron,requested.aileron) && Near(approach.throttle,requested.throttle),
        "continuous controls and discrete gear survive phase mapping");
    for(auto stage:{CurriculumStage::Loc,CurriculumStage::Glide})
        Check(ApplyPhaseControls(requested,Phase::Approach,c.initial,stage).gearDown==c.initial.gearDown,
            "capture lessons preserve the configured gear instead of applying untrained switches");
}
struct InterceptProbe final
{
    double initialError{},finalError{},localizerReward{};
    bool ended{};
};

InterceptProbe ProbeIntercept(Config config,double aileron)
{
    config.initial.runwayAlignedProbability=0;
    config.initial.distanceMeters={10000,10000};config.initial.bearingDegrees={20,20};
    config.initial.heightMeters={600,600};config.initial.speedMps={65,65};
    config.initial.headingOffsetDegrees={0,0};config.initial.rollDegrees={0,0};
    config.initial.pitchDegrees={2,2};config.initial.slopeDegrees={0,0};
    auto& randomization=config.runwayRandomization;
    randomization.northMeters={0,0};randomization.eastMeters={0,0};
    randomization.headingOffsetDegrees={0,0};randomization.deckHeightMeters={15,15};
    randomization.speedMps={0,0};randomization.heaveAmplitudeMeters={0,0};
    randomization.pitchAmplitudeDegrees={0,0};randomization.rollAmplitudeDegrees={0,0};
    FlightEnvironment environment(config,DefaultDataRoot(),731U);
    environment.Reset();
    InterceptProbe probe{};
    probe.initialError=std::abs(environment.CurrentGuidance().localizer);
    Controls const controls{0,aileron,0,config.initial.throttle,0,false};
    for(std::size_t action=0;action<15;++action)
    {
        auto const step=environment.Step(controls);
        probe.localizerReward+=step.rewards[static_cast<std::size_t>(Reward::Localizer)];
        if(step.terminated || step.truncated) {probe.ended=true;break;}
    }
    probe.finalError=std::abs(environment.CurrentGuidance().localizer);
    return probe;
}

void InterceptActionReward(Config const& config)
{
    auto const neutral=ProbeIntercept(config,0);
    auto const positive=ProbeIntercept(config,0.2);
    auto const negative=ProbeIntercept(config,-0.2);
    auto const& inward=positive.finalError<negative.finalError ? positive : negative;
    auto const& outward=positive.finalError<negative.finalError ? negative : positive;
    Check(neutral.initialError>0 && Near(neutral.initialError,positive.initialError,1e-9)
        && Near(neutral.initialError,negative.initialError,1e-9)
        && !neutral.ended && !positive.ended && !negative.ended
        && !Near(positive.finalError,negative.finalError,1e-8)
        && inward.localizerReward>neutral.localizerReward
        && neutral.localizerReward>outward.localizerReward,
        "real FDM LOC reward prefers the aileron direction that establishes the inward interception");
}

void CloseApproachStarts(Config c)
{
    Check(!c.mission.patrolEnabled && c.initial.distanceMeters.minimum==4000
        && c.initial.distanceMeters.maximum==6500,"delivered starts are closer to the ILS cone");
    c.initial.approach=false; // Disabled patrol must override this legacy initial phase.
    for(auto const stage : {CurriculumStage::Loc,CurriculumStage::Glide,CurriculumStage::Joint,CurriculumStage::JointMobile})
    {
        c.training.curriculumStage=stage;
        FlightEnvironment environment(c,DefaultDataRoot(),981U);
        bool left=false,right=false;
        for(int episode=0;episode<24;++episode)
        {
            auto const observation=environment.Reset();
            auto const guidance=environment.CurrentGuidance();
            auto const& runway=environment.Settings().runway;
            auto const& aircraft=environment.Aircraft();
            double const heading=WrapAngle(aircraft.heading-runway.headingDegrees*DegreesToRadians);
            Check(guidance.distance>=3999.9 && guidance.distance<=6500.1 && guidance.ilsValid
                && std::abs(guidance.localizer)<=10.01*DegreesToRadians
                && std::abs(heading)<=12.1*DegreesToRadians
                && Dot(aircraft.velocity,RunwayForward(runway))>0
                && environment.Plan().count==0 && environment.Mission().phase==Phase::Approach
                && !guidance.targetValid && observation[0]==0 && observation[1]==0 && observation[2]==0,
                "real JSBSim resets remain near the cone, inbound to the moving runway and point-free");
            left=left || guidance.cross< -0.1;right=right || guidance.cross>0.1;
            if(IsCaptureLesson(stage))
            {
                auto const& start=c.training.captureStart;
                auto const local=DeckCoordinates(runway,aircraft.position);
                double const offset=local.up-(runway.thresholdHeightMeters
                    -local.north*std::tan(c.mission.glideDegrees*DegreesToRadians));
                Check(std::abs(aircraft.airspeed-c.mission.approachSpeed)<=start.speedToleranceMps+0.01
                    && std::abs(aircraft.roll)<=start.rollDegrees*DegreesToRadians+1e-8
                    && aircraft.pitch>=start.pitchDegrees.minimum*DegreesToRadians-1e-8
                    && aircraft.pitch<=start.pitchDegrees.maximum*DegreesToRadians+1e-8
                    && environment.Settings().mission.maxSeconds==std::min(c.mission.maxSeconds,c.training.captureMaxSeconds),
                    "native capture lessons start within their speed/attitude envelope and bounded duration");
                if(environment.Settings().training.curriculumStage==CurriculumStage::Loc)
                    Check(std::abs(offset)<=start.localizerHeightOffsetMeters+0.02,
                        "LOC starts avoid an independent large vertical recovery task");
                else Check(std::abs(offset)<=c.training.glideOffsetMeters+0.02
                    && std::abs(offset)>=0.25*c.training.glideOffsetMeters-0.02,
                    "GLIDE lessons retain genuine vertical offsets");
            }
            else Check(environment.Settings().initial.rollDegrees.maximum==c.initial.rollDegrees.maximum
                && environment.Settings().initial.speedMps.maximum==c.initial.speedMps.maximum
                && environment.Settings().mission.maxSeconds==c.mission.maxSeconds,
                "Joint restores the complete initial distribution and landing duration");
            auto const result=environment.Step({0,0,0,c.initial.throttle,0,false});
            Check(environment.Mission().phase==Phase::Approach && !environment.CurrentGuidance().targetValid
                && result.rewards[static_cast<std::size_t>(Reward::Path)]==0,
                "the first physical action cannot reactivate disabled patrol");
        }
        if(stage!=CurriculumStage::Glide) Check(left && right,"closer LOC starts retain both sides");
    }
    c.mission.patrolEnabled=true;
    FlightEnvironment patrol(c,DefaultDataRoot(),982U);
    patrol.Reset();
    Check(patrol.Mission().phase==Phase::Patrol && patrol.Plan().count>1 && patrol.CurrentGuidance().targetValid,
        "explicit reactivation restores the original airborne patrol guidance");
}

void NativeCaptureCompletion(Config c)
{
    c.initial.bearingDegrees={0.5,0.5};c.initial.headingOffsetDegrees={0,0};
    c.mission.captureHoldSeconds=0.05;
    c.training.gateHoldSeconds=0.1;c.training.captureMaxSeconds=4;
    c.training.captureStart.speedToleranceMps=0.1;c.training.captureStart.rollDegrees=0;
    c.training.captureStart.pitchDegrees={2,2};c.training.captureStart.localizerHeightOffsetMeters=0;
    c.training.glideOffsetMeters=1;c.training.locReminderProbability=0;
    for(auto const stage : {CurriculumStage::Loc,CurriculumStage::Glide})
    {
        c.training.curriculumStage=stage;
        FlightEnvironment environment(c,DefaultDataRoot(),773U);
        environment.Reset();StepResult result{};
        for(int step=0;step<40 && environment.Mission().outcome==Outcome::Flying;++step)
            result=environment.Step({0,0,0,c.initial.throttle,0,false});
        Check(result.terminated && !result.truncated && environment.Mission().outcome==Outcome::CaptureCompleted
            && environment.ObservationNow()[41]==1 && !environment.Mission().touchedDown
            && result.rewards[static_cast<std::size_t>(Reward::Stop)]==0,
            "real JSBSim can complete a stabilized capture task without attempting a landing");
        Rejects([&] {environment.Step({});},"native step after capture completion requires reset");
    }
}

void CurriculumRunwayMotion(Config c)
{
    c.runwayRandomization.speedMps={2,2}; // A nonzero source range must not move the first three gates.
    for(auto const stage : {CurriculumStage::Loc,CurriculumStage::Glide,CurriculumStage::Joint,CurriculumStage::JointMobile})
    {
        c.training.curriculumStage=stage;
        FlightEnvironment environment(c,DefaultDataRoot(),92U);
        double const speed=stage==CurriculumStage::JointMobile ? 2.0 : 0.0;
        Runway first{};
        bool positionVaries=false,heightVaries=false;
        for(int episode=0;episode<8;++episode)
        {
            auto const observation=environment.Reset();
            auto const initial=environment.Settings().runway;
            auto const expected=Scale(RunwayForward(initial),speed);
            Check(Near(initial.velocity.north,expected.north) && Near(initial.velocity.east,expected.east)
                && initial.velocity.up==0 && Dot(initial.angularVelocity,initial.angularVelocity)==0,
                "only the fourth JointMobile gate translates; all shipped lessons remain wave-free");
            Check(Near(observation[42],expected.north/c.normalization.speed)
                && Near(observation[43],expected.east/c.normalization.speed) && observation[44]==0,
                "native policy observation receives the actual stage-specific runway velocity");
            auto const relative=environment.CurrentGuidance().relativeVelocity;
            double const course=std::atan2(relative.east,relative.north);
            Check(Near(observation[27],std::sin(course)) && Near(observation[28],std::cos(course)),
                "native JSBSim observations expose actual runway-relative course in all four stages");
            if(episode==0) first=initial;
            positionVaries|=initial.origin.north!=first.origin.north || initial.origin.east!=first.origin.east;
            heightVaries|=initial.origin.up!=first.origin.up;
            for(int step=0;step<10 && environment.Mission().outcome==Outcome::Flying;++step)
                environment.Step({0,0,0,c.initial.throttle,0,false});
            double const elapsed=static_cast<double>(environment.Mission().steps)*c.mission.physicsDt;
            auto const final=environment.Settings().runway;
            Check(elapsed>0 && Near(final.origin.north-initial.origin.north,expected.north*elapsed)
                && Near(final.origin.east-initial.origin.east,expected.east*elapsed)
                && Near(final.origin.up,initial.origin.up),"native steps preserve a fixed runway until JointMobile");
            auto const action=ApplyPhaseControls({0,0,0,0,0,true},Phase::Approach,c.initial,stage);
            Check(action.gearDown==(IsCaptureLesson(stage) ? c.initial.gearDown : true),
                "JointMobile retains learned landing gear rather than capture-only controls");
        }
        Check(positionVaries && heightVaries,"a fixed runway still changes position and altitude between episodes");
    }
}

void IndependentResetAttitude(Config c)
{
    c.initial.distanceMeters={6000,6000};c.initial.speedMps={60,60};
    std::mt19937 rng(183U);
    JsbsimAircraft fdm(c,DefaultDataRoot());
    for(double heading : {0.,90.,181.,359.})
    {
        auto motion=SampleRunway(c,rng);
        motion.base.headingDegrees=heading;
        fdm.SetRunwayMotion(motion);
        auto const runway=RunwayAt(motion,0);
        for(double roll : {-15.,0.,15.}) for(double pitch : {-4.,2.,8.}) for(double slope : {-3.,0.,3.})
        {
            c.initial.rollDegrees={roll,roll};c.initial.pitchDegrees={pitch,pitch};c.initial.slopeDegrees={slope,slope};
            auto const requested=SampleAircraft(c,runway,SpawnPlacement::Left,rng);
            auto const actual=fdm.Reset(requested);
            double const actualSlope=std::atan2(actual.velocity.up,
                std::hypot(actual.velocity.north,actual.velocity.east))/DegreesToRadians;
            Check(Near(actual.roll,requested.roll,1e-8) && Near(actual.pitch,requested.pitch,1e-8)
                && Near(WrapAngle(actual.heading-requested.heading),0,1e-8)
                && Near(actual.airspeed,requested.airspeed,1e-5)
                && std::abs(actualSlope-slope)<0.2,
                "native reset preserves sampled attitude independently of flight-path slope and calibrated speed");
            for(int step=0;step<20;++step) fdm.Advance({0.2,-0.2,0.1,c.initial.throttle,0,true});
            auto const repeated=fdm.Reset(requested);
            Check(Near(repeated.roll,actual.roll,1e-8) && Near(repeated.pitch,actual.pitch,1e-8)
                && Near(WrapAngle(repeated.heading-actual.heading),0,1e-8)
                && Near(repeated.velocity.north,actual.velocity.north,1e-7)
                && Near(repeated.velocity.east,actual.velocity.east,1e-7)
                && Near(repeated.velocity.up,actual.velocity.up,1e-7),
                "a previous maneuver cannot contaminate the next reset attitude or velocity");
        }
    }
    auto invalid=SampleAircraft(c,RunwayAt(SampleRunway(c,rng),0),SpawnPlacement::Left,rng);
    invalid.velocity.up=1000;
    Rejects([&] {fdm.Reset(invalid);},"impossible initial climb must fail explicitly");
}
}

int main()
{
    try
    {
        auto c=LoadConfig(PPO_FLIGHT_CONFIG);
        FiniteSurfaceContactAuthority(c);
        PropulsionAndPhaseReset(c);
        CurriculumRunwayMotion(c);
        IndependentResetAttitude(c);
        NativeCaptureCompletion(c);
        CloseApproachStarts(c);
        PhaseControlMapping(c);
        c.initial.distanceMeters={10000,10000}; c.initial.bearingDegrees={8,8}; c.initial.heightMeters={600,600};
        c.initial.headingOffsetDegrees={0,0}; c.initial.rollDegrees={0,0}; c.initial.pitchDegrees={2,2};
        c.initial.slopeDegrees={0,0}; c.initial.speedMps={65,65};
        c.initial.approach=true;c.initial.runwayAlignedProbability=1;
        auto& d=c.runwayRandomization;
        d.northMeters={0,0}; d.eastMeters={0,0}; d.headingOffsetDegrees={0,0}; d.deckHeightMeters={15,15};
        d.speedMps={0,0}; d.heaveAmplitudeMeters={0,0}; d.pitchAmplitudeDegrees={0,0}; d.rollAmplitudeDegrees={0,0};
        FlightEnvironment env(c);
        auto const obs=env.Reset(); auto const initial=env.Aircraft();
        Check(env.Mission().phase==Phase::Approach,"default real FDM reset prioritizes ILS without a compulsory patrol lap");
        Check(IsFinite(initial) && !initial.weightOnWheels && Near(initial.gearPosition,0,0.01),"initial gear must really be up");
        double const initialHeight=env.Settings().runway.thresholdHeightMeters
            +10000*std::tan(c.mission.glideDegrees*DegreesToRadians);
        auto const point=FromRunway(env.Settings().runway,-10000,0,initialHeight);
        Check(Near(initial.position.up,point.up,0.02),"JSBSim altitude conversion");
        Check(Near(initial.position.north,point.north,0.02) && Near(initial.position.east,point.east,0.02),"JSBSim geodetic conversion");
        for(float value:obs) Check(std::isfinite(value) && std::abs(value)<=1,"JSBSim finite observation");
        auto const action=Controls{0,0,0,c.initial.throttle,0,true};
        for (int i=0; i<60; ++i)
        {
            auto const step=env.Step(action);
            Check(!step.terminated && !step.truncated && IsFinite(env.Aircraft()),"six-second real FDM flight");
        }
        Check(env.Aircraft().gearPosition>=c.mission.gearDownMinimum,"five-second gear travel is physical");
        Check(std::hypot(env.Aircraft().position.north-initial.position.north,env.Aircraft().position.east-initial.position.east)>100,"actual JSBSim dynamics advanced");
        env.Reset();
        Check(Near(env.Aircraft().position.up,initial.position.up) && Near(env.Aircraft().gearPosition,0),"reset clears physical gear and position");
        auto const baseline=env.Step({0,0,0,c.initial.throttle,0,false});
        auto const neutral=env.Aircraft();
        env.Reset(); env.Step({0.5,0.6,0.2,c.initial.throttle,0,false});
        Check(std::abs(env.Aircraft().rollRate-neutral.rollRate)>1e-4 && std::abs(env.Aircraft().pitchRate-neutral.pitchRate)>1e-4,"aileron/elevator affect actual FDM");
        Check(std::isfinite(baseline.reward),"real transition reward");
        InterceptActionReward(c);
        // Curriculum starts must keep LEFT/RIGHT balanced and GLIDE above/below.
        auto locConfig=c;locConfig.training.curriculumStage=CurriculumStage::Loc;
        FlightEnvironment locEnv(locConfig,DefaultDataRoot(),761U);
        bool seenLeft=false,seenRight=false;
        for(int e=0;e<20;++e)
        {
            locEnv.Reset();
            auto const g=locEnv.CurrentGuidance();
            seenLeft|=g.cross<0;seenRight|=g.cross>0;
            Check(locEnv.Settings().training.curriculumStage==CurriculumStage::Loc,"LOC gate active");
        }
        Check(seenLeft && seenRight,"LOC gate explores both sides of runway");
        auto glideConfig=c;glideConfig.training.curriculumStage=CurriculumStage::Glide;
        glideConfig.training.locReminderProbability=0;
        FlightEnvironment glideEnv(glideConfig,DefaultDataRoot(),762U);
        bool high=false,low=false;
        for(int e=0;e<20;++e)
        {
            glideEnv.Reset();
            auto const g=glideEnv.CurrentGuidance();
            high|=g.glideError>0;low|=g.glideError<0;
            Check(std::abs(g.cross)<0.1,"glide gate begins on LOC axis");
        }
        Check(high && low,"glide gate samples both sides of descending plane");
        glideConfig.training.locReminderProbability=1;
        FlightEnvironment reminderEnv(glideConfig,DefaultDataRoot(),763U);
        reminderEnv.Reset();
        Check(reminderEnv.Settings().training.curriculumStage==CurriculumStage::Loc,
            "gate 2 can replay true LOC-only episodes");
        for(int step=0;step<10 && reminderEnv.Mission().outcome==Outcome::Flying;++step)
        {
            auto const reminderStep=reminderEnv.Step(action);
            Check(reminderStep.rewards[static_cast<std::size_t>(Reward::Slope)]==0
                && reminderStep.rewards[static_cast<std::size_t>(Reward::Glide)]==0,
                "native Glide reminders use the LOC reward without any vertical shaping");
        }
        c.mission.maxSeconds=0.2; c.mission.actionSubsteps=1;
        FlightEnvironment shortEpisode(c); shortEpisode.Reset();
        StepResult last{};
        for (int i=0; i<25 && !last.truncated; ++i) last=shortEpisode.Step(action);
        Check(last.truncated && !last.terminated,"actual FDM time limit");
        Rejects([&] { shortEpisode.Step(action); },"step after reset boundary");
        auto random=LoadConfig(PPO_FLIGHT_CONFIG);
        // Keep native moving/tilted contact coverage with waves disabled by default.
        random.runwayRandomization.heaveAmplitudeMeters={0.4,0.8};
        random.runwayRandomization.pitchAmplitudeDegrees={0.1,0.25};
        random.runwayRandomization.rollAmplitudeDegrees={0.2,0.4};
        random.initial.runwayAlignedProbability=1;
        FlightEnvironment randomized(random);
        for(int episode=0;episode<32;++episode)
        {
            randomized.Reset();
            auto const local=DeckCoordinates(randomized.Settings().runway,randomized.Aircraft().position);
            double const expectedHeight=randomized.Settings().runway.thresholdHeightMeters
                -local.north*std::tan(random.mission.glideDegrees*DegreesToRadians);
            Check(Near(local.up,expectedHeight,0.1),"real FDM reset preserves the configured glide height");
            Check(local.north<0 && Near(local.east,0,0.02)
                && Near(WrapAngle(randomized.Aircraft().heading-randomized.Settings().runway.headingDegrees*DegreesToRadians),0,1e-6),
                "native JSBSim reset preserves runway centreline and inbound heading on moving tilted runways");
            Check(randomized.Aircraft().airspeed>=random.initial.speedMps.minimum-0.1 && randomized.Aircraft().airspeed<=random.initial.speedMps.maximum+0.1,"real FDM randomized calibrated speed");
        }
        JsbsimAircraft fdm(random,DefaultDataRoot());
        std::mt19937 generator(77);
        SpawnPlacementSampler placements{};
        auto const motion=SampleRunway(random,generator);
        auto const deck=RunwayAt(motion,0);
        fdm.SetRunwayMotion(motion);
        auto const placement=placements.Next(random.initial.runwayAlignedProbability,generator);
        fdm.Reset(SampleAircraft(random,deck,placement,generator));
        auto const probe=fdm.ProbeGround(FromRunway(deck,500,0,10));
        auto const expected=RunwayNormal(deck),velocity=DeckVelocity(deck,FromRunway(deck,500,0,0));
        Check(Near(probe.height,10,1e-6) && Near(probe.normal.north,expected.north,1e-6)
            && Near(probe.normal.up,expected.up,1e-6),"JSBSim uses the tilted moving contact plane");
        Check(Near(probe.velocity.north,velocity.north,1e-6) && Near(probe.velocity.up,velocity.up,1e-6),"JSBSim receives moving surface velocity");
        auto const offDeck=FromRunway(deck,500,random.runway.widthMeters+100,10);
        Check(Near(fdm.ProbeGround(offDeck).height,offDeck.up-motion.seaHeight,1e-6),"finite deck falls back to sea outside runway");
        random.initial.gearDown=true;random.initial.throttle=0;
        JsbsimAircraft touchdownFdm(random,DefaultDataRoot());
        touchdownFdm.SetRunwayMotion(motion);
        AircraftState descending{};
        descending.position=FromRunway(deck,500,0,2.5);descending.airspeed=3;
        descending.heading=deck.headingDegrees*DegreesToRadians;
        descending.pitch=deck.pitch;descending.roll=deck.roll;descending.velocity.up=-1;
        auto contact=touchdownFdm.Reset(descending);
        bool contacted=contact.weightOnWheels;
        for(int i=0;i<300 && !contacted;++i)
        {
            contact=touchdownFdm.Advance({0,0,0,0,1,true});
            Check(IsFinite(contact),"finite native moving-deck contact dynamics");
            contacted=contact.weightOnWheels;
        }
        Check(contacted && contact.position.up>motion.seaHeight+5,"real wheels contact the moving elevated runway");
    }
    catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 1; }
    std::cout<<"Native JSBSim regressions passed\n";
}
