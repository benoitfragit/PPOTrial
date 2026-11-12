#include "test_support.hpp"
using namespace ppo::examples::flight;
int main()
{
    try
    {
        auto c=LoadConfig(PPO_FLIGHT_CONFIG);
        c.initial.distanceMeters={10000,10000}; c.initial.bearingDegrees={0,0}; c.initial.heightMeters={600,600};
        c.initial.headingOffsetDegrees={0,0}; c.initial.rollDegrees={0,0}; c.initial.pitchDegrees={2,2};
        c.initial.slopeDegrees={0,0}; c.initial.speedMps={65,65};
        auto& d=c.runwayRandomization;
        d.northMeters={0,0}; d.eastMeters={0,0}; d.headingOffsetDegrees={0,0}; d.deckHeightMeters={15,15};
        d.speedMps={0,0}; d.heaveAmplitudeMeters={0,0}; d.pitchAmplitudeDegrees={0,0}; d.rollAmplitudeDegrees={0,0};
        FlightEnvironment env(c);
        auto const obs=env.Reset(); auto const initial=env.Aircraft();
        Check(env.Mission().phase==Phase::Approach,"default real FDM reset prioritizes ILS without a compulsory patrol lap");
        Check(IsFinite(initial) && !initial.weightOnWheels && Near(initial.gearPosition,0,0.01),"initial gear must really be up");
        Check(Near(initial.position.up,615,0.02),"JSBSim altitude conversion");
        auto const point=FromRunway(env.Settings().runway,-10000,0,600);
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
        c.mission.maxSeconds=0.2; c.mission.actionSubsteps=1;
        FlightEnvironment shortEpisode(c); shortEpisode.Reset();
        StepResult last{};
        for (int i=0; i<25 && !last.truncated; ++i) last=shortEpisode.Step(action);
        Check(last.truncated && !last.terminated,"actual FDM time limit");
        Rejects([&] { shortEpisode.Step(action); },"step after reset boundary");
        auto random=LoadConfig(PPO_FLIGHT_CONFIG);
        random.initial.runwayAlignedProbability=1;
        FlightEnvironment randomized(random);
        for(int episode=0;episode<32;++episode)
        {
            randomized.Reset();
            auto const local=DeckCoordinates(randomized.Settings().runway,randomized.Aircraft().position);
            Check(local.up>=random.initial.heightMeters.minimum-0.1 && local.up<=random.initial.heightMeters.maximum+0.1,"real FDM randomized airborne height");
            Check(local.north<0 && Near(local.east,0,0.02)
                && Near(WrapAngle(randomized.Aircraft().heading-randomized.Settings().runway.headingDegrees*DegreesToRadians),0,1e-6),
                "native JSBSim reset preserves runway centreline and inbound heading on moving tilted runways");
            Check(randomized.Aircraft().airspeed>=random.initial.speedMps.minimum-0.1 && randomized.Aircraft().airspeed<=random.initial.speedMps.maximum+0.1,"real FDM randomized calibrated speed");
        }
        JsbsimAircraft fdm(random,DefaultDataRoot());
        std::mt19937 generator(77);
        auto const motion=SampleRunway(random,generator);
        auto const deck=RunwayAt(motion,0);
        fdm.SetRunwayMotion(motion);
        fdm.Reset(SampleAircraft(random,deck,generator));
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
