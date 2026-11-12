#include "runway_motion.hpp"

namespace ppo::examples::flight {
Vec3 Add(Vec3 a, Vec3 b) noexcept { return {a.north+b.north,a.east+b.east,a.up+b.up}; }
Vec3 Subtract(Vec3 a, Vec3 b) noexcept { return {a.north-b.north,a.east-b.east,a.up-b.up}; }
Vec3 Scale(Vec3 a, double scale) noexcept { return {a.north*scale,a.east*scale,a.up*scale}; }
double Dot(Vec3 a, Vec3 b) noexcept { return a.north*b.north+a.east*b.east+a.up*b.up; }
Vec3 Cross(Vec3 a, Vec3 b) noexcept
{ return {a.east*b.up-a.up*b.east,a.up*b.north-a.north*b.up,a.north*b.east-a.east*b.north}; }
Vec3 RunwayForward(Runway const& r) noexcept
{
    double const h=r.headingDegrees*DegreesToRadians;
    return {std::cos(r.pitch)*std::cos(h),std::cos(r.pitch)*std::sin(h),std::sin(r.pitch)};
}
Vec3 RunwayRight(Runway const& r) noexcept
{
    double const h=r.headingDegrees*DegreesToRadians;
    Vec3 const right{-std::sin(h),std::cos(h),0};
    Vec3 const up{-std::sin(r.pitch)*std::cos(h),-std::sin(r.pitch)*std::sin(h),std::cos(r.pitch)};
    return Subtract(Scale(right,std::cos(r.roll)),Scale(up,std::sin(r.roll)));
}
Vec3 RunwayNormal(Runway const& r) noexcept { return Cross(RunwayForward(r),RunwayRight(r)); }
Vec3 DeckCoordinates(Runway const& r, Vec3 point) noexcept
{
    auto const delta=Subtract(point,r.origin);
    return {Dot(delta,RunwayForward(r)),Dot(delta,RunwayRight(r)),Dot(delta,RunwayNormal(r))};
}
Vec3 DeckVelocity(Runway const& r, Vec3 point) noexcept
{ return Add(r.velocity,Cross(r.angularVelocity,Subtract(point,r.origin))); }

Runway RunwayAt(RunwayMotion const& motion, double seconds) noexcept
{
    auto r=motion.base;
    double const omega=2*Pi/motion.period;
    double const heave=motion.heaveAmplitude*std::sin(omega*seconds+motion.heavePhase);
    double const heaveRate=omega*motion.heaveAmplitude*std::cos(omega*seconds+motion.heavePhase);
    r.pitch=motion.pitchAmplitude*std::sin(omega*seconds+motion.pitchPhase);
    r.roll=motion.rollAmplitude*std::sin(omega*seconds+motion.rollPhase);
    double const pitchRate=omega*motion.pitchAmplitude*std::cos(omega*seconds+motion.pitchPhase);
    double const rollRate=omega*motion.rollAmplitude*std::cos(omega*seconds+motion.rollPhase);
    auto const forward=RunwayForward(motion.base);
    Vec3 const right{-forward.east,forward.north,0};
    r.angularVelocity=Add(Scale(right,-pitchRate),Scale(RunwayForward(r),-rollRate));
    // Pitch/roll pivot at runway centre, not at the threshold.
    auto centre=Add(motion.base.origin,Scale(forward,r.lengthMeters/2+motion.forwardSpeed*seconds));
    centre.up+=heave;
    r.origin=Subtract(centre,Scale(RunwayForward(r),r.lengthMeters/2));
    Vec3 centreVelocity=Scale(forward,motion.forwardSpeed); centreVelocity.up+=heaveRate;
    r.velocity=Subtract(centreVelocity,Cross(r.angularVelocity,Scale(RunwayForward(r),r.lengthMeters/2)));
    return r;
}

namespace {
double Draw(Range range, std::mt19937& rng)
{ return std::uniform_real_distribution<double>(range.minimum,range.maximum)(rng); }
}
RunwayMotion SampleRunway(Config const& c, std::mt19937& rng)
{
    auto const& d=c.runwayRandomization;
    RunwayMotion motion{};
    motion.base=c.runway;
    motion.base.origin={Draw(d.northMeters,rng),Draw(d.eastMeters,rng),Draw(d.deckHeightMeters,rng)};
    motion.base.headingDegrees=std::fmod(c.runway.headingDegrees+Draw(d.headingOffsetDegrees,rng)+360,360);
    motion.forwardSpeed=Draw(d.speedMps,rng);
    motion.heaveAmplitude=Draw(d.heaveAmplitudeMeters,rng);
    motion.period=Draw(d.wavePeriodSeconds,rng);
    motion.pitchAmplitude=Draw(d.pitchAmplitudeDegrees,rng)*DegreesToRadians;
    motion.rollAmplitude=Draw(d.rollAmplitudeDegrees,rng)*DegreesToRadians;
    motion.heavePhase=Draw(d.phaseDegrees,rng)*DegreesToRadians;
    motion.pitchPhase=Draw(d.phaseDegrees,rng)*DegreesToRadians;
    motion.rollPhase=Draw(d.phaseDegrees,rng)*DegreesToRadians;
    motion.seaHeight=d.seaHeightMeters;
    return motion;
}
AircraftState SampleAircraft(Config const& c, Runway const& runway, std::mt19937& rng)
{
    auto const& i=c.initial;
    double const distance=Draw(i.distanceMeters,rng),bearing=Draw(i.bearingDegrees,rng)*DegreesToRadians;
    double const height=Draw(i.heightMeters,rng);
    AircraftState aircraft{};
    auto const local=Add(Scale(RunwayForward(runway),-distance*std::cos(bearing)),Scale(RunwayRight(runway),distance*std::sin(bearing)));
    aircraft.position=Add(runway.origin,local);
    // Height is above the instantaneous runway tangent plane.
    aircraft.position=Add(aircraft.position,Scale(RunwayNormal(runway),height));
    aircraft.airspeed=Draw(i.speedMps,rng);
    aircraft.heading=WrapAngle((runway.headingDegrees+Draw(i.headingOffsetDegrees,rng))*DegreesToRadians);
    aircraft.roll=Draw(i.rollDegrees,rng)*DegreesToRadians;
    aircraft.pitch=Draw(i.pitchDegrees,rng)*DegreesToRadians;
    double const slope=Draw(i.slopeDegrees,rng)*DegreesToRadians;
    aircraft.velocity={aircraft.airspeed*std::cos(slope)*std::cos(aircraft.heading),
        aircraft.airspeed*std::cos(slope)*std::sin(aircraft.heading),aircraft.airspeed*std::sin(slope)};
    return aircraft;
}
} // namespace ppo::examples::flight
