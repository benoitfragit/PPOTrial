#include "flight_mission.hpp"
#include "runway_motion.hpp"
#include <algorithm>
#include <limits>

namespace ppo::examples::flight {
namespace {
double Distance(Vec3 const& a, Vec3 const& b) noexcept
{
    return std::hypot(std::hypot(a.north-b.north, a.east-b.east), a.up-b.up);
}
Vec3 RelativeVelocity(Runway const& r, AircraftState const& state) noexcept
{
    auto const velocity=Subtract(state.velocity,DeckVelocity(r,state.position));
    return {Dot(velocity,RunwayForward(r)),Dot(velocity,RunwayRight(r)),Dot(velocity,RunwayNormal(r))};
}
double Quality(double error, double tolerance) noexcept
{
    return std::max(0.0, 1.0-std::abs(error)/tolerance);
}
float Unit(double value) noexcept
{
    return std::isfinite(value) ? static_cast<float>(std::clamp(value, -1.0, 1.0)) : 0.F;
}
bool OnRunway(Config const& c, Guidance const& g) noexcept
{
    return g.along >= 0 && g.along <= c.runway.lengthMeters && std::abs(g.cross) <= c.runway.widthMeters/2;
}
std::array<double,3> RelativeAttitude(Runway const& deck,AircraftState const& a) noexcept
{
    Runway pose{};pose.headingDegrees=a.heading/DegreesToRadians;pose.roll=a.roll;pose.pitch=a.pitch;
    auto const forward=RunwayForward(pose),normal=RunwayNormal(deck);
    return {std::atan2(-Dot(RunwayRight(pose),normal),Dot(RunwayNormal(pose),normal)),
        std::asin(std::clamp(Dot(forward,normal),-1.0,1.0)),
        std::atan2(Dot(forward,RunwayRight(deck)),Dot(forward,RunwayForward(deck)))};
}
bool LandingSafe(Config const& c, Runway const& previous, AircraftState const& before, AircraftState const& after) noexcept
{
    auto const& m = c.mission;
    auto const bv=RelativeVelocity(previous,before),av=RelativeVelocity(c.runway,after);
    auto const ba=RelativeAttitude(previous,before),aa=RelativeAttitude(c.runway,after);
    return after.gearPosition >= m.gearDownMinimum && !after.structuralContact
        && std::max(-bv.up,-av.up) <= m.maxTouchdownSink
        && std::max(std::hypot(bv.north,bv.east),std::hypot(av.north,av.east)) <= m.maxTouchdownSpeed
        && std::max(std::abs(ba[0]),std::abs(aa[0])) <= m.maxTouchdownRollDegrees*DegreesToRadians
        && std::min(ba[1],aa[1]) >= m.minTouchdownPitchDegrees*DegreesToRadians
        && std::max(ba[1],aa[1]) <= m.maxTouchdownPitchDegrees*DegreesToRadians
        && std::max(std::abs(ba[2]),std::abs(aa[2])) <= m.maxTouchdownHeadingDegrees*DegreesToRadians;
}
} // namespace

Vec3 FromRunway(Runway const& r, double along, double cross, double up) noexcept
{
    return Add(r.origin,Add(Scale(RunwayForward(r),along),Add(Scale(RunwayRight(r),cross),Scale(RunwayNormal(r),up))));
}
std::array<double,2> ToRunway(Runway const& r, Vec3 const& point) noexcept
{
    auto const local=DeckCoordinates(r,point);
    return {local.north,local.east};
}

FlightPlan MakeFlightPlan(Config const& c) noexcept
{
    FlightPlan plan{};
    auto const& m = c.mission;
    // One closed counter-clockwise stadium, bounded by validated capacity.
    for (std::size_t side = 0; side < 2; ++side)
    {
        double const sign = side == 0 ? 1.0 : -1.0;
        for (std::size_t j = 0; j < m.straightSamples; ++j)
        {
            double const fraction = static_cast<double>(j)/static_cast<double>(m.straightSamples);
            plan.points[plan.count++] = FromRunway(c.runway,
                m.racetrackCenterAlong + sign*m.racetrackHalfStraight*(2*fraction-1),
                m.racetrackCenterCross-sign*m.racetrackRadius, m.patrolAltitude);
        }
        for (std::size_t j = 0; j < m.turnSamples; ++j)
        {
            double const angle = -Pi/2 + static_cast<double>(side)*Pi + Pi*static_cast<double>(j)/static_cast<double>(m.turnSamples);
            plan.points[plan.count++] = FromRunway(c.runway,
                m.racetrackCenterAlong + sign*m.racetrackHalfStraight + m.racetrackRadius*std::cos(angle),
                m.racetrackCenterCross+m.racetrackRadius*std::sin(angle), m.patrolAltitude);
        }
    }
    return plan;
}

Guidance Guide(Config const& c, FlightPlan const& plan, MissionState const& mission, AircraftState const& a) noexcept
{
    Guidance g{};
    auto const runway = ToRunway(c.runway,a.position);
    g.along = runway[0]; g.cross = runway[1];
    g.height=DeckCoordinates(c.runway,a.position).up;
    g.relativeVelocity=RelativeVelocity(c.runway,a);
    g.distance = std::hypot(g.along,g.cross);
    g.localizer = std::atan2(g.cross, -g.along);
    double const slope = c.mission.glideDegrees*DegreesToRadians;
    g.glideError = std::atan2(g.height-c.runway.thresholdHeightMeters, -g.along)-slope;
    g.ilsValid = g.along < 0 && g.distance <= c.mission.ilsRangeMeters && g.height > 0
        && std::abs(g.localizer) <= c.mission.ilsSectorDegrees*DegreesToRadians;
    g.targetSpeed = c.mission.patrolSpeed;
    if (mission.phase == Phase::Patrol && plan.count > 1)
    {
        std::size_t const next = mission.waypoint % plan.count;
        g.target = plan.points[next];
        auto const& previous = plan.points[(next+plan.count-1)%plan.count];
        double const dn = g.target.north-previous.north, de = g.target.east-previous.east;
        double const lengthSquared = dn*dn+de*de;
        double const t = lengthSquared > 0 ? std::clamp(((a.position.north-previous.north)*dn
            +(a.position.east-previous.east)*de)/lengthSquared, 0.0, 1.0) : 0;
        g.pathDistance = Distance(a.position, {previous.north+t*dn, previous.east+t*de, previous.up+t*(g.target.up-previous.up)});
    }
    else
    {
        bool const flare = g.height <= c.mission.flareHeightMeters;
        g.targetSpeed = mission.phase == Phase::Rollout ? 0 : (flare ? c.mission.flareSpeed : c.mission.approachSpeed);
        g.targetSlope = flare || mission.phase == Phase::Rollout ? 0 : -slope;
        g.target = FromRunway(c.runway, 0, 0, c.runway.thresholdHeightMeters);
        g.pathDistance = std::hypot(g.cross, g.height-(c.runway.thresholdHeightMeters-std::min(g.along,0.0)*std::tan(slope)));
    }
    g.courseError = WrapAngle(std::atan2(g.target.east-a.position.east,g.target.north-a.position.north)-a.heading);
    return g;
}

StepResult AdvanceMission(Config const& c, FlightPlan const& plan, MissionState& s,
    AircraftState const& before, AircraftState const& after, Runway const* previousRunway) noexcept
{
    StepResult result{};
    if (s.outcome != Outcome::Flying)
    {
        result.terminated = s.outcome != Outcome::Timeout;
        result.truncated = s.outcome == Outcome::Timeout;
        return result;
    }
    ++s.steps;
    auto g = Guide(c,plan,s,after);
    if (s.phase == Phase::Patrol && g.distance <= c.mission.switchDistanceMeters && g.along < 0)
        s.phase = Phase::Approach;
    if (s.phase == Phase::Patrol && plan.count > 1 && Distance(after.position,plan.points[s.waypoint%plan.count]) <= c.mission.waypointRadius)
        s.waypoint = (s.waypoint+1)%plan.count;
    g = Guide(c,plan,s,after);
    auto const& previousDeck=previousRunway ? *previousRunway : c.runway;
    bool const contact = after.weightOnWheels || after.structuralContact || g.height <= c.mission.contactHeightMeters;
    bool crash = !IsFinite(before) || !IsFinite(after) || g.distance > c.mission.maxDistanceMeters
        || g.height > c.mission.maxAltitudeMeters;
    if (contact)
        crash = crash || s.phase == Phase::Patrol || !OnRunway(c,g)
            || !LandingSafe(c,previousDeck,before,after) || !after.weightOnWheels;
    if (s.phase == Phase::Rollout && !OnRunway(c,g)) crash = true;
    if (crash)
    {
        s.outcome = Outcome::Crash;
        result.rewards[static_cast<std::size_t>(Reward::Crash)] = c.reward.weights[static_cast<std::size_t>(Reward::Crash)];
    }
    else
    {
        auto add = [&](Reward channel, double value)
        {
            auto const i = static_cast<std::size_t>(channel);
            result.rewards[i] = c.reward.weights[i]*value;
        };
        double const dt = c.mission.physicsDt;
        if (s.phase == Phase::Patrol)
            add(Reward::Path,dt*Quality(g.pathDistance,c.reward.pathTolerance));
        if (s.phase == Phase::Approach && g.ilsValid)
        {
            add(Reward::Localizer,dt*Quality(g.localizer,c.reward.localizerToleranceDegrees*DegreesToRadians));
            add(Reward::Glide,dt*Quality(g.glideError,c.reward.glideToleranceDegrees*DegreesToRadians));
            if (!s.capturedLocalizer && std::abs(g.localizer) <= c.mission.localizerCaptureDegrees*DegreesToRadians)
            { s.capturedLocalizer = true; add(Reward::EnterLocalizer,1); }
            if (!s.capturedGlide && std::abs(g.glideError) <= c.mission.glideCaptureDegrees*DegreesToRadians)
            { s.capturedGlide = true; add(Reward::EnterGlide,1); }
        }
        if (s.phase != Phase::Rollout)
        {
            add(Reward::Speed,dt*Quality(after.airspeed-g.targetSpeed,c.reward.speedTolerance));
            double const relativeSlope=std::atan2(g.relativeVelocity.up,std::hypot(g.relativeVelocity.north,g.relativeVelocity.east));
            add(Reward::Slope,dt*Quality(relativeSlope-g.targetSlope,c.reward.slopeToleranceDegrees*DegreesToRadians));
        }
        if (contact && !s.touchedDown)
        {
            s.touchedDown = true;
            s.touchdownFromThreshold = g.along;
            s.phase = Phase::Rollout;
            add(Reward::Touchdown,1);
        }
        if (s.phase == Phase::Rollout)
        {
            auto const previousPosition=DeckCoordinates(previousDeck,before.position);
            s.groundRoll += std::hypot(g.along-previousPosition.north,g.cross-previousPosition.east);
            bool const stopped = after.weightOnWheels && std::hypot(g.relativeVelocity.north,g.relativeVelocity.east) <= c.mission.stoppedSpeed;
            s.stoppedSeconds = stopped ? s.stoppedSeconds+dt : 0;
            if (s.stoppedSeconds >= c.mission.stoppedHoldSeconds)
            {
                s.stopFromThreshold = g.along;
                s.outcome = Outcome::Stopped;
                add(Reward::Stop,1);
            }
        }
        if (s.outcome == Outcome::Flying && static_cast<double>(s.steps)*dt >= c.mission.maxSeconds)
            s.outcome = Outcome::Timeout;
    }
    for (double reward : result.rewards) result.reward += reward;
    result.terminated = s.outcome == Outcome::Crash || s.outcome == Outcome::Stopped;
    result.truncated = s.outcome == Outcome::Timeout;
    return result;
}

Observation Observe(Config const& c, FlightPlan const& plan, MissionState const& s, AircraftState const& a) noexcept
{
    if (!IsFinite(a)) return {}; // Terminal invalid FDM observations never propagate NaNs to PPO.
    auto const g = Guide(c,plan,s,a);
    auto const& n = c.normalization;
    double const dn = g.target.north-a.position.north, de = g.target.east-a.position.east;
    double const forward = dn*std::cos(a.heading)+de*std::sin(a.heading);
    double const right = -dn*std::sin(a.heading)+de*std::cos(a.heading);
    double const heading = WrapAngle(a.heading-c.runway.headingDegrees*DegreesToRadians);
    double const slope = std::atan2(g.relativeVelocity.up,std::hypot(g.relativeVelocity.north,g.relativeVelocity.east));
    auto const deckVelocity=DeckVelocity(c.runway,a.position);
    auto const normal=RunwayNormal(c.runway);
    return {Unit(forward/n.distance), Unit(right/n.distance), Unit((g.target.up-a.position.up)/n.altitude),
        Unit(g.along/n.distance), Unit(g.cross/n.distance), Unit(g.distance/n.distance), Unit(g.height/n.altitude),
        Unit(a.airspeed/n.speed), Unit(a.velocity.north/n.speed), Unit(a.velocity.east/n.speed), Unit(a.velocity.up/n.speed),
        Unit(std::sin(a.roll)), Unit(std::cos(a.roll)), Unit(std::sin(a.pitch)), Unit(std::cos(a.pitch)),
        Unit(std::sin(heading)), Unit(std::cos(heading)), Unit(a.rollRate/n.rate), Unit(a.pitchRate/n.rate), Unit(a.yawRate/n.rate),
        Unit(std::sin(slope)), Unit(std::cos(slope)), Unit(g.ilsValid ? g.localizer/(n.localizerDegrees*DegreesToRadians) : 0),
        Unit(g.ilsValid ? g.glideError/(n.glideDegrees*DegreesToRadians) : 0), Unit(g.ilsValid), Unit(g.targetSpeed/n.speed),
        Unit(g.targetSlope/Pi), Unit(std::sin(g.courseError)), Unit(std::cos(g.courseError)), Unit(g.pathDistance/n.path),
        Unit(a.gearPosition), Unit(a.weightOnWheels), Unit(s.phase==Phase::Patrol), Unit(s.phase==Phase::Approach), Unit(s.phase==Phase::Rollout),
        Unit(static_cast<double>(s.steps)*c.mission.physicsDt/c.mission.maxSeconds), Unit(s.capturedLocalizer), Unit(s.capturedGlide),
        Unit(s.touchedDown), Unit(s.touchdownFromThreshold/n.distance), Unit((c.runway.lengthMeters-g.along)/n.distance),
        Unit(s.stoppedSeconds/c.mission.stoppedHoldSeconds),Unit(deckVelocity.north/n.speed),Unit(deckVelocity.east/n.speed),Unit(deckVelocity.up/n.speed),
        Unit(normal.north),Unit(normal.east),Unit(normal.up),Unit(c.runway.angularVelocity.north/n.rate),
        Unit(c.runway.angularVelocity.east/n.rate),Unit(c.runway.angularVelocity.up/n.rate)};
}
} // namespace ppo::examples::flight
