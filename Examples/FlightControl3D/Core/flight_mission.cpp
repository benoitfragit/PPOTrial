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
double GuidanceQuality(double error, double tolerance) noexcept
{
    // A finite error outside the capture tolerance must still provide guidance.
    return tolerance/(tolerance+std::abs(error));
}
double GuidanceReward(Config const& c,double previousQuality,double quality,double progress,double sustainScale=1.0) noexcept
{
    // The signed difference cancels on a closed error excursion. Do not clip it:
    // clipping would allow slow improvement / fast retreat to farm convergence.
    return c.mission.physicsDt*sustainScale*quality*progress+c.reward.convergenceSeconds*(quality-previousQuality);
}
AircraftState InCurrentDeckFrame(Runway const& previous,Runway const& current,AircraftState state) noexcept
{
    auto const local=DeckCoordinates(previous,state.position);
    state.position=FromRunway(current,local.north,local.east,local.up);
    return state;
}
double SegmentDistance(Vec3 point,Vec3 start,Vec3 end) noexcept
{
    auto const direction=Subtract(end,start);
    double const square=Dot(direction,direction);
    double const fraction=square>0 ? std::clamp(Dot(Subtract(point,start),direction)/square,0.0,1.0) : 0;
    return Distance(point,Add(start,Scale(direction,fraction)));
}

struct PlanProjection final
{
    Vec3 point{};
    std::size_t segmentEnd{};
    double distance{std::numeric_limits<double>::max()};
};

/**
 * @brief Projects a position onto the closest segment of a closed flight plan.
 * @param plan Bounded closed flight plan.
 * @param position World position to project.
 * @return Closest point, segment endpoint index and three-dimensional distance.
 */
PlanProjection ProjectOnPlan(FlightPlan const& plan,Vec3 const& position) noexcept
{
    PlanProjection best{};
    if(plan.count==0) { best.point=position;best.distance=0;return best; }
    for(std::size_t endIndex=0;endIndex<plan.count;++endIndex)
    {
        auto const& start=plan.points[(endIndex+plan.count-1)%plan.count];
        auto const& end=plan.points[endIndex];
        auto const direction=Subtract(end,start);
        double const square=Dot(direction,direction);
        double const fraction=square>0
            ? std::clamp(Dot(Subtract(position,start),direction)/square,0.0,1.0) : 0;
        auto const point=Add(start,Scale(direction,fraction));
        double const distance=Distance(position,point);
        if(distance<best.distance)
        {
            best={point,endIndex,distance};
        }
    }
    return best;
}

/**
 * @brief Advances a bounded lookahead from a projection along a closed flight plan.
 * @param plan Bounded closed flight plan.
 * @param projection Valid projection on one of the plan segments.
 * @param lookahead Non-negative distance to advance along the loop.
 * @return World point reached after wrapping at most once around the circuit.
 */
Vec3 LookAheadOnPlan(FlightPlan const& plan,PlanProjection const& projection,double lookahead) noexcept
{
    if(plan.count==0) return projection.point;
    double circuit{};
    for(std::size_t endIndex=0;endIndex<plan.count;++endIndex)
        circuit+=Distance(plan.points[(endIndex+plan.count-1)%plan.count],plan.points[endIndex]);
    if(circuit<=0) return projection.point;
    double remaining=std::fmod(std::max(lookahead,0.0),circuit);
    auto start=projection.point;
    std::size_t endIndex=projection.segmentEnd;
    for(std::size_t segment=0;segment<=plan.count;++segment)
    {
        auto const& end=plan.points[endIndex];
        double const length=Distance(start,end);
        if(length>0 && remaining<=length)
            return Add(start,Scale(Subtract(end,start),remaining/length));
        remaining-=length;
        start=end;
        endIndex=(endIndex+1)%plan.count;
    }
    return start;
}

double AdvancePatrol(Config const& c,FlightPlan const& plan,MissionState& s,
    AircraftState const& before,AircraftState const& after,Runway const& previousDeck) noexcept
{
    if(s.phase!=Phase::Patrol || plan.count<2) return 0;
    auto const start=DeckCoordinates(c.runway,plan.points[(s.waypoint+plan.count-1)%plan.count]);
    auto const end=DeckCoordinates(c.runway,plan.points[s.waypoint%plan.count]);
    auto const old=DeckCoordinates(previousDeck,before.position),current=DeckCoordinates(c.runway,after.position);
    auto const movement=Subtract(current,old),segment=Subtract(end,start);
    double const length=std::sqrt(Dot(segment,segment));
    bool const qualified=length>0 && std::sqrt(Dot(movement,movement))<=c.mission.maximumWaypointStepMeters
        && SegmentDistance(old,start,end)<=c.mission.pathCorridorMeters && SegmentDistance(current,start,end)<=c.mission.pathCorridorMeters;
    if(!qualified)
    {
        s.pathQualified=false;s.waypointsPassed=0;
        s.waypoint=ClosestPatrolWaypoint(plan,after.position);
        return 0;
    }
    double const progress=Dot(movement,segment)/length;
    if(progress>0 && SegmentDistance(end,old,current)<=c.mission.waypointRadius)
    {
        if(s.pathQualified) ++s.waypointsPassed;
        else { s.pathQualified=true;s.waypointsPassed=0; }
        if(s.waypointsPassed>=plan.count) { ++s.completedLaps;s.waypointsPassed=0; }
        s.waypoint=(s.waypoint+1)%plan.count;
    }
    else if(Dot(Subtract(current,start),segment)/length>length+c.mission.waypointRadius)
    {
        // A missed gate must not leave the aircraft chasing a point behind it.
        // Reacquire guidance, but discard all credit for the incomplete lap.
        s.pathQualified=false;s.waypointsPassed=0;
        auto const closest=ClosestPatrolWaypoint(plan,after.position);
        s.waypoint=closest==s.waypoint ? (s.waypoint+1)%plan.count : closest;
        return 0;
    }
    return std::max(progress,0.0);
}
/** @brief Capture edges confirmed by the acquisition/loss state machine. */
struct CaptureTransition final { bool entered{},lost{}; };

/**
 * @brief Updates one reversible beam capture with acquisition/loss hysteresis.
 * @param eligible True only for finite airborne front-sector geometry and a compatible heading.
 * @param error Signed angular beam error in radians.
 * @param tolerance Capture tolerance in radians.
 * @param c Validated mission timing and hysteresis settings.
 * @param acquisitionSeconds In/out continuous acquisition hold duration.
 * @param lossSeconds In/out continuous release hold duration.
 * @param captured In/out active capture flag.
 * @return Entry and loss transitions confirmed during this step.
 */
CaptureTransition UpdateCapture(bool eligible,double error,double tolerance,Config const& c,
    double& acquisitionSeconds,double& lossSeconds,bool& captured) noexcept
{
    double const absoluteError=std::abs(error);
    double const hold=c.mission.captureHoldSeconds;
    if(captured)
    {
        acquisitionSeconds=hold;
        bool const outsideRelease=!eligible || absoluteError>tolerance*c.mission.captureHysteresis;
        lossSeconds=outsideRelease ? std::min(lossSeconds+c.mission.physicsDt,hold) : 0;
        if(lossSeconds>=hold)
        {
            captured=false;
            acquisitionSeconds=0;
            lossSeconds=0;
            return {false,true};
        }
        return {};
    }
    lossSeconds=0;
    if(!eligible) { acquisitionSeconds=0;return {}; }
    if(absoluteError<=tolerance)
        acquisitionSeconds=std::min(acquisitionSeconds+c.mission.physicsDt,hold);
    else if(absoluteError>tolerance*c.mission.captureHysteresis)
        acquisitionSeconds=0;
    if(acquisitionSeconds>=hold)
    {
        captured=true;
        return {true,false};
    }
    return {};
}
/**
 * @brief Returns navigation to racetrack reacquisition after a confirmed ILS loss.
 * @param plan Bounded closed racetrack used to select the closest re-entry segment.
 * @param state Mission state whose captures and partial lap are invalidated.
 * @param aircraft Current aircraft state used for the racetrack projection seed.
 */
void ReturnToPatrol(FlightPlan const& plan,MissionState& state,AircraftState const& aircraft) noexcept
{
    state.phase=Phase::Patrol;
    state.localizerHoldSeconds=0;
    state.glideHoldSeconds=0;
    state.localizerLossSeconds=0;
    state.glideLossSeconds=0;
    state.capturedLocalizer=false;
    state.capturedGlide=false;
    state.pathQualified=false;
    state.waypointsPassed=0;
    state.waypoint=ClosestPatrolWaypoint(plan,aircraft.position);
}
float Unit(double value) noexcept
{
    return std::isfinite(value) ? static_cast<float>(std::clamp(value, -1.0, 1.0)) : 0.F;
}
bool OnRunway(Config const& c, Guidance const& g) noexcept
{
    return g.along >= 0 && g.along <= c.runway.lengthMeters && std::abs(g.cross) <= c.runway.widthMeters/2;
}
bool WheelsInside(Config const& c,AircraftState const& a) noexcept
{
    if(!a.wheelDataValid) return true;
    for(auto const& wheel:a.wheels)
    {
        auto const local=DeckCoordinates(c.runway,wheel.point);
        if(local.north<0 || local.north>c.runway.lengthMeters || std::abs(local.east)>c.runway.widthMeters/2) return false;
    }
    return true;
}
bool AllWheelsDown(AircraftState const& a) noexcept
{
    if(!a.wheelDataValid) return a.weightOnWheels;
    for(auto const& wheel:a.wheels) if(!wheel.onGround) return false;
    return true;
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

std::size_t ClosestPatrolWaypoint(FlightPlan const& plan,Vec3 position) noexcept
{
    std::size_t closest{};
    double distance=std::numeric_limits<double>::max();
    for(std::size_t i=0;i<plan.count;++i)
    {
        double const candidate=SegmentDistance(position,plan.points[(i+plan.count-1)%plan.count],plan.points[i]);
        bool const atSegmentStart=Distance(position,plan.points[(i+plan.count-1)%plan.count])==0;
        if(candidate<distance || (candidate==distance && atSegmentStart)) {distance=candidate;closest=i;}
    }
    return closest;
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
        g.mode=GuidanceMode::Patrol;
        // Guidance and lap validation are deliberately separate.  Always use
        // the Blue-patrol projection/preview principle for control; the fixed
        // ordered gates below only validate physical circuit completion.  A
        // missed 100 m gate can therefore never leave the actor orbiting a
        // stale point behind or below the aircraft.
        auto const projection=ProjectOnPlan(plan,a.position);
        g.target=LookAheadOnPlan(plan,projection,
            c.mission.patrolSpeed*c.mission.patrolLookaheadSeconds);
        g.pathDistance=projection.distance;
        g.targetValid=true;
    }
    else
    {
        bool const flare = mission.capturedGlide && g.ilsValid
            && std::abs(g.glideError)<=c.mission.glideCaptureDegrees*DegreesToRadians*c.mission.captureHysteresis
            && g.height <= c.mission.flareHeightMeters;
        g.mode=mission.phase==Phase::Rollout ? GuidanceMode::Rollout
            : flare ? GuidanceMode::Flare
            : mission.capturedGlide ? GuidanceMode::Glide : GuidanceMode::Localizer;
        g.targetSpeed = mission.phase == Phase::Rollout ? 0 : (flare ? c.mission.flareSpeed : c.mission.approachSpeed);
        g.targetSlope = flare || mission.phase == Phase::Rollout ? 0 : -slope;
        // Captured approach, flare and rollout do not own a geometric point.
        // Keep the stored value at the aircraft as a defensive no-op for any
        // consumer that does not inspect targetValid.
        g.target = a.position;
        g.pathDistance = std::hypot(g.cross, g.height-(c.runway.thresholdHeightMeters-std::min(g.along,0.0)*std::tan(slope)));
        g.targetValid=false;
        if(mission.phase == Phase::Approach
            && (!mission.capturedLocalizer || !mission.capturedGlide))
        {
            // Until both beams are stably captured, keep one fixed front-side
            // intercept. A target derived from the aircraft along coordinate
            // would drift toward the threshold and jump after an overflight.
            double const targetAlong=-c.mission.switchDistanceMeters;
            g.target = FromRunway(c.runway,targetAlong,0,
                c.runway.thresholdHeightMeters-targetAlong*std::tan(slope));
            g.pathDistance = Distance(a.position,g.target);
            g.targetValid=true;
        }
    }
    g.courseError = g.targetValid
        ? WrapAngle(std::atan2(g.target.east-a.position.east,g.target.north-a.position.north)-a.heading) : 0;
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
    auto const& previousDeck=previousRunway ? *previousRunway : c.runway;
    double const pathProgress=AdvancePatrol(c,plan,s,before,after,previousDeck);
    bool const contact = after.weightOnWheels || after.structuralContact || g.height <= c.mission.contactHeightMeters;
    double const headingError=WrapAngle(after.heading-c.runway.headingDegrees*DegreesToRadians);
    bool const headingCompatible=std::abs(headingError)<=c.mission.approachHeadingToleranceDegrees*DegreesToRadians;
    bool enteredLocalizer{},enteredGlide{},lostLocalizer{},lostGlide{};
    if(s.phase!=Phase::Rollout)
    {
        // Once both beams have carried the aircraft across the threshold, keep
        // the final approach committed: the front-side ILS geometry naturally
        // becomes invalid behind its origin and must not trigger a go-around.
        bool const committedApproach=s.phase==Phase::Approach && s.capturedLocalizer
            && s.capturedGlide && g.along>=0;
        if(!committedApproach)
        {
            bool const captureGeometry=g.ilsValid && headingCompatible && !contact && IsFinite(after);
            auto const localizer=UpdateCapture(captureGeometry,g.localizer,
                c.mission.localizerCaptureDegrees*DegreesToRadians,c,s.localizerHoldSeconds,
                s.localizerLossSeconds,s.capturedLocalizer);
            enteredLocalizer=localizer.entered && !s.rewardedLocalizerCapture;
            lostLocalizer=localizer.lost;
            if(localizer.entered) s.rewardedLocalizerCapture=true;
            if(!localizer.lost && s.capturedLocalizer)
            {
                auto const glide=UpdateCapture(captureGeometry,g.glideError,
                    c.mission.glideCaptureDegrees*DegreesToRadians,c,s.glideHoldSeconds,
                    s.glideLossSeconds,s.capturedGlide);
                enteredGlide=glide.entered && !s.rewardedGlideCapture;
                lostGlide=glide.lost;
                if(glide.entered) s.rewardedGlideCapture=true;
            }
            else
            {
                s.glideHoldSeconds=0;
                s.glideLossSeconds=0;
                s.capturedGlide=false;
            }
        }
        else
        {
            s.localizerLossSeconds=0;
            s.glideLossSeconds=0;
        }
    }
    else
    {
        s.localizerHoldSeconds=0;s.glideHoldSeconds=0;
        s.localizerLossSeconds=0;s.glideLossSeconds=0;
    }
    // A confirmed beam loss abandons the intercept and re-enters the racetrack
    // through its projection/look-ahead guidance. Both captures are released
    // so that a fresh confirmed LOC then GLIDE restores approach priority.
    if(s.phase==Phase::Approach && (lostLocalizer || lostGlide))
        ReturnToPatrol(plan,s,after);
    else if(s.phase==Phase::Patrol && s.capturedLocalizer) s.phase=Phase::Approach;
    g = Guide(c,plan,s,after);
    if(contact && !s.contactAttempted && IsFinite(after))
    {
        s.contactAttempted=true;s.firstWheelContact=after.weightOnWheels;
        auto point=after.position;
        if(after.wheelDataValid) for(auto const& wheel:after.wheels) if(wheel.onGround) { point=wheel.point;break; }
        auto const local=DeckCoordinates(c.runway,point);
        s.touchdownFromThreshold=local.north;s.touchdownCrossMeters=local.east;
        auto const preVelocity=RelativeVelocity(previousDeck,before);
        s.touchdownSinkMps=std::max(-preVelocity.up,-g.relativeVelocity.up);
        s.touchdownSpeedMps=std::hypot(g.relativeVelocity.north,g.relativeVelocity.east);
        s.touchdownGearPosition=after.gearPosition;
    }
    bool crash = !IsFinite(before) || !IsFinite(after) || g.distance > c.mission.maxDistanceMeters
        || g.height > c.mission.maxAltitudeMeters;
    if (contact)
        crash = crash || s.phase == Phase::Patrol || !OnRunway(c,g) || !WheelsInside(c,after)
            || !LandingSafe(c,previousDeck,before,after) || !after.weightOnWheels;
    if (s.phase == Phase::Rollout && (!OnRunway(c,g) || !WheelsInside(c,after))) crash = true;
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
        auto const referenceBefore=InCurrentDeckFrame(previousDeck,c.runway,before);
        // Use the same phase and waypoint for both ends, including a gate pass.
        auto const previousGuidance=Guide(c,plan,s,referenceBefore);
        if (s.phase == Phase::Patrol)
            add(Reward::Path,GuidanceReward(c,GuidanceQuality(previousGuidance.pathDistance,c.reward.pathTolerance),
                GuidanceQuality(g.pathDistance,c.reward.pathTolerance),std::clamp(pathProgress/(c.mission.patrolSpeed*dt),0.0,1.0)));
        if (s.phase == Phase::Approach)
        {
            double const progress=std::clamp((g.along-previousGuidance.along)/(g.targetSpeed*dt),-1.0,1.0);
            // Decay only progress; the signed error potential must cancel across delayed reversals.
            double const ilsScale=std::exp(-static_cast<double>(s.steps-1)*dt/c.reward.ilsDecaySeconds);
            for(auto const channel : {Reward::Localizer,Reward::Glide})
            {
                bool const localizer=channel==Reward::Localizer;
                double const tolerance=(localizer ? c.reward.localizerToleranceDegrees : c.reward.glideToleranceDegrees)*DegreesToRadians;
                double const previousQuality=previousGuidance.ilsValid
                    ? GuidanceQuality(localizer ? previousGuidance.localizer : previousGuidance.glideError,tolerance) : 0;
                double const quality=g.ilsValid ? GuidanceQuality(localizer ? g.localizer : g.glideError,tolerance) : 0;
                double const value=GuidanceReward(c,previousQuality,quality,progress,ilsScale);
                // Angular errors can shrink merely by flying away from the
                // threshold; that must never earn a positive ILS reward.
                add(channel,progress<0 ? std::min(value,0.0) : value);
            }
            if(!s.capturedLocalizer || !s.capturedGlide)
            {
                double const previousDistance=previousGuidance.pathDistance;
                double const progressToIntercept=std::clamp((previousDistance-g.pathDistance)
                    /(c.mission.approachSpeed*dt),-1.0,1.0);
                add(Reward::Path,GuidanceReward(c,GuidanceQuality(previousDistance,c.reward.pathTolerance),
                    GuidanceQuality(g.pathDistance,c.reward.pathTolerance),progressToIntercept));
            }
            if(enteredLocalizer) add(Reward::EnterLocalizer,1);
            if(enteredGlide) add(Reward::EnterGlide,1);
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
            s.phase = Phase::Rollout;
            add(Reward::Touchdown,1);
        }
        if (s.phase == Phase::Rollout)
        {
            auto const previousPosition=DeckCoordinates(previousDeck,before.position);
            s.groundRoll += std::hypot(g.along-previousPosition.north,g.cross-previousPosition.east);
            bool const stopped = AllWheelsDown(after) && std::hypot(g.relativeVelocity.north,g.relativeVelocity.east) <= c.mission.stoppedSpeed;
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
    double const dn = g.targetValid ? g.target.north-a.position.north : 0;
    double const de = g.targetValid ? g.target.east-a.position.east : 0;
    double const forward = dn*std::cos(a.heading)+de*std::sin(a.heading);
    double const right = -dn*std::sin(a.heading)+de*std::cos(a.heading);
    double const heading = WrapAngle(a.heading-c.runway.headingDegrees*DegreesToRadians);
    double const slope = std::atan2(g.relativeVelocity.up,std::hypot(g.relativeVelocity.north,g.relativeVelocity.east));
    auto const deckVelocity=DeckVelocity(c.runway,a.position);
    auto const normal=RunwayNormal(c.runway);
    return {Unit(forward/n.distance), Unit(right/n.distance),
        Unit(g.targetValid ? (g.target.up-a.position.up)/n.altitude : 0),
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
        Unit(c.runway.angularVelocity.east/n.rate),Unit(c.runway.angularVelocity.up/n.rate),
        Unit(plan.count ? static_cast<double>(s.waypointsPassed)/static_cast<double>(plan.count) : 0),
        Unit(c.mission.requiredLaps ? static_cast<double>(s.completedLaps)/static_cast<double>(c.mission.requiredLaps) : 1),Unit(s.pathQualified),
        Unit(s.localizerHoldSeconds/c.mission.captureHoldSeconds),Unit(s.glideHoldSeconds/c.mission.captureHoldSeconds)};
}
} // namespace ppo::examples::flight
