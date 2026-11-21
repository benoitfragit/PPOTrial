/**
 * @file jsbsim_aircraft.cpp
 * @brief JSBSim adapter: unit and frame conversions, deck ground callback, reset and stepping.
 *
 * Conventions:
 * - lengths: JSBSim uses feet; FlightControl3D uses metres (FeetToMeters = 0.3048);
 * - frames: JSBSim local frames are NED (down positive); FlightControl3D is north-east-up,
 *   so the third component changes sign in ToNed / FromNed;
 * - geodesy: the local tangent frame is anchored at the runway reference location on the
 *   WGS84 ellipsoid; JSBSim locations are converted with FGLocation::LocationToLocal;
 * - speeds: the initial calibrated airspeed is given to JSBSim in knots.
 *
 * Call order inside Reset matters: JSBSim's initial-condition setters interact (see the
 * comment on SetClimbRateFpsIC), so the velocity is set from NED components before the
 * attitude.
 *
 * Read next: jsbsim_aircraft.hpp, flight_environment.cpp (command filter and action loop).
 */
#include "jsbsim_aircraft.hpp"
#include "flight_aircraft_performance.hpp"
#include <FGFDMExec.h>
#include <input_output/FGPropertyManager.h>
#include <models/FGFCS.h>
#include <models/FGPropulsion.h>
#include <models/propulsion/FGEngine.h>
#include <models/FGGroundReactions.h>
#include <models/FGLGear.h>
#include <models/FGInertial.h>
#include <models/FGPropagate.h>
#include <input_output/FGGroundCallback.h>
#include <initialization/FGInitialCondition.h>
#include <initialization/FGTrim.h>
#include <algorithm>
#include <stdexcept>

namespace ppo::examples::flight {
namespace {
// WGS84 constants; these are units/geodesy, not tunable mission parameters.
constexpr double EarthSemiMajor = 6378137.0;
constexpr double EarthSemiMinor = 6356752.314245179;
/** @brief One international knot (1852 m per hour), in metres per second. */
constexpr double KnotsToMetersPerSecond = 1852.0 / 3600.0;
/** @brief JSBSim engine index meaning "every engine". */
constexpr int AllEngines = -1;

/** @brief Applies the WGS84 ellipsoid, in feet, to a JSBSim location. */
void SetWgs84Ellipse(JSBSim::FGLocation& location)
{
    location.SetEllipse(EarthSemiMajor / FeetToMeters, EarthSemiMinor / FeetToMeters);
}

/** @brief JSBSim location of the local tangent frame origin (the configured runway reference point). */
JSBSim::FGLocation ReferenceLocation(Runway const& runway)
{
    JSBSim::FGLocation reference;
    SetWgs84Ellipse(reference);
    reference.SetPositionGeodetic(runway.longitudeDegrees * DegreesToRadians,
        runway.latitudeDegrees * DegreesToRadians, runway.elevationMeters / FeetToMeters);
    return reference;
}

/** @brief Converts a north-east-up point in metres to a JSBSim local NED vector in feet. */
JSBSim::FGColumnVector3 ToNed(Vec3 point)
{
    return {point.north / FeetToMeters, point.east / FeetToMeters, -point.up / FeetToMeters};
}

/** @brief Converts a JSBSim local NED vector in feet to a north-east-up vector in metres. */
Vec3 FromNed(JSBSim::FGColumnVector3 const& ned)
{
    return {ned(1) * FeetToMeters, ned(2) * FeetToMeters, -ned(3) * FeetToMeters};
}

/**
 * @brief Ground seen by JSBSim: a finite level deck surrounded by the sea.
 *
 * The real JSBSim wheel/structure solver queries this callback, so gear forces, contact
 * flags and crashes on the sea are all computed natively.
 */
class DeckGround final : public JSBSim::FGGroundCallback
{
public:
    /** @brief Local tangent frame origin. */
    JSBSim::FGLocation reference;
    /** @brief Current episode runway; its landable surface is the deck. */
    Runway deck{};
    /** @brief Sea height outside the deck, in metres. */
    double seaHeight{};

    DeckGround(Runway const& runway, double sea)
        : reference(ReferenceLocation(runway)), deck(runway), seaHeight(sea)
    {
    }

    /**
     * @brief Height above the deck or the sea, in feet, with the contact point, a level normal
     *        and the surface velocity in ECEF feet per second.
     */
    double GetAGLevel(double, JSBSim::FGLocation const& location, JSBSim::FGLocation& contact,
        JSBSim::FGColumnVector3& normal, JSBSim::FGColumnVector3& velocity,
        JSBSim::FGColumnVector3& angular) const override
    {
        auto const point = FromNed(reference.LocationToLocal(location));
        auto const local = DeckCoordinates(deck, point);
        bool const onDeck = OnLandingSurface(deck, local.north, local.east);
        double surfaceHeight = seaHeight;
        Vec3 surfaceVelocity{};
        if (onDeck)
        {
            surfaceHeight = deck.origin.up;
            surfaceVelocity = deck.velocity;
        }
        Vec3 const surface{point.north, point.east, surfaceHeight};
        contact = reference.LocalToLocation(ToNed(surface));
        SetWgs84Ellipse(contact);
        // Both the deck and the sea are level. The sea is static; a moving deck reports its
        // velocity, so JSBSim computes the strut compression and tyre slip relative to it.
        normal = reference.GetTl2ec() * JSBSim::FGColumnVector3(0, 0, -1);
        velocity = reference.GetTl2ec() * ToNed(surfaceVelocity);
        angular = JSBSim::FGColumnVector3(0, 0, 0);
        return (point.up - surfaceHeight) / FeetToMeters;
    }
};

/** @brief JSBSim properties read at every step, in the order of PropertyNames. */
enum PropertyIndex : std::size_t
{
    Latitude,
    Longitude,
    AltitudeAboveSeaLevel,
    VelocityNorth,
    VelocityEast,
    VelocityDown,
    CalibratedSpeed,
    Roll,
    Pitch,
    Heading,
    RollRate,
    PitchRate,
    YawRate,
    GearPosition,
    NoseWheelWeight,
    LeftWheelWeight,
    RightWheelWeight,
    FlapPosition,
    PropertyCount
};

/** @brief JSBSim property paths, in PropertyIndex order. */
constexpr std::array<char const*, PropertyCount> PropertyNames{
    "position/lat-geod-rad", "position/long-gc-rad", "position/h-sl-ft",
    "velocities/v-north-fps", "velocities/v-east-fps", "velocities/v-down-fps",
    "velocities/vc-fps", "attitude/phi-rad", "attitude/theta-rad", "attitude/psi-rad",
    "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec", "gear/gear-pos-norm",
    "gear/unit[0]/WOW", "gear/unit[1]/WOW", "gear/unit[2]/WOW", "fcs/flap-pos-norm"};
} // namespace

/** @brief JSBSim executive, property handles and ground callback of one aircraft. */
struct JsbsimAircraft::Impl final
{
    Config config;
    JSBSim::FGFDMExec fdm{};
    /** @brief Local tangent frame origin. */
    JSBSim::FGLocation reference;
    /** @brief Owned by FGFDMExec::FGInertial through unique_ptr. */
    DeckGround* ground{};
    /** @brief Property handles, in PropertyIndex order. */
    std::array<SGPropertyNode*, PropertyCount> reads{};
    SGPropertyNode* gearCommand{};

    explicit Impl(Config const& configuration, std::filesystem::path const& root)
        : config(configuration), reference(ReferenceLocation(configuration.runway))
    {
        fdm.SetDebugLevel(0);
        fdm.SetRootDir(SGPath(root.string()));
        fdm.SetAircraftPath(SGPath("aircraft"));
        fdm.SetEnginePath(SGPath("engine"));
        fdm.SetSystemsPath(SGPath("systems"));
        fdm.Setdt(config.mission.physicsDt);
        if (!fdm.LoadModel(config.aircraft))
        {
            throw std::runtime_error("FlightControl3D: JSBSim model loading failed");
        }
        fdm.DisableOutput();
        BindProperties();
        // JSBSim takes ownership of the callback; keep a raw pointer to move the deck per episode.
        auto callback = std::make_unique<DeckGround>(config.runway,
            config.runwayRandomization.seaHeightMeters);
        ground = callback.get();
        fdm.GetInertial()->SetGroundCallback(callback.release());
    }

    /** @brief Resolves every property handle once, so each step reads them without lookups. */
    void BindProperties()
    {
        auto const manager = fdm.GetPropertyManager();
        for (std::size_t index = 0; index < reads.size(); ++index)
        {
            reads[index] = manager->GetNode(PropertyNames[index], false);
            if (!reads[index])
            {
                throw std::runtime_error(std::string{"Missing JSBSim property: "} + PropertyNames[index]);
            }
        }
        gearCommand = manager->GetNode("gear/gear-cmd-norm", false);
        if (!gearCommand)
        {
            throw std::runtime_error("Missing JSBSim gear command");
        }
    }

    /** @brief Fills the wheel samples and the structural contact flag from the gear units. */
    void ReadContacts(AircraftState& state) const
    {
        auto const propagate = fdm.GetPropagate();
        auto const groundReactions = fdm.GetGroundReactions();
        std::size_t wheelIndex = 0;
        for (int unit = 0; unit < groundReactions->GetNumGearUnits(); ++unit)
        {
            auto const contact = groundReactions->GetGearUnit(unit);
            // Bogey units are wheels; any other contact point is structure.
            if (!contact->IsBogey() && contact->GetWOW())
            {
                state.structuralContact = true;
            }
            if (contact->IsBogey() && wheelIndex < state.wheels.size())
            {
                auto const wheelLocation = propagate->GetLocation().LocalToLocation(contact->GetLocalGear());
                state.wheels[wheelIndex++] = {FromNed(reference.LocationToLocal(wheelLocation)),
                    contact->GetWOW()};
            }
        }
        state.wheelDataValid = wheelIndex == state.wheels.size();
    }

    /** @brief Reads the SI aircraft state from JSBSim. */
    AircraftState Read() const
    {
        std::array<double, PropertyCount> value{};
        for (std::size_t index = 0; index < value.size(); ++index)
        {
            value[index] = reads[index]->getDoubleValue();
        }
        AircraftState state{};
        auto const propagate = fdm.GetPropagate();
        state.position = FromNed(reference.LocationToLocal(propagate->GetLocation()));
        // Body velocity -> ECEF -> local frame of the reference point.
        state.velocity = FromNed(reference.GetTec2l() * propagate->GetTl2ec() * propagate->GetVel());
        state.airspeed = value[CalibratedSpeed] * FeetToMeters;
        state.roll = value[Roll];
        state.pitch = value[Pitch];
        state.heading = value[Heading];
        state.rollRate = value[RollRate];
        state.pitchRate = value[PitchRate];
        state.yawRate = value[YawRate];
        state.gearPosition = value[GearPosition];
        state.flapPosition = value[FlapPosition];
        state.weightOnWheels = value[NoseWheelWeight] != 0 || value[LeftWheelWeight] != 0
            || value[RightWheelWeight] != 0;
        ReadContacts(state);
        state.finite = IsFinite(state);
        return state;
    }

    /** @brief Sets the initial position, calibrated speed, velocity direction and attitude. */
    void SetInitialKinematics(AircraftState const& initial)
    {
        auto location = reference.LocalToLocation(ToNed(initial.position));
        SetWgs84Ellipse(location);
        auto* const conditions = fdm.GetIC().get();
        conditions->SetGeodLatitudeDegIC(location.GetGeodLatitudeDeg());
        conditions->SetLongitudeDegIC(location.GetLongitude() / DegreesToRadians);
        conditions->SetTerrainElevationFtIC(config.runway.elevationMeters / FeetToMeters);
        conditions->SetAltitudeASLFtIC(location.GetGeodAltitude());
        conditions->SetVcalibratedKtsIC(initial.airspeed / KnotsToMetersPerSecond);
        // SetClimbRateFpsIC preserves incidence by changing theta: using it after
        // SetThetaRadIC silently replaces the sampled pitch by the flight-path angle.
        // Explicit NED velocity setters retain the velocity when attitude is set last.
        double const trueSpeedFps = conditions->GetVtrueFpsIC();
        double const climbFps = initial.velocity.up / FeetToMeters;
        if (std::abs(climbFps) > trueSpeedFps)
        {
            throw std::invalid_argument("FlightControl3D: initial climb exceeds true airspeed");
        }
        // Keep JSBSim's true airspeed and the requested climb; the requested
        // horizontal direction (or the heading when hovering) sets the course.
        double const horizontalSquared = trueSpeedFps * trueSpeedFps - climbFps * climbFps;
        double const horizontalFps = std::sqrt(std::max(0.0, horizontalSquared));
        double const requestedHorizontal = std::hypot(initial.velocity.north, initial.velocity.east);
        double course = initial.heading;
        if (requestedHorizontal > 0)
        {
            course = std::atan2(initial.velocity.east, initial.velocity.north);
        }
        conditions->SetVNorthFpsIC(horizontalFps * std::cos(course));
        conditions->SetVEastFpsIC(horizontalFps * std::sin(course));
        conditions->SetVDownFpsIC(-climbFps);
        conditions->SetPhiRadIC(initial.roll);
        conditions->SetThetaRadIC(initial.pitch);
        conditions->SetPsiRadIC(initial.heading);
        conditions->SetPRadpsIC(0);
        conditions->SetQRadpsIC(0);
        conditions->SetRRadpsIC(0);
    }

    /**
     * @brief Sets the initial commands of a reset.
     * @param phase Phase selecting the flap detent.
     * @param gear Initial gear lever and position: 1 down and locked, 0 retracted.
     * @param commands Initial elevator and throttle.
     */
    void SetInitialControls(Phase phase, double gear, InitialCommands commands)
    {
        auto* const fcs = fdm.GetFCS().get();
        fcs->SetDaCmd(0);
        fcs->SetDrCmd(0);
        fcs->SetDsCmd(0);
        fcs->SetDeCmd(commands.elevator);
        // A previous Trim leaves a pitch-trim command; episodes fly on the elevator only.
        fcs->SetPitchTrimCmd(0);
        fcs->SetThrottleCmd(AllEngines, commands.throttle);
        fcs->SetMixtureCmd(AllEngines, config.initial.mixture);
        fcs->SetPropAdvanceCmd(AllEngines, config.initial.propeller);
        double flap = config.initial.rolloutFlap;
        if (phase == Phase::Approach)
        {
            flap = config.initial.approachFlap;
        }
        fcs->SetDfCmd(flap);
        // Start with the flaps already at the commanded detent.
        fcs->SetDfPos(JSBSim::ofDeg, config.performance.flapTravelDegrees * flap);
        fcs->SetLBrake(0);
        fcs->SetRBrake(0);
        fcs->SetCBrake(0);
        // The gear starts at rest in the requested position, never in transit.
        gearCommand->setDoubleValue(gear);
        fcs->SetGearPos(gear);
    }
};

JsbsimAircraft::JsbsimAircraft(Config const& config, std::filesystem::path const& dataRoot)
    : impl_(std::make_unique<Impl>(config, dataRoot))
{
}

JsbsimAircraft::~JsbsimAircraft() = default;

AircraftState JsbsimAircraft::Reset(AircraftState const& initial, Phase phase)
{
    // Airborne starts hold the action reference (the trimmed approach); rolling starts are at idle.
    InitialCommands commands{};
    if (phase == Phase::Approach)
    {
        commands = {impl_->config.initial.elevatorTrim, impl_->config.initial.throttle};
    }
    return Reset(initial, phase, commands);
}

AircraftState JsbsimAircraft::Reset(AircraftState const& initial, Phase phase, InitialCommands commands)
{
    if (!IsFinite(initial) || !std::isfinite(commands.elevator) || !std::isfinite(commands.throttle))
    {
        throw std::invalid_argument("FlightControl3D: invalid JSBSim initial state");
    }
    auto& impl = *impl_;
    impl.fdm.ResetToInitialConditions(JSBSim::FGFDMExec::DONT_EXECUTE_RUN_IC);
    impl.SetInitialKinematics(initial);
    commands.elevator = std::clamp(commands.elevator, -1.0, 1.0);
    commands.throttle = std::clamp(commands.throttle, 0.0, 1.0);
    impl.SetInitialControls(phase, initial.gearPosition, commands);
    if (!impl.fdm.RunIC())
    {
        throw std::runtime_error("FlightControl3D: JSBSim RunIC failed");
    }
    // FGPropulsion::InitRunning also solves steady propulsion with 0.5 s
    // internal steps. Initialize each native engine directly and let normal
    // physics integrate the transient.
    auto const propulsion = impl.fdm.GetPropulsion();
    for (unsigned int engine = 0; engine < propulsion->GetNumEngines(); ++engine)
    {
        propulsion->GetEngine(engine)->InitRunning();
    }
    auto state = impl.Read();
    if (!IsFinite(state))
    {
        throw std::runtime_error("FlightControl3D: non-finite JSBSim reset");
    }
    return state;
}

TrimResult JsbsimAircraft::Trim(TrimCondition const& condition)
{
    auto& impl = *impl_;
    // FGTrim would corrupt the engine state of an aircraft it cannot trim.
    if (!impl.config.performance.jsbsimTrim)
    {
        return {};
    }
    impl.fdm.ResetToInitialConditions(JSBSim::FGFDMExec::DONT_EXECUTE_RUN_IC);
    // Wings level above the local frame origin, heading north; the trim
    // itself solves the angle of attack, so only the path and speed are set.
    auto location = impl.reference.LocalToLocation(ToNed({0, 0, condition.heightMeters}));
    SetWgs84Ellipse(location);
    auto* const conditions = impl.fdm.GetIC().get();
    // Start from blank initial conditions: the attitude, incidence and sideslip of the
    // previous reset would otherwise seed the trim and make it depend on the history.
    conditions->InitializeIC();
    conditions->SetGeodLatitudeDegIC(location.GetGeodLatitudeDeg());
    conditions->SetLongitudeDegIC(location.GetLongitude() / DegreesToRadians);
    conditions->SetTerrainElevationFtIC(impl.config.runway.elevationMeters / FeetToMeters);
    conditions->SetAltitudeASLFtIC(location.GetGeodAltitude());
    conditions->SetVcalibratedKtsIC(condition.calibratedSpeed / KnotsToMetersPerSecond);
    conditions->SetFlightPathAngleRadIC(condition.flightPathRadians);
    conditions->SetPhiRadIC(0);
    conditions->SetPsiRadIC(0);
    auto* const fcs = impl.fdm.GetFCS().get();
    fcs->SetDaCmd(0);
    fcs->SetDrCmd(0);
    fcs->SetDeCmd(0);
    fcs->SetPitchTrimCmd(0);
    fcs->SetMixtureCmd(AllEngines, impl.config.initial.mixture);
    fcs->SetPropAdvanceCmd(AllEngines, impl.config.initial.propeller);
    fcs->SetDfCmd(condition.flap);
    fcs->SetDfPos(JSBSim::ofDeg, impl.config.performance.flapTravelDegrees * condition.flap);
    impl.gearCommand->setDoubleValue(condition.gear);
    fcs->SetGearPos(condition.gear);
    if (!impl.fdm.RunIC())
    {
        throw std::runtime_error("FlightControl3D: JSBSim RunIC failed before trim");
    }
    auto const propulsion = impl.fdm.GetPropulsion();
    for (unsigned int engine = 0; engine < propulsion->GetNumEngines(); ++engine)
    {
        propulsion->GetEngine(engine)->InitRunning();
    }
    // Longitudinal trim: throttle for udot, alpha for wdot, pitch trim for qdot.
    // The gamma fallback stays disabled: a trim that cannot hold the requested
    // path with the throttle fails instead of silently changing the path.
    JSBSim::FGTrim trim(&impl.fdm, JSBSim::tLongitudinal);
    TrimResult result{};
    result.converged = trim.DoTrim();
    result.elevator = fcs->GetPitchTrimCmd();
    result.throttle = fcs->GetThrottleCmd(0);
    result.pitch = impl.fdm.GetPropagate()->GetEuler(JSBSim::FGJSBBase::eTht);
    fcs->SetPitchTrimCmd(0);
    return result;
}

AircraftState JsbsimAircraft::Advance(Controls const& controls)
{
    std::array<double, 7> const values{controls.elevator, controls.aileron, controls.rudder,
        controls.throttle,
        controls.brake, controls.flap, controls.gear};
    for (double const value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("FlightControl3D: invalid flight control");
        }
    }
    auto& impl = *impl_;
    auto* const fcs = impl.fdm.GetFCS().get();
    fcs->SetDeCmd(std::clamp(controls.elevator, -1.0, 1.0));
    fcs->SetDaCmd(std::clamp(controls.aileron, -1.0, 1.0));
    fcs->SetDrCmd(std::clamp(controls.rudder, -1.0, 1.0));
    // The rudder pedals also steer the nose wheel, as on the real T-6. The
    // pinned FCS does not wire fcs/steer-cmd-norm, so without this command the
    // nose wheel stays straight and the ground track cannot be corrected once
    // the rudder loses its aerodynamic authority. Airborne, the wheel carries
    // no load and the steering has no effect. A negative rudder command yaws
    // the T-6 nose right while a positive steering command turns the wheel
    // right, hence the opposite sign: both effects then turn the same way.
    fcs->SetDsCmd(-std::clamp(controls.rudder, -1.0, 1.0));
    fcs->SetThrottleCmd(AllEngines, std::clamp(controls.throttle, 0.0, 1.0));
    fcs->SetDfCmd(std::clamp(controls.flap, 0.0, 1.0));
    // One symmetric brake command drives the left, right and centre brakes.
    fcs->SetLBrake(std::clamp(controls.brake, 0.0, 1.0));
    fcs->SetRBrake(std::clamp(controls.brake, 0.0, 1.0));
    fcs->SetCBrake(std::clamp(controls.brake, 0.0, 1.0));
    impl.gearCommand->setDoubleValue(std::clamp(controls.gear, 0.0, 1.0));
    bool const advanced = impl.fdm.Run();
    auto state = impl.Read();
    state.finite = state.finite && advanced;
    return state;
}

std::filesystem::path DefaultDataRoot()
{
    return std::filesystem::path{PPO_JSBSIM_DATA_ROOT};
}

void JsbsimAircraft::SetRunway(Runway const& runway)
{
    impl_->ground->deck = runway;
}

GroundSample JsbsimAircraft::ProbeGround(Vec3 point) const
{
    auto const& impl = *impl_;
    auto const location = impl.reference.LocalToLocation(ToNed(point));
    JSBSim::FGLocation contact;
    JSBSim::FGColumnVector3 normal;
    JSBSim::FGColumnVector3 velocity;
    JSBSim::FGColumnVector3 angular;
    double const heightFeet = impl.fdm.GetInertial()->GetContactPoint(location, contact, normal, velocity,
        angular);
    auto const localNormal = impl.reference.GetTec2l() * normal;
    return {heightFeet * FeetToMeters, FromNed(impl.reference.LocationToLocal(contact)),
        {localNormal(1), localNormal(2), -localNormal(3)}, FromNed(impl.reference.GetTec2l() * velocity)};
}
} // namespace ppo::examples::flight
