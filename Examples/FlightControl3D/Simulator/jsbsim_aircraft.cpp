#include "jsbsim_aircraft.hpp"
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
#include <algorithm>
#include <stdexcept>

namespace ppo::examples::flight {
namespace {
// WGS84 constants; these are units/geodesy, not tunable mission parameters.
constexpr double EarthSemiMajor = 6378137.0;
constexpr double EarthSemiMinor = 6356752.314245179;

JSBSim::FGLocation ReferenceLocation(Runway const& r)
{
    JSBSim::FGLocation reference;
    reference.SetEllipse(EarthSemiMajor/FeetToMeters,EarthSemiMinor/FeetToMeters);
    reference.SetPositionGeodetic(r.longitudeDegrees*DegreesToRadians,r.latitudeDegrees*DegreesToRadians,r.elevationMeters/FeetToMeters);
    return reference;
}
JSBSim::FGColumnVector3 ToNed(Vec3 point) { return {point.north/FeetToMeters,point.east/FeetToMeters,-point.up/FeetToMeters}; }
Vec3 FromNed(JSBSim::FGColumnVector3 const& p) { return {p(1)*FeetToMeters,p(2)*FeetToMeters,-p(3)*FeetToMeters}; }

/** The real JSBSim wheel/structure solver queries this finite level deck surrounded by the sea. */
class DeckGround final : public JSBSim::FGGroundCallback
{
public:
    JSBSim::FGLocation reference;
    Runway deck{};
    double seaHeight{};
    DeckGround(Runway const& runway,double sea) : reference(ReferenceLocation(runway)),deck(runway),seaHeight(sea) {}
    double GetAGLevel(double, JSBSim::FGLocation const& location, JSBSim::FGLocation& contact,
        JSBSim::FGColumnVector3& normal, JSBSim::FGColumnVector3& velocity, JSBSim::FGColumnVector3& angular) const override
    {
        auto const point=FromNed(reference.LocationToLocal(location));
        auto const local=DeckCoordinates(deck,point);
        bool const onDeck=OnLandingSurface(deck,local.north,local.east);
        double const surfaceHeight=onDeck ? deck.origin.up : seaHeight;
        Vec3 const surface{point.north,point.east,surfaceHeight};
        contact=reference.LocalToLocation(ToNed(surface));
        contact.SetEllipse(EarthSemiMajor/FeetToMeters,EarthSemiMinor/FeetToMeters);
        // Both the deck and the sea are level and static in the local tangent frame.
        normal=reference.GetTl2ec()*JSBSim::FGColumnVector3(0,0,-1);
        velocity=JSBSim::FGColumnVector3(0,0,0);
        angular=JSBSim::FGColumnVector3(0,0,0);
        return (point.up-surfaceHeight)/FeetToMeters;
    }
};
}

struct JsbsimAircraft::Impl final
{
    Config config;
    JSBSim::FGFDMExec fdm{};
    JSBSim::FGLocation reference;
    DeckGround* ground{}; // Owned by FGFDMExec::FGInertial through unique_ptr.
    std::array<SGPropertyNode*,18> reads{};
    SGPropertyNode* gearCommand{};
    explicit Impl(Config const& c, std::filesystem::path const& root) : config(c),reference(ReferenceLocation(c.runway))
    {
        fdm.SetDebugLevel(0);
        fdm.SetRootDir(SGPath(root.string()));
        fdm.SetAircraftPath(SGPath("aircraft"));
        fdm.SetEnginePath(SGPath("engine"));
        fdm.SetSystemsPath(SGPath("systems"));
        fdm.Setdt(config.mission.physicsDt);
        if (!fdm.LoadModel(config.aircraft)) throw std::runtime_error("FlightControl3D: JSBSim model loading failed");
        fdm.DisableOutput();
        auto const manager = fdm.GetPropertyManager();
        std::array<char const*,18> const names{
            "position/lat-geod-rad", "position/long-gc-rad", "position/h-sl-ft",
            "velocities/v-north-fps", "velocities/v-east-fps", "velocities/v-down-fps",
            "velocities/vc-fps", "attitude/phi-rad", "attitude/theta-rad", "attitude/psi-rad",
            "velocities/p-rad_sec", "velocities/q-rad_sec", "velocities/r-rad_sec", "gear/gear-pos-norm",
            "gear/unit[0]/WOW", "gear/unit[1]/WOW", "gear/unit[2]/WOW", "fcs/flap-pos-norm"};
        for (std::size_t i=0; i<reads.size(); ++i)
        {
            reads[i] = manager->GetNode(names[i],false);
            if (!reads[i]) throw std::runtime_error(std::string{"Missing JSBSim property: "}+names[i]);
        }
        gearCommand = manager->GetNode("gear/gear-cmd-norm",false);
        if (!gearCommand) throw std::runtime_error("Missing JSBSim gear command");
        auto callback=std::make_unique<DeckGround>(config.runway,config.runwayRandomization.seaHeightMeters);
        ground=callback.get();
        fdm.GetInertial()->SetGroundCallback(callback.release());
    }

    AircraftState Read() const
    {
        std::array<double,18> value{};
        for (std::size_t i=0; i<value.size(); ++i) value[i] = reads[i]->getDoubleValue();
        AircraftState a{};
        auto const propagate=fdm.GetPropagate();
        a.position=FromNed(reference.LocationToLocal(propagate->GetLocation()));
        a.velocity=FromNed(reference.GetTec2l()*propagate->GetTl2ec()*propagate->GetVel());
        a.airspeed = value[6]*FeetToMeters;
        a.roll = value[7]; a.pitch = value[8]; a.heading = value[9];
        a.rollRate = value[10]; a.pitchRate = value[11]; a.yawRate = value[12]; a.gearPosition = value[13];
        a.flapPosition=value[17];
        a.weightOnWheels = value[14] != 0 || value[15] != 0 || value[16] != 0;
        auto const ground = fdm.GetGroundReactions();
        std::size_t wheelIndex=0;
        for (int i=0; i<ground->GetNumGearUnits(); ++i)
        {
            auto const contact = ground->GetGearUnit(i);
            if (!contact->IsBogey() && contact->GetWOW()) a.structuralContact = true;
            if(contact->IsBogey() && wheelIndex<a.wheels.size())
            {
                auto const wheelLocation=propagate->GetLocation().LocalToLocation(contact->GetLocalGear());
                a.wheels[wheelIndex++]={FromNed(reference.LocationToLocal(wheelLocation)),contact->GetWOW()};
            }
        }
        a.wheelDataValid=wheelIndex==a.wheels.size();
        a.finite = IsFinite(a);
        return a;
    }
};

JsbsimAircraft::JsbsimAircraft(Config const& config, std::filesystem::path const& dataRoot)
    : impl_(std::make_unique<Impl>(config,dataRoot)) {}
JsbsimAircraft::~JsbsimAircraft() = default;

AircraftState JsbsimAircraft::Reset(AircraftState const& initial,Phase phase)
{
    if (!IsFinite(initial)) throw std::invalid_argument("FlightControl3D: invalid JSBSim initial state");
    auto& p = *impl_;
    p.fdm.ResetToInitialConditions(JSBSim::FGFDMExec::DONT_EXECUTE_RUN_IC);
    auto location=p.reference.LocalToLocation(ToNed(initial.position));
    location.SetEllipse(EarthSemiMajor/FeetToMeters,EarthSemiMinor/FeetToMeters);
    auto* const ic = p.fdm.GetIC().get();
    ic->SetGeodLatitudeDegIC(location.GetGeodLatitudeDeg());
    ic->SetLongitudeDegIC(location.GetLongitude()/DegreesToRadians);
    ic->SetTerrainElevationFtIC(p.config.runway.elevationMeters/FeetToMeters);
    ic->SetAltitudeASLFtIC(location.GetGeodAltitude());
    ic->SetVcalibratedKtsIC(initial.airspeed/(1852.0/3600.0));
    // SetClimbRateFpsIC preserves incidence by changing theta: using it after
    // SetThetaRadIC silently replaces the sampled pitch by the flight-path angle.
    // Explicit NED velocity setters retain the velocity when attitude is set last.
    double const trueSpeed=ic->GetVtrueFpsIC();
    double const climb=initial.velocity.up/FeetToMeters;
    if(std::abs(climb)>trueSpeed)
        throw std::invalid_argument("FlightControl3D: initial climb exceeds true airspeed");
    double const horizontal=std::sqrt(std::max(0.0,trueSpeed*trueSpeed-climb*climb));
    double const requestedHorizontal=std::hypot(initial.velocity.north,initial.velocity.east);
    double const course=requestedHorizontal>0
        ? std::atan2(initial.velocity.east,initial.velocity.north) : initial.heading;
    ic->SetVNorthFpsIC(horizontal*std::cos(course));
    ic->SetVEastFpsIC(horizontal*std::sin(course));
    ic->SetVDownFpsIC(-climb);
    ic->SetPhiRadIC(initial.roll); ic->SetThetaRadIC(initial.pitch); ic->SetPsiRadIC(initial.heading);
    ic->SetPRadpsIC(0); ic->SetQRadpsIC(0); ic->SetRRadpsIC(0);
    // Airborne starts hold the trimmed approach controls; rolling starts are at idle.
    bool const airborne=phase==Phase::Approach;
    p.fdm.GetFCS()->SetDaCmd(0); p.fdm.GetFCS()->SetDrCmd(0); p.fdm.GetFCS()->SetDsCmd(0);
    p.fdm.GetFCS()->SetDeCmd(airborne ? p.config.initial.elevatorTrim : 0.0);
    p.fdm.GetFCS()->SetThrottleCmd(-1,airborne ? p.config.initial.throttle : 0.0);
    p.fdm.GetFCS()->SetMixtureCmd(-1,p.config.initial.mixture);
    p.fdm.GetFCS()->SetPropAdvanceCmd(-1,p.config.initial.propeller);
    double const flap=phase==Phase::Approach ? p.config.initial.approachFlap : p.config.initial.rolloutFlap;
    p.fdm.GetFCS()->SetDfCmd(flap);
    // The pinned T-6 has 30 degrees of flap travel.
    constexpr double flapTravelDegrees=30.0;
    p.fdm.GetFCS()->SetDfPos(JSBSim::ofDeg,flapTravelDegrees*flap);
    p.fdm.GetFCS()->SetLBrake(0); p.fdm.GetFCS()->SetRBrake(0); p.fdm.GetFCS()->SetCBrake(0);
    // The landing gear is locked down for the whole mission.
    p.gearCommand->setDoubleValue(1);
    p.fdm.GetFCS()->SetGearPos(1);
    if (!p.fdm.RunIC()) throw std::runtime_error("FlightControl3D: JSBSim RunIC failed");
    // FGPropulsion::InitRunning also solves steady propulsion with 0.5 s
    // internal steps. Initialize each native engine directly and let normal
    // physics integrate the transient.
    auto const propulsion=p.fdm.GetPropulsion();
    for(unsigned int engine=0;engine<propulsion->GetNumEngines();++engine)
        propulsion->GetEngine(engine)->InitRunning();
    auto state = p.Read();
    if (!IsFinite(state)) throw std::runtime_error("FlightControl3D: non-finite JSBSim reset");
    return state;
}

AircraftState JsbsimAircraft::Advance(Controls const& controls)
{
    std::array<double,6> const values{controls.elevator,controls.aileron,controls.rudder,controls.throttle,controls.brake,controls.flap};
    for (double value : values) if (!std::isfinite(value)) throw std::invalid_argument("FlightControl3D: invalid flight control");
    auto& p = *impl_;
    auto* const fcs = p.fdm.GetFCS().get();
    fcs->SetDeCmd(std::clamp(controls.elevator,-1.0,1.0));
    fcs->SetDaCmd(std::clamp(controls.aileron,-1.0,1.0));
    fcs->SetDrCmd(std::clamp(controls.rudder,-1.0,1.0));
    // The rudder pedals also steer the nose wheel, as on the real T-6. The
    // pinned FCS does not wire fcs/steer-cmd-norm, so without this command the
    // nose wheel stays straight and the ground track cannot be corrected once
    // the rudder loses its aerodynamic authority. Airborne, the wheel carries
    // no load and the steering has no effect. A negative rudder command yaws
    // the T-6 nose right while a positive steering command turns the wheel
    // right, hence the opposite sign: both effects then turn the same way.
    fcs->SetDsCmd(-std::clamp(controls.rudder,-1.0,1.0));
    fcs->SetThrottleCmd(-1,std::clamp(controls.throttle,0.0,1.0));
    fcs->SetDfCmd(std::clamp(controls.flap,0.0,1.0));
    fcs->SetLBrake(std::clamp(controls.brake,0.0,1.0));
    fcs->SetRBrake(std::clamp(controls.brake,0.0,1.0));
    fcs->SetCBrake(std::clamp(controls.brake,0.0,1.0));
    p.gearCommand->setDoubleValue(1);
    bool const advanced = p.fdm.Run();
    auto state = p.Read();
    state.finite = state.finite && advanced;
    return state;
}

std::filesystem::path DefaultDataRoot() { return std::filesystem::path{PPO_JSBSIM_DATA_ROOT}; }
void JsbsimAircraft::SetRunway(Runway const& runway) { impl_->ground->deck=runway; }
GroundSample JsbsimAircraft::ProbeGround(Vec3 point) const
{
    auto const& p=*impl_;
    auto const location=p.reference.LocalToLocation(ToNed(point));
    JSBSim::FGLocation contact;
    JSBSim::FGColumnVector3 normal,velocity,angular;
    double const height=p.fdm.GetInertial()->GetContactPoint(location,contact,normal,velocity,angular);
    auto const localNormal=p.reference.GetTec2l()*normal;
    return {height*FeetToMeters,FromNed(p.reference.LocationToLocal(contact)),
        {localNormal(1),localNormal(2),-localNormal(3)}};
}
} // namespace ppo::examples::flight
