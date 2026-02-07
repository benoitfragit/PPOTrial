#include "aircraft.hpp"

#include "snasystem.hpp"
#include "flightmodelsystem.hpp"
#include "snacontrolsystem.hpp"
#include "randominitializersystem.hpp"
#include "checkpointinitializersystem.hpp"

#include "serializer.hpp"

namespace ppo::model {

Aircraft::Aircraft(entt::registry* registry, std::string const& name, eEntitySide side):
    Entity(registry, side, eEntityType::Aircraft)
{
    std::filesystem::path filepath{ RESOURCE_PATH };
    filepath = filepath / "Aircrafts";
    filepath = filepath / name;
    filepath.replace_extension(".json");
    filepath.make_preferred();

    if (std::filesystem::exists(filepath))
    {
        auto content = ppo::serialization::Serializer::Parse(filepath);

        auto section = content[name];

        FlightModelSystem::Create(*this, section);
        SNASystem::Create(*this, section);
        SNAControlSystem::Create(*this, section);
    }
}

void
Aircraft::OnInit(void) noexcept
{
    if (has<sSNAControlState>())
        SNAControlSystem::Init(*this);

    if (has<sFlightModelState>())
        FlightModelSystem::Init(*this);

    if (has<sSNAState>())
        SNASystem::Init(*this);
}

void
Aircraft::Run(void) noexcept
{
    if (has<sSNAControlState>())
        SNAControlSystem::Run(*this);

    if (has<sFlightModelState>())
        FlightModelSystem::Run(*this);

    Entity::Run();

    if (has<sSNAState>())
        SNASystem::Run(*this);
}

}