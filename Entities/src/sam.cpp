#include "sam.hpp"

#include "snasystem.hpp"

#include "serializer.hpp"

namespace ppo::model {

SAM::SAM(entt::registry* registry, std::string const& name, eEntitySide side):
    Entity(registry, side, eEntityType::SAM)
{
    std::filesystem::path filepath{ RESOURCE_PATH };
    filepath = filepath / "Grounds";
    filepath = filepath / name;
    filepath.replace_extension(".json");
    filepath.make_preferred();

    if (std::filesystem::exists(filepath))
    {
        auto content = ppo::serialization::Serializer::Parse(filepath);

        auto section = content[name];

        SNASystem::Create(*this, section);
    }
}

void
SAM::OnInit(void) noexcept
{
    if (has<sSNAState>())
        SNASystem::Init(*this);
}

void
SAM::Run(void) noexcept
{
    Entity::Run();

    if (has<sSNAState>())
        SNASystem::Run(*this);
}

}