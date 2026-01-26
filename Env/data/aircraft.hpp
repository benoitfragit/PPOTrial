#pragma once

#include "entity.hpp"
#include "serializer.hpp"

#include "sna.hpp"
#include "flightmodel.hpp"

namespace ppo::model {

class Aircraft: 
    public Entity
{
public:
    template <typename Ty>
    Aircraft(std::string const& name, eEntitySide side, Ty& env):
        Entity(name, side, eEntityType::Aircraft, env) 
    {        
        std::filesystem::path filepath{ "Aircrafts" };
        filepath = filepath / name;
        filepath.replace_extension(".toml");
        
        ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
        serializer.Register(
            filepath,
            [this](ppo::serialization::TomlReadArchive& ar)
            {
                serialize(ar);
        
                UpdateDoubleBuffer();
            },
            nullptr
        );

    };

    template<typename Ar>
    void serialize(Ar& ar)
    {
        auto section = ar.begin_section(m_name.c_str());

        m_flightModel.serialize (section);

        if (p_sna != nullptr)
            p_sna->serialize         (section);
    };


    void              SetSNA        (std::shared_ptr<SNA> sna)                noexcept;
    void              Init          (void)                                    noexcept final;
    void              Run           (void)                                    noexcept final;
    void              UpdateEnv     (void)                              const noexcept final;
    void              SetCheckpoint (CheckPointBuffer const& buffer)          noexcept final;
    CheckPointBuffer  GetCheckpoint (void)                                    noexcept final;           

    void              SetTarget     (EntityId targetId)                       noexcept;

    SNA         const& GetSNA        (void)         const noexcept;    
    FlightModel const& GetFlightModel(void)         const noexcept;

    ISNAState*          GetISNA         (void)            noexcept override { return p_sna.get(); };
    IFlightModelState*  GetIFlightModel (void)            noexcept override { return &m_flightModel; };

protected:
    FlightModel             m_flightModel   { m_state };
    std::shared_ptr<SNA>    p_sna           { nullptr };
};

}