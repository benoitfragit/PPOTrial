#pragma once
#include "aircraft.hpp"

namespace ppo::mission {

class RLTargetSelector {
public:
    class IEnvironment
    {
    public:
        virtual                     ~IEnvironment(void) = default;
        virtual ppo::model::Entity* GetEntityAtIndex(unsigned int index) noexcept = 0;
        virtual unsigned int        GetNumberOfEntities(void) const noexcept = 0;
    };

    template<class Ty>
    class EnvironmentWrapper :
        public IEnvironment
    {
    public:
        EnvironmentWrapper(Ty& env) : m_env{ env } {};

        ppo::model::Entity* 
        GetEntityAtIndex(unsigned int index) noexcept
        {
            return dynamic_cast<ppo::model::Entity*>(m_env.GetEntityAtIndex(index));
        };
        
        unsigned int
            GetNumberOfEntities(void) const noexcept
        {
            return m_env.GetNumberOfEntities();
        };

    private:
        Ty& m_env;
    };

    template<typename Ty>
    RLTargetSelector(ppo::model::Aircraft* aircraft, ppo::model::SNA* sna, Ty& env) :
        p_aircraft{ aircraft },
        p_sna{ sna },
        p_environment{ std::make_shared<EnvironmentWrapper<Ty>>(env) }
    {

    };

    static constexpr unsigned char MAX_TRACKS{ 10 };

    enum class eFeatures : unsigned int
    {
        NorthRange,
        EastRange,
        Range,
        ClosingSpeed,
        CosBearing,
        SinBearing,
        Side,
        IsObjective,
        IsMissile,
        IsAircraft,
        Importancy,
        Available,

        _First = NorthRange,
        _Last = Available,
        _Size = _Last - _First + 1
    };

    using STATES  = std::array<float, MAX_TRACKS* static_cast<unsigned int>(eFeatures::_Size)>;
    using ACTIONS = std::array<float, MAX_TRACKS>;

    void SetNextAction(ACTIONS const& action) noexcept;
    void SetNewState  (STATES& state) noexcept;

private:
    std::shared_ptr<IEnvironment>   p_environment{ nullptr };
    ppo::model::Aircraft*           p_aircraft{ nullptr };
    ppo::model::SNA*                p_sna{ nullptr };  
};

} /* ppo::mission */