#include "player.hpp"
#include "utils.hpp"
#include "serialization.hpp"

#include <array>
#include <chrono>
#include <atomic>
#include <thread>
#include <type_traits>
#include <functional>
#include <random>

namespace ppo::model {

class RandomGenerator 
{
private:
    std::mt19937 m_rng;
    std::vector<std::function<void()>> m_generators;
public:
    RandomGenerator()
    {
        std::random_device rd;
        std::seed_seq seq{
            rd(), rd(), rd(), rd(),
            rd(), rd(), rd(), rd()
        };
        m_rng.seed(seq);
    }

    // Template pour les types entiers (int, long, short, etc.)
    template<typename T>
    std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>
        registerAttr(T& attr, T min = 0, T max = 100) {
        m_generators.push_back([this, &attr, min, max]() {
            std::uniform_int_distribution<T> dist(min, max);
            attr = dist(m_rng);
            });
    }

    // Template pour les types flottants (float, double, long double)
    template<typename T>
    std::enable_if_t<std::is_floating_point_v<T>>
        registerAttr(T& attr, T min = 0.0, T max = 1.0) {
        m_generators.push_back([this, &attr, min, max]() {
            std::uniform_real_distribution<T> dist(min, max);
            attr = dist(m_rng);
            });
    }

    // Spécialisation pour bool
    template<typename T>
    std::enable_if_t<std::is_same_v<T, bool>>
        registerAttr(T& attr, double probability = 0.5) {
        m_generators.push_back([this, &attr, probability]() {
            std::bernoulli_distribution dist(probability);
            attr = dist(m_rng);
            });
    }

    // Génère toutes les valeurs aléatoires
    void generate() {
        for (auto& gen : m_generators) {
            gen();
        }
    }

    // Réinitialise les générateurs enregistrés
    void clear() {
        m_generators.clear();
    }
};


struct Player::Impl :
    public sPlayerState
{
    Impl(std::string const& name, PlayerId playerId):
        m_name{ name }
    {
        m_playerId = playerId;
    };

    void Init(float x, float y);
    void Run(float dt) noexcept;
    void UpdateDoubleBuffer(void);

    template<typename Ar>
    void serialize(Ar& ar)
    {
        auto section = ar.begin_section(m_name.c_str());

        section("NoEscapeZoneRange", m_noescapeZoneRange);
        section("MaxFuel", m_maxfuel);
        section("MaxMissile", m_maxnbmissiles);
        section("MaxSpeed", m_maxSpeed);
        section("MaxRotationAngle", m_rotationAngle);
    };

    ppoArray4f              m_fightingArea{ 0.0, 0.0, 0.0, 0.0 };
    sPlayerState            m_buffers[2];
    std::atomic<uint32_t>   m_index{ 0 };
    std::string             m_name{ "unknown" };

    static inline sPlayerState const EMPTY_PLAYER_STATE{};
    static inline ppoArray2f const   EMPTY_PPO_ARRAY2F{};
};

void
Player::Impl::UpdateDoubleBuffer(void)
{
   const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    m_buffers[next].m_maxSpeed         = m_maxSpeed;
    m_buffers[next].m_noescapeZoneRange= m_noescapeZoneRange ;
    m_buffers[next].m_target           = m_target            ;
    m_buffers[next].m_dx               = m_dx                ;
    m_buffers[next].m_dy               = m_dy                ;
    m_buffers[next].m_x                = m_x                 ;
    m_buffers[next].m_y                = m_y                 ;
    m_buffers[next].m_target_x         = m_target_x          ;
    m_buffers[next].m_target_y         = m_target_y          ;
    m_buffers[next].m_speed            = m_speed             ;
    m_buffers[next].m_rotation         = m_rotation          ;
    m_buffers[next].m_rotationAngle    = m_rotationAngle     ;
    m_buffers[next].m_fuel             = m_fuel              ;
    m_buffers[next].m_fuelconsumption  = m_fuelconsumption;
    m_buffers[next].m_escapezone       = m_escapezone        ;
    m_buffers[next].m_noescapezone     = m_noescapezone      ;
    m_buffers[next].m_maxfuel          = m_maxfuel           ;
    m_buffers[next].m_nbmissiles       = m_nbmissiles        ;
    m_buffers[next].m_maxnbmissiles    = m_maxnbmissiles     ;
    m_buffers[next].m_destroyed        = m_destroyed         ;
    m_buffers[next].m_playerId         = m_playerId;

    m_index.store(next, std::memory_order_release);
}

void
Player::Impl::Init(float x, float y)
{
    m_destroyed = false;
    m_x = x;
    m_y = y;

    UpdateDoubleBuffer();
}

void
Player::Impl::Run(float dt) noexcept
{
    if (!m_destroyed)
    {
        m_speed = sqrt(m_dx * m_dx + m_dy * m_dy);

        if (m_maxSpeed < m_speed)
        {
            float const factor{ m_maxSpeed / m_speed };

            m_speed *= factor;

            m_dx *= factor;
            m_dy *= factor;
        }

        float const newRotation{ static_cast<float>(wrapMinusPi2Pi(atan2(m_dy, m_dx))) };
        float const diffRotation{static_cast<float>(wrapMinusPi2Pi(newRotation - m_rotation)) };
        
        // bornage
        float const maxRotationAngle{ m_rotationAngle * dt };
        if (maxRotationAngle < diffRotation)
        {
            m_dx = m_speed * cos(m_rotation + maxRotationAngle);
            m_dy = m_speed * sin(m_rotation + maxRotationAngle);
        }
        else if (diffRotation < -maxRotationAngle)
        {
            m_dx = m_speed * cos(m_rotation - maxRotationAngle);
            m_dy = m_speed * sin(m_rotation - maxRotationAngle);
        }

        m_rotation = wrapMinusPi2Pi(atan2(m_dy, m_dx));

        // new position according to rotation and speeds
        m_x += m_dx * dt;
        m_y += m_dy * dt;

        float const consumption{ static_cast<float>(MIN_CONSUMPTION
                              + m_speed * LINEAR_CONSUMPTION
                              + m_speed * m_speed * QUADRA_CONSUMPTION) };

        m_fuelconsumption = consumption * dt;

        m_fuel -= m_fuelconsumption;
        m_fuel = std::max(m_fuel, 0.F);

        if (m_fuel < std::numeric_limits<float>::epsilon()
        || m_speed < std::numeric_limits<float>::epsilon())
            m_destroyed = true;
    }

    UpdateDoubleBuffer();
}

Player::Player(std::string const& name, PlayerId playerId):
   p_impl{std::make_unique<Impl>(name, playerId)}
{

}

Player::~Player(void) = default;

void 
Player::Init(sEnvironmentState const& env) noexcept
{
    if (p_impl == nullptr)
        return;

    RandomGenerator generator{};

    float           northSpeed{ 0.F };
    float           eastSpeed{ 0.F };
    float           fuel{ 0.F };
    float           noescapezone{ 0.F };
    float           escapeZone{ 0.F };
    unsigned short  nbmissiles{ 0 };
    float           x{ 0.F };
    float           y{ 0.F };
    float       maxSpeed            { GetMaxSpeed() };
    auto const  maxFuel             { GetMaxFuel() };
    auto const& noEscapeZoneRanges  { GetNoEscapeZoneRange() };
    auto const  maxMissiles         { GetMaxMissiles() };
    ppoArray4f const fightingArea   { env.m_maxNorth / 2.F,
                                    -env.m_maxEast / 2.F,
                                    -env.m_maxNorth / 2.F,
                                    env.m_maxEast / 2.F };

    generator.registerAttr(northSpeed, -maxSpeed, maxSpeed);
    generator.registerAttr(eastSpeed,  -maxSpeed, maxSpeed);
    generator.registerAttr(fuel, maxFuel/5.F, maxFuel);

    generator.registerAttr(noescapezone, noEscapeZoneRanges[0], noEscapeZoneRanges[1]);
    generator.registerAttr(nbmissiles, (unsigned short)0, maxMissiles);
    generator.registerAttr(x, fightingArea[2], fightingArea[0]);
    generator.registerAttr(y, fightingArea[1], fightingArea[3]);

    generator.generate();

    generator.clear();

    generator.registerAttr(escapeZone, 1.5F * noescapezone, 3.F * noescapezone);

    generator.generate();

    p_impl->m_escapezone = 2.F*noEscapeZoneRanges[1];
    p_impl->m_noescapezone = noEscapeZoneRanges[1];
    p_impl->m_dx = northSpeed;
    p_impl->m_dy = eastSpeed;
    p_impl->m_fuel = maxFuel;
    p_impl->m_nbmissiles=nbmissiles;
    p_impl->m_target_x = 0.F;
    p_impl->m_target_y = 0.F;

    p_impl->Init(x, y);
}

void
Player::Run(float dt) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->Run(dt);
}

void 
Player::SetRemainingFuel(float power)         noexcept 
{
    if (p_impl == nullptr)
        return;

    p_impl->m_fuel = power;
}

void
Player::SetDestroyed(bool destroyed)                           noexcept 
{
    if (p_impl == nullptr)
        return;

    p_impl->m_destroyed = destroyed;
}

void
Player::SetMaxSpeed(float maxSpeed)            noexcept 
{
    if (p_impl == nullptr)
        return;

    p_impl->m_maxSpeed = maxSpeed;
}

void
Player::SetMaxFuel(float maxFuel)          noexcept 
{
    if (p_impl == nullptr)
        return;

    p_impl->m_maxfuel = maxFuel;
}

void
Player::SetNoEscapeZoneRange(float minNEZ, float maxNEZ)  noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_noescapeZoneRange[0] = minNEZ;
    p_impl->m_noescapeZoneRange[1] = maxNEZ;
}

void 
Player::SetEscapeZone(float escapezone)                         noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_escapezone = escapezone;
}

void 
Player::SetNoEscapeZone(float noescapezone)                     noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_noescapezone = noescapezone;
}

std::string const&
Player::GetName(void) const noexcept
{
    if (p_impl == nullptr)
    {
        static std::string empty{ "" };
        return empty;
    }

    return p_impl->m_name;
}

bool
Player::GetDestroyed(void)                               const noexcept
{
    if (p_impl == nullptr)
        return false;

    return p_impl->m_destroyed;
}

float 
Player::GetRotation(void)                const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_rotation;
}

float
Player::GetNorthPosition(void)               const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_x;
}

float
Player::GetEastPosition(void)               const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_y;
}

float
Player::GetRemainingFuel(void)          const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_fuel;
}

float
Player::GetFuelConsumption(void)                const noexcept
{
    if (p_impl == nullptr)
        return 0.F;
    return p_impl->m_fuelconsumption;
}

void    
Player::SetNbMissiles(unsigned short nb) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_nbmissiles = nb;
}

void
Player::SetMaxMissiles(unsigned short maxMissiles) noexcept
{
    if (p_impl == nullptr)
        return;
    p_impl->m_maxnbmissiles = maxMissiles;
}

unsigned short   
Player::GetRemainingMissiles(void) const noexcept
{
    if (p_impl == nullptr)
        return 0;

    return p_impl->m_nbmissiles;
}

float 
Player::GetEscapeZone(void)     const noexcept
{
    if (p_impl == nullptr)
        return 0;

    return p_impl->m_escapezone;
}

float                                         
Player::GetNoEscapeZone(void)   const noexcept
{
    if (p_impl == nullptr)
        return 0;

    return p_impl->m_noescapezone;
}

float                                         
Player::GetMaxFuel(void)        const noexcept
{
    if (p_impl == nullptr)
        return 0;

    return p_impl->m_maxfuel;
}

float
Player::GetMaxSpeed(void)    const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_maxSpeed;
}

unsigned short                                
Player::GetMaxMissiles(void)   const noexcept 
{
    if (p_impl == nullptr)
        return 0;

    return p_impl->m_maxnbmissiles;
}

float
Player::GetSpeed(void) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_speed;
}

float
Player::GetNorthSpeed(void) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_dx;
}

float
Player::GetEastSpeed(void)   const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_dy;
}

PlayerId 
Player::GetPlayerId(void) const noexcept
{
    if (p_impl == nullptr)
        return ULLONG_MAX;
    return p_impl->m_playerId;
}

sPlayerState
Player::GetSnapshot() const noexcept
{
    if (p_impl == nullptr)
        return Impl::EMPTY_PLAYER_STATE;

    return p_impl->m_buffers[p_impl->m_index.load(std::memory_order_acquire)]; 
}

void 
Player::SetFightingArea(float topleftnorth,
                        float toplefteast,
                        float bottomrightnorth,
                        float bottomrighteast) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_fightingArea[0] = topleftnorth;
    p_impl->m_fightingArea[1] = toplefteast;
    p_impl->m_fightingArea[2] = bottomrightnorth;
    p_impl->m_fightingArea[3] = bottomrighteast;
}

void
Player::SetTargetPosition(float north, float east) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_target_x = north;
    p_impl->m_target_y = east;
}

void     
Player::SetTarget(PlayerId targetId)       noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_target = targetId;
}

void
Player::SetNorthSpeed(float speed) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_dx = speed;
}

void
Player::SetEastSpeed(float speed) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_dy = speed;
}

PlayerId                                          
Player::GetTarget(void)              const noexcept
{
    if (p_impl == nullptr)
        return ULLONG_MAX;

    return p_impl->m_target;
}

ppoArray2f const& 
Player::GetNoEscapeZoneRange(void)   const noexcept
{
    if (p_impl == nullptr)
        return Impl::EMPTY_PPO_ARRAY2F;

    return p_impl->m_noescapeZoneRange;
}

float 
Player::GetTargetPositionNorth(void) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_target_x;
}

float 
Player::GetTargetPositionEast(void)  const noexcept
{
    if (p_impl == nullptr)
        return 0.F;
    return p_impl->m_target_y;
}

float
Player::StateNorthRange(sEnvironmentState const& env) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    float dx{ p_impl->m_target_x  - p_impl->m_x };
    float const dy{ p_impl->m_target_y  - p_impl->m_y};
    float const d{ sqrt(dx * dx + dy * dy) };
    dx /= std::max(d , 1e-6F);

    return dx;
}

float
Player::StateEastRange(sEnvironmentState const& env) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    float const dx{ p_impl->m_target_x - p_impl->m_x };
    float dy{ p_impl->m_target_y - p_impl->m_y };
    float const d{ sqrt(dx * dx + dy * dy) };
    dy /= std::max(d, 1e-6F);

    return dy;
}

float
Player::StateRange(sEnvironmentState const& env) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    float dx = p_impl->m_target_x - p_impl->m_x;
    float dy = p_impl->m_target_y - p_impl->m_y;

    float d = std::sqrt(dx * dx + dy * dy);

    float scale = p_impl->m_maxSpeed * env.m_dt * 20.f;

    return 1.0f - std::exp(-d / scale);
}

float 
Player::StateRemainingFuel(sEnvironmentState const& env) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;
    return p_impl->m_fuel / p_impl->m_maxfuel;
}

float
Player::StateRemainingMissiles(sEnvironmentState const& env) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;
    return static_cast<float>(p_impl->m_nbmissiles) / static_cast<float>(p_impl->m_maxnbmissiles);
}

void
Player::ActionSetNorthSpeed(float value)                 noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_dx = value * p_impl->m_maxSpeed;
}

void 
Player::ActionSetEastSpeed(float value) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_dy = value * p_impl->m_maxSpeed;
}

void 
Player::LoadFromFile(std::string const& filepath)
{
    if (p_impl == nullptr)
        return;

    std::string mypath{ filepath };
    mypath.append("/");
    mypath.append(p_impl->m_name);
    mypath.append(".toml");

    serialization::LoadFromToml(mypath, *p_impl);
}

void 
Player::SaveToFile(std::string const& filepath)
{
    if (p_impl == nullptr)
        return;

    std::string mypath{ filepath };
    mypath.append("/");
    mypath.append(p_impl->m_name);
    mypath.append(".toml");

    serialization::SaveToToml(mypath, *p_impl);
}

}