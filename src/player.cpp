#include "player.hpp"
#include "utils.hpp"
#include "serializer.hpp"

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

        ppo::serialization::Serializer& serializer{ ppo::serialization::Serializer::GetInstance() };
        serializer.Register(
            [this](ppo::serialization::TomlReadArchive& ar)
            {
                serialize(ar);

                UpdateDoubleBuffer();
            },
            [this](ppo::serialization::TomlWriteArchive& ar)
            {
                serialize(ar);
            }
        );
    };

    void Init(sEnvironmentState const& env);
    void Run(float dt) noexcept;
    void UpdateDoubleBuffer(void);
    void ComputeFktKillProbability(void) noexcept;

    template<typename Ar>
    void serialize(Ar& ar)
    {
        auto section = ar.begin_section(m_name.c_str());

        section("MaxNoEscapeZoneRange", m_maxNoescapeZone);
        section("MaxFuel",              m_maxfuel);
        section("MaxMissile",           m_maxnbmissiles);
        section("MaxSpeed",             m_maxSpeed);
        section("MaxRotationAngle",     m_rotationAngle);
        section("MaxShootAngle",        m_maxShootAngle);
    };

    sPlayerState            m_buffers[2];
    std::atomic<uint32_t>   m_index{ 0 };
    std::string             m_name{ "unknown" };

    static inline sPlayerState const EMPTY_PLAYER_STATE{};
};

void
Player::Impl::UpdateDoubleBuffer(void)
{
   const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    m_buffers[next].m_maxSpeed         = m_maxSpeed;
    m_buffers[next].m_maxNoescapeZone  = m_maxNoescapeZone;
    m_buffers[next].m_target           = m_target            ;
    m_buffers[next].m_dx               = m_dx                ;
    m_buffers[next].m_dy               = m_dy                ;
    m_buffers[next].m_x                = m_x                 ;
    m_buffers[next].m_y                = m_y                 ;
    m_buffers[next].m_target_x         = m_target_x          ;
    m_buffers[next].m_target_y         = m_target_y          ;
    m_buffers[next].m_target_dx        = m_target_dx;
    m_buffers[next].m_target_dy        = m_target_dy;
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
    m_buffers[next].m_maxShootAngle    = m_maxShootAngle     ;

    m_index.store(next, std::memory_order_release);
}

void
Player::Impl::Init(sEnvironmentState const& env)
{
    RandomGenerator generator{};

    generator.registerAttr(m_dx, -m_maxSpeed, m_maxSpeed);
    generator.registerAttr(m_dy, -m_maxSpeed, m_maxSpeed);

    generator.registerAttr(m_x, -env.m_maxNorth / 2.F, env.m_maxNorth / 2.F);
    generator.registerAttr(m_y, -env.m_maxEast / 2.F, env.m_maxEast / 2.F);

    generator.generate();

    m_pk            = 0.F;
    m_destroyed     = false;
    m_escapezone    = 2.F * m_maxNoescapeZone;
    m_noescapezone  = m_maxNoescapeZone;
    m_fuel          = m_maxfuel;
    m_nbmissiles    = m_maxnbmissiles;
    m_target_x      = 0.F;
    m_target_y      = 0.F;
    m_rotation      = wrapMinusPi2Pi(atan2(m_dy, m_dx));

    UpdateDoubleBuffer();
}

inline static float sigmoid(float x)
{
    return 1.F / (1.F + std::exp(-x));
}

void
Player::Impl::ComputeFktKillProbability(void) noexcept
{
    float const d{ static_cast<float>(sqrt(pow(m_x - m_target_x, 2.F) + pow(m_y - m_target_y, 2.F))) };
    float const ux{ (m_target_x - m_x) / d };
    float const uy{ (m_target_y - m_y) / d };

    float az = atan2(uy, ux);
    az = wrapMinusPi2Pi(az);
    az = wrapMinusPi2Pi(az - m_rotation);

    float const k0{ 5.F / m_maxShootAngle };
    float const pkangle{sigmoid(k0*(m_maxShootAngle - std::fabs(az))) };

    float const dmid{ (m_noescapezone + m_escapezone) / 2.F };
    float const deltad{ m_escapezone - m_noescapezone + 1e-6F };
    float const kd{ 8.F / deltad };
    float const pkrange{sigmoid(kd*(dmid - d))};
    
    m_pk = pkrange * pkangle;
}

void
Player::Impl::Run(float dt) noexcept
{
    m_pk = 0.F;

    if (!m_destroyed)
    {
        m_speed = sqrt(m_dx * m_dx + m_dy * m_dy);

        if (m_maxSpeed < m_speed)
        {
            float const factor{ m_maxSpeed / m_speed };

            m_speed = m_maxSpeed;

            m_dx *= factor;
            m_dy *= factor;
        }

        m_rotation = wrapMinusPi2Pi(atan2(m_dy, m_dx));

        //float const newRotation{ static_cast<float>(wrapMinusPi2Pi(atan2(m_dy, m_dx))) };
        //float const diffRotation{static_cast<float>(wrapMinusPi2Pi(newRotation - m_rotation)) };
        //
        //// bornage
        //float const maxRotationAngle{ m_rotationAngle * dt };
        //float const limitDiff{ maxRotationAngle * tanh(diffRotation / maxRotationAngle) };
        //
        //m_rotation = wrapMinusPi2Pi(m_rotation + limitDiff);
        //
        //m_dx = m_speed * std::cos(m_rotation);
        //m_dy = m_speed * std::sin(m_rotation);

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

        if (!m_destroyed)
            ComputeFktKillProbability();
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

    p_impl->Init(env);
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
Player::SetMaxNoEscapeZone(float maxNEZ)  noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_maxNoescapeZone = maxNEZ;
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

float 
Player::GetKillProbability(void) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_pk;
}

sPlayerState
Player::GetSnapshot() const noexcept
{
    if (p_impl == nullptr)
        return Impl::EMPTY_PLAYER_STATE;

    return p_impl->m_buffers[p_impl->m_index.load(std::memory_order_acquire)]; 
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

void 
Player::SetTargetSpeed(float north, float east) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_target_dx = north;
    p_impl->m_target_dy = east;
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

    return 1.0F - std::exp(-d / 8000.F);
}

float
Player::StateClosingSpeed(sEnvironmentState const& env) const noexcept
{ 
    if (p_impl == nullptr)
        return 0.F;

    float dx = p_impl->m_target_x - p_impl->m_x;
    float dy = p_impl->m_target_y - p_impl->m_y;

    float d = std::sqrt(dx * dx + dy * dy);

    float ux = dx / (d + 1e-6F);
    float uy = dy / (d + 1e-6F);
    
    float vrx = p_impl->m_dx - p_impl->m_target_dx;
    float vry = p_impl->m_dy - p_impl->m_target_dy;

    return (vrx * ux + vry * uy) / p_impl->m_maxSpeed;
}

float
Player::StateCosRotation(sEnvironmentState const& env) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return cos(p_impl->m_rotation);
}

float
Player::StateSinRotation(sEnvironmentState const& env) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return sin(p_impl->m_rotation);
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

    p_impl->m_dx = p_impl->m_maxSpeed * std::clamp(value, -1.F, 1.F);
}

void 
Player::ActionSetEastSpeed(float value) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_dy = p_impl->m_maxSpeed * std::clamp(value, -1.F, 1.F);
}

}