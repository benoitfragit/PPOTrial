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
    Impl(std::string const& datapath, std::string const& name, ePlayerSide side,  PlayerId playerId, std::shared_ptr<IEnvironment> pEnv)   :
        p_env{pEnv},
        m_name{ name }
    {
        m_playerId = playerId;
        m_side     = side;

        std::filesystem::path filepath{ datapath };
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

    void Init(void);
    void OnInit(void);
    void Run(void) noexcept;
    void UpdateDoubleBuffer(void);
    void ComputeFktData(void) noexcept;

    template<typename Ar>
    void serialize(Ar& ar)
    {
        auto section = ar.begin_section(m_name.c_str());

        section("MaxNoEscapeZoneRange", m_maxNoescapeZone);
        section("MaxFuel",              m_maxfuel);
        section("MaxMissile",           m_maxnbmissiles);
        section("MaxSpeed",             m_maxSpeed);
        section("MinSpeed",             m_minSpeed);
        section("MaxShootAngle",        m_maxShootAngle);
        section("MaxRotAcceleration",m_maxAccRot); 
        section("MaxAcceleration", m_maxAcc);
    };

    std::shared_ptr<IEnvironment>   p_env   { nullptr };
    sPlayerState                    m_buffers[2];
    std::atomic<uint32_t>           m_index { 0 };
    std::string                     m_name  { "unknown" };

    static inline sPlayerState const EMPTY_PLAYER_STATE{};
};

void
Player::Impl::UpdateDoubleBuffer(void)
{
   const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    m_buffers[next].m_maxSpeed         = m_maxSpeed;
    m_buffers[next].m_minSpeed         = m_minSpeed;
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
    m_buffers[next].m_fuel             = m_fuel              ;
    m_buffers[next].m_fuelconsumption  = m_fuelconsumption;
    m_buffers[next].m_escapezone       = m_escapezone        ;
    m_buffers[next].m_noescapezone     = m_noescapezone      ;
    m_buffers[next].m_maxfuel          = m_maxfuel           ;
    m_buffers[next].m_nbmissiles       = m_nbmissiles        ;
    m_buffers[next].m_maxnbmissiles    = m_maxnbmissiles     ;
    m_buffers[next].m_destroyed        = m_destroyed         ;
    m_buffers[next].m_playerId         = m_playerId;
    m_buffers[next].m_side             = m_side;
    m_buffers[next].m_maxShootAngle    = m_maxShootAngle     ;
    m_buffers[next].m_closingspeed     = m_closingspeed;
    m_buffers[next].m_target_los       = m_target_los;
    m_buffers[next].m_target_los_rate  = m_target_los_rate;
    m_buffers[next].m_maxAccRot        = m_maxAccRot;
    m_buffers[next].m_maxAcc           = m_maxAcc;
    m_buffers[next].m_accpsi           = m_accpsi;
    m_buffers[next].m_acc              = m_acc;
    m_index.store(next, std::memory_order_release);
}

void
Player::Impl::Init(void)
{
    if (p_env == nullptr)
        return;

    RandomGenerator generator{};

    generator.registerAttr(m_rotation, 0.F, static_cast<float>(2.0 * PI));

    if (m_minSpeed < m_maxSpeed)
        generator.registerAttr(m_speed, m_minSpeed + 50.F, m_maxSpeed);

    generator.registerAttr(m_accpsi, -m_maxAccRot / 2.F, m_maxAccRot / 2.F);
    generator.registerAttr(m_acc, -m_maxAcc / 2.F, m_maxAcc / 2.F);

    generator.registerAttr(m_x, -p_env->GetMaxNorth() / 2.F, p_env->GetMaxNorth() / 2.F);
    generator.registerAttr(m_y, -p_env->GetMaxEast() / 2.F,  p_env->GetMaxEast()  / 2.F);

    generator.generate();
    
    OnInit();
}

void
Player::Impl::OnInit(void)
{
    m_pk = 0.F;
    m_destroyed = false;
    m_escapezone = 2.F * m_maxNoescapeZone;
    m_noescapezone = m_maxNoescapeZone;
    m_fuel = m_maxfuel;
    m_nbmissiles = m_maxnbmissiles;
    m_target_x = 0.F;
    m_target_y = 0.F;
    m_dx = m_speed * cos(m_rotation);
    m_dy = m_speed * sin(m_rotation);

    m_closingspeed = 0.F;
    m_target_los = 0.F;
    m_target_los_rate = 0.F;

    UpdateDoubleBuffer();
}

inline static float 
sigmoid(float x)
{
    return 1.F / (1.F + std::exp(-x));
}

void
Player::Impl::ComputeFktData(void) noexcept
{
    if (p_env == nullptr)
        return;

    if (ppo::model::Player* const target{ p_env->GetPlayerById(m_target) };
        target != nullptr)
    {
        m_target_x = target->GetNorthPosition();
        m_target_y = target->GetEastPosition();
        m_target_dx = target->GetNorthSpeed();
        m_target_dy = target->GetEastSpeed();

        float const dx{ m_target_x - m_x };
        float const dy{ m_target_y - m_y };

        float const d{ std::sqrt(dx * dx + dy * dy) };

        float const ux{ dx / (d + 1e-6F) };
        float const uy{ dy / (d + 1e-6F) };

        // 1. computing the kill probability
        float az = atan2(uy, ux);
        az = wrapMinusPi2Pi(az);
        az = wrapMinusPi2Pi(az - m_rotation);

        float const k0{ 5.F / m_maxShootAngle };
        float const pkangle{ sigmoid(k0 * (m_maxShootAngle - std::fabs(az))) };

        float const dmid{ (m_noescapezone + m_escapezone) / 2.F };
        float const deltad{ m_escapezone - m_noescapezone + 1e-6F };
        float const kd{ 8.F / deltad };
        float const pkrange{ sigmoid(kd * (dmid - d)) };

        m_pk = pkrange * pkangle;

        // 2. computing the closing speed
        float const vrx{ m_target_dx - m_dx };
        float const vry{ m_target_dy - m_dy };

        m_closingspeed = -(vrx * ux + vry * uy);

        // 3. compute new LOS value
        float const LOS{ (ux * vry - uy * vrx) / d };

        // 4. update the LOS rate
        m_target_los_rate = LOS - m_target_los;

        // 5. compute the LOS
        m_target_los = LOS;
    }
    else
    {
        m_target = 0;
        m_target_dx = 0.F;
        m_target_dy = 0.F;
        m_target_los = 0.F;
        m_target_los_rate = 0.F;
        m_target_x = 0.F;
        m_target_y = 0.F;
        m_closingspeed = 0.F;
    }
}

void
Player::Impl::Run(void) noexcept
{
    if (p_env == nullptr)
        return;

    m_pk = 0.F;

    if (!m_destroyed)
    {
        float const dt{ p_env->GetTimeStep() };
        float const prevRotation{ m_rotation };
        m_rotation += m_accpsi * dt;
        m_speed    += m_acc * dt ;

        if (m_maxSpeed < m_speed)
            m_speed = m_maxSpeed;

        m_dx = m_speed * cos(m_rotation);
        m_dy = m_speed * sin(m_rotation);

        // new position according to rotation and speeds
        m_x += m_dx * dt;
        m_y += m_dy * dt;

        float const diffRotation{ std::fabs(static_cast<float>(wrapMinusPi2Pi(m_rotation - prevRotation))) };
        float const w{ diffRotation / dt };
        float const consumption{ static_cast<float>(C0
                                                  + CV3 * m_speed*m_speed*m_speed 
                                                  + CROT * w*w)};

        m_fuelconsumption = consumption * dt;

        m_fuel -= m_fuelconsumption;
        m_fuel = std::max(m_fuel, 0.F);

        ComputeFktData();

        if (m_fuel < std::numeric_limits<float>::epsilon()
        || (2.F*m_speed) < m_minSpeed)
            m_destroyed = true;
    }

    UpdateDoubleBuffer();
}

Player::Player(std::string const& datapath, std::string const& name, ePlayerSide side, PlayerId playerId, std::shared_ptr<IEnvironment> pEnv):
   p_impl{std::make_unique<Impl>(datapath, name, side, playerId, pEnv)}
{

}

Player::~Player(void) = default;

void 
Player::Init(void) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->Init();
}

void
Player::Run(void) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->Run();
}

void
Player::SetDestroyed(bool destroyed)                           noexcept 
{
    if (p_impl == nullptr)
        return;

    p_impl->m_destroyed = destroyed;
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
Player::GetFuelConsumption(void)                const noexcept
{
    if (p_impl == nullptr)
        return 0.F;
    return p_impl->m_fuelconsumption;
}

float
Player::GetMaxSpeed(void)    const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_maxSpeed;
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

PlayerId                                          
Player::GetTarget(void) const noexcept
{
    if (p_impl == nullptr)
        return ULLONG_MAX;

    return p_impl->m_target;
}

float
Player::GetCLosingSpeed(void) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_closingspeed;
}

float
Player::GetTargetLOS(void) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_target_los;
}

float 
Player::GetTargetLOSRate(void) const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_target_los_rate;
}

float
Player::GetTargetNorthPosition(void)  const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_target_x;
}

float
Player::GetTargetEastPosition(void)   const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_target_y;
}

float
Player::GetTargetNorthSpeed(void)     const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_target_dx;
}

float
Player::GetTargetEastSpeed(void)      const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_target_dy;
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
Player::GetMaxRotSpeed(void)                                     const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_maxAccRot;
}

float 
Player::GetMaxLongitudinalAcc(void)                              const noexcept
{
    if (p_impl == nullptr)
        return 0.F;

    return p_impl->m_maxAcc;
}

void  
Player::SetRotSpeed(float value) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_accpsi = value;
}

void                                             
Player::SetLongitudinalAcc(float value) noexcept 
{
    if (p_impl == nullptr)
        return;

    p_impl->m_acc = value;
}

CheckPointBuffer
Player::GetCheckpoint(void) noexcept
{
    CheckPointBuffer buffer{};
    if (p_impl == nullptr
    ||  p_impl->p_env == nullptr)
        return buffer;

    RandomGenerator generator{};

    generator.registerAttr(buffer[0], -p_impl->p_env->GetMaxNorth() / 2.F, p_impl->p_env->GetMaxNorth()/2.F);
    generator.registerAttr(buffer[1], -p_impl->p_env->GetMaxEast()  / 2.F, p_impl->p_env->GetMaxEast() /2.F);
    generator.registerAttr(buffer[2], 0.F, static_cast<float>(2.0 * PI));
    if (p_impl->m_minSpeed +50.F < p_impl->m_maxSpeed)
        generator.registerAttr(buffer[3], p_impl->m_minSpeed + 50.F, p_impl->m_maxSpeed);
    generator.registerAttr(buffer[4], -p_impl->m_maxAccRot / 2.F, p_impl->m_maxAccRot / 2.F);
    generator.registerAttr(buffer[5], -p_impl->m_maxAcc / 2.F,    p_impl->m_maxAcc / 2.F);

    generator.generate();
   
    return buffer;
}

void  
Player::SetCheckpoint(CheckPointBuffer const& buffer) noexcept
{
    if (p_impl == nullptr)
        return;

    p_impl->m_x         = buffer[0];
    p_impl->m_y         = buffer[1];
    p_impl->m_rotation  = buffer[2];
    p_impl->m_speed     = buffer[3];
    p_impl->m_accpsi    = buffer[4];
    p_impl->m_acc       = buffer[5];

    p_impl->OnInit();
}

}