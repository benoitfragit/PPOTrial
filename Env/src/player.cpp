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

Player::Player( std::string const& datapath, 
                std::string const& name, 
                ePlayerSide side, 
                ePlayerType type, 
                PlayerId playerId, 
                std::shared_ptr<IEnvironment> pEnv)   :
    p_env{ pEnv },
    m_name{ name },
    m_playerId{playerId},
    m_side{side},
    m_type{type}
{
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
}

Player::~Player(void) = default;

void 
Player::Init(void) noexcept
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
    generator.registerAttr(m_y, -p_env->GetMaxEast() / 2.F, p_env->GetMaxEast() / 2.F);

    generator.generate();

    OnInit();
}

void
Player::UpdateDoubleBuffer(void)
{
   const uint32_t next = 1u - m_index.load(std::memory_order_relaxed);

    m_buffers[next].m_maxSpeed         = m_maxSpeed;
    m_buffers[next].m_minSpeed         = m_minSpeed;
    m_buffers[next].m_maxNoescapeZone  = m_maxNoescapeZone;
    m_buffers[next].m_dx               = m_dx                ;
    m_buffers[next].m_dy               = m_dy                ;
    m_buffers[next].m_x                = m_x                 ;
    m_buffers[next].m_y                = m_y                 ;
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
    m_buffers[next].m_type             = m_type;
    m_buffers[next].m_maxShootAngle    = m_maxShootAngle     ;
    m_buffers[next].m_maxAccRot        = m_maxAccRot;
    m_buffers[next].m_maxAcc           = m_maxAcc;
    m_buffers[next].m_accpsi           = m_accpsi;
    m_buffers[next].m_acc              = m_acc;
    m_index.store(next, std::memory_order_release);
}

void
Player::OnInit(void)
{
    m_destroyed = false;
    m_escapezone = 2.F * m_maxNoescapeZone;
    m_noescapezone = m_maxNoescapeZone;
    m_fuel = m_maxfuel;
    m_nbmissiles = m_maxnbmissiles;

    m_dx = m_speed * cos(m_rotation);
    m_dy = m_speed * sin(m_rotation);

    UpdateDoubleBuffer();
}

void
Player::Run(void) noexcept
{
    OnRun();

    UpdateDoubleBuffer();
}

void
Player::OnRun(void)
{
    if (p_env == nullptr)
        return;

    if (!m_destroyed)
    {
        float const dt{ p_env->GetTimeStep() };
        float const prevRotation{ m_rotation };
        m_rotation += m_accpsi * dt;
        m_speed += m_acc * dt;

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
                                                  + CV3 * m_speed * m_speed * m_speed
                                                  + CROT * w * w) };

        m_fuelconsumption = consumption * dt;

        m_fuel -= m_fuelconsumption;
        m_fuel = std::max(m_fuel, 0.F);

        if (m_fuel < std::numeric_limits<float>::epsilon()
            || (2.F * m_speed) < m_minSpeed)
            m_destroyed = true;
    }
}

void
Player::SetDestroyed(bool destroyed)                           noexcept 
{
    m_destroyed = destroyed;
}

std::string const&
Player::GetName(void) const noexcept
{
    return m_name;
}

bool
Player::GetDestroyed(void)                               const noexcept
{
    return m_destroyed;
}

float 
Player::GetRotation(void)                const noexcept
{
    return m_rotation;
}

float
Player::GetNorthPosition(void)               const noexcept
{
    return m_x;
}

float
Player::GetEastPosition(void)               const noexcept
{
    return m_y;
}

float
Player::GetFuelConsumption(void)                const noexcept
{
    return m_fuelconsumption;
}

float
Player::GetMaxSpeed(void)    const noexcept
{
    return m_maxSpeed;
}

float
Player::GetNorthSpeed(void) const noexcept
{
    return m_dx;
}

float
Player::GetEastSpeed(void)   const noexcept
{
    return m_dy;
}

PlayerId 
Player::GetPlayerId(void) const noexcept
{
    return m_playerId;
}

sPlayerState
Player::GetSnapshot() const noexcept
{
    return m_buffers[m_index.load(std::memory_order_acquire)]; 
}

float 
Player::GetMaxRotSpeed(void)                                     const noexcept
{
    return m_maxAccRot;
}

float 
Player::GetMaxLongitudinalAcc(void)                              const noexcept
{
    return m_maxAcc;
}

void  
Player::SetRotSpeed(float value) noexcept
{
    m_accpsi = value;
}

void                                             
Player::SetLongitudinalAcc(float value) noexcept 
{
    m_acc = value;
}

CheckPointBuffer
Player::GetCheckpoint(void) noexcept
{
    CheckPointBuffer buffer{};
    if (p_env == nullptr)
        return buffer;

    RandomGenerator generator{};

    generator.registerAttr(buffer[0], -p_env->GetMaxNorth() / 2.F, p_env->GetMaxNorth()/2.F);
    generator.registerAttr(buffer[1], -p_env->GetMaxEast()  / 2.F, p_env->GetMaxEast() /2.F);
    generator.registerAttr(buffer[2], 0.F, static_cast<float>(2.0 * PI));
    if (m_minSpeed +50.F < m_maxSpeed)
        generator.registerAttr(buffer[3], m_minSpeed + 50.F, m_maxSpeed);
    generator.registerAttr(buffer[4], -m_maxAccRot / 2.F, m_maxAccRot / 2.F);
    generator.registerAttr(buffer[5], -m_maxAcc / 2.F,    m_maxAcc / 2.F);

    generator.generate();
   
    return buffer;
}

void  
Player::SetCheckpoint(CheckPointBuffer const& buffer) noexcept
{
    m_x         = buffer[0];
    m_y         = buffer[1];
    m_rotation  = buffer[2];
    m_speed     = buffer[3];
    m_accpsi    = buffer[4];
    m_acc       = buffer[5];

    OnInit();
}

}