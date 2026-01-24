#pragma once

#include <type_traits>
#include <functional>
#include <random>
#include <vector>

namespace ppo::model {

constexpr double PI                      = 3.1415926536;
constexpr double TWOPI                   = 2.0 * PI;
constexpr double HALFPI                  = PI / 2.0;

constexpr double DEG_TO_RAD              = PI /180.0;

constexpr float  KT_TO_MS                = 0.51444444F;
constexpr float  MS_TO_KT                = 1.94384F;

constexpr double EULER                   = 2.7182818284;

constexpr float  MIN_VELOCITY            = 100.F * KT_TO_MS;
constexpr float  C0                      = 0.6F; // Kg.s-1
constexpr float  CV3                     = 8e-6F; // Kg.m-3
constexpr float  CROT                    = 1.5F;    //kg.s-1.rad2
constexpr double EPS                     = 1e-8;

double wrapMinusPi2Pi(double az);

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

    // Sp�cialisation pour bool
    template<typename T>
    std::enable_if_t<std::is_same_v<T, bool>>
        registerAttr(T& attr, double probability = 0.5) {
        m_generators.push_back([this, &attr, probability]() {
            std::bernoulli_distribution dist(probability);
            attr = dist(m_rng);
            });
    }

    // G�n�re toutes les valeurs al�atoires
    void generate() {
        for (auto& gen : m_generators) {
            gen();
        }
    }

    // R�initialise les g�n�rateurs enregistr�s
    void clear() {
        m_generators.clear();
    }
};

}