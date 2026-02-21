/**
* @file Entities/data/utils.hpp
* @brief Declarations for utils.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once

#include <type_traits>
#include <functional>
#include <random>
#include <vector>
#include <atomic>

namespace ppo::model {

/**
* @brief Pi constant in radians.
*/
constexpr double PI = 3.1415926536;
/**
* @brief Full turn angle in radians.
*/
constexpr double TWOPI = 2.0 * PI;
/**
* @brief Half-turn angle in radians.
*/
constexpr double HALFPI = PI / 2.0;

/**
* @brief Degrees-to-radians conversion factor.
*/
constexpr double DEG_TO_RAD = PI / 180.0;

/**
* @brief Knots-to-meters-per-second conversion factor.
*/
constexpr float KT_TO_MS = 0.51444444F;
/**
* @brief Meters-per-second-to-knots conversion factor.
*/
constexpr float MS_TO_KT = 1.94384F;

/**
* @brief Euler number.
*/
constexpr double EULER = 2.7182818284;

/**
* @brief Minimum allowed platform speed.
*/
constexpr float MIN_VELOCITY = 100.F * KT_TO_MS;
/**
* @brief Constant fuel-consumption term.
*/
constexpr float C0 = 0.6F;
/**
* @brief Cubic-speed fuel-consumption coefficient.
*/
constexpr float CV3 = 8e-6F;
/**
* @brief Rotational fuel-consumption coefficient.
*/
constexpr float CROT = 1.5F;
/**
* @brief Generic numerical epsilon.
*/
constexpr double EPS = 1e-8;

/**
* @brief Wraps an angle to the range [-pi, pi].
* @param az Input angle in radians.
* @return Wrapped angle in radians.
*/
double wrapMinusPi2Pi(double az);

/**
* @brief Utility random generator with persistent registered attributes.
*/
class RandomGenerator
{
private:
    /**
    * @brief Underlying Mersenne Twister engine.
    */
    std::mt19937 m_rng;
    /**
    * @brief Deferred generators used to refresh registered attributes.
    */
    std::vector<std::function<void()>> m_generators;

public:
    /**
    * @brief Seeds the generator with entropy from the operating system.
    */
    RandomGenerator()
    {
        std::random_device rd;
        std::seed_seq seq{
            rd(), rd(), rd(), rd(),
            rd(), rd(), rd(), rd()
        };
        m_rng.seed(seq);
    }

    /**
    * @brief Registers an integral attribute for random generation.
    * @tparam T Integral attribute type.
    * @param attr Attribute to mutate.
    * @param min Minimum generated value.
    * @param max Maximum generated value.
    */
    template<typename T>
    std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>
    registerAttr(T& attr, T min = 0, T max = 100)
    {
        m_generators.push_back([this, &attr, min, max]()
        {
            std::uniform_int_distribution<T> dist(min, max);
            attr = dist(m_rng);
        });
    }

    /**
    * @brief Registers a floating-point attribute for random generation.
    * @tparam T Floating-point attribute type.
    * @param attr Attribute to mutate.
    * @param min Minimum generated value.
    * @param max Maximum generated value.
    */
    template<typename T>
    std::enable_if_t<std::is_floating_point_v<T>>
    registerAttr(T& attr, T min = 0.0, T max = 1.0)
    {
        m_generators.push_back([this, &attr, min, max]()
        {
            std::uniform_real_distribution<T> dist(min, max);
            attr = dist(m_rng);
        });
    }

    /**
    * @brief Registers a boolean attribute for Bernoulli generation.
    * @tparam T Boolean attribute type.
    * @param attr Attribute to mutate.
    * @param probability Probability of generating true.
    */
    template<typename T>
    std::enable_if_t<std::is_same_v<T, bool>>
    registerAttr(T& attr, double probability = 0.5)
    {
        m_generators.push_back([this, &attr, probability]()
        {
            std::bernoulli_distribution dist(probability);
            attr = dist(m_rng);
        });
    }

    /**
    * @brief Regenerates every registered attribute once.
    */
    void generate()
    {
        for (auto& gen : m_generators)
            gen();
    }

    /**
    * @brief Removes every registered attribute generator.
    */
    void clear()
    {
        m_generators.clear();
    }
};

} /* ppo::model */
