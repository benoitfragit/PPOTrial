/**
 * @file Framework/Simulation/World2D/data/restrictedzonegeometry.hpp
 * @brief Interface ou implémentation du composant restrictedzonegeometry.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#pragma once

namespace ppo::model {

/** @brief Scale applied to the lethal radius for pseudo-LiDAR warning. */
inline constexpr float kRestrictedZoneWarningRadiusScale = 1.10F;

/** @brief Two-dimensional position using north/east coordinates in metres. */
struct PlanarPosition final
{
    float north{};
    float east{};
};

/**
 * @brief Computes the documented pseudo-LiDAR warning radius.
 * @param lethalRadius Deterministic lethal radius in metres.
 * @return Warning radius in metres, or zero for an invalid radius.
 */
float RestrictedZoneWarningRadius(float lethalRadius) noexcept;

/**
 * @brief Tests whether a point is inside a circular restricted zone.
 * @param point Position tested in metres.
 * @param centre Zone centre in metres.
 * @param lethalRadius Inclusive lethal radius in metres.
 * @return True when the point lies on or inside the lethal circle.
 */
bool IsInsideRestrictedZone(PlanarPosition point,
                            PlanarPosition centre,
                            float lethalRadius) noexcept;

/**
 * @brief Tests a swept mobile trajectory against a circular restricted zone.
 * @param previous Mobile centre before the simulation step, in metres.
 * @param current Mobile centre after the simulation step, in metres.
 * @param centre Stationary zone centre in metres.
 * @param lethalRadius Inclusive lethal radius in metres.
 * @return True when either endpoint or the segment penetrates the zone.
 */
bool CrossesRestrictedZone(PlanarPosition previous,
                           PlanarPosition current,
                           PlanarPosition centre,
                           float lethalRadius) noexcept;

} // namespace ppo::model
