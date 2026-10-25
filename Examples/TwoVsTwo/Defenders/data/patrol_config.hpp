#pragma once

#include "patrol_core.hpp"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace ppo::examples::twovstwo::defenders {

inline Parameters LoadPatrolParameters(std::filesystem::path const& settings)
{
    auto read = [](std::filesystem::path const& path) {
        std::ifstream stream{path};
        if (!stream)
            throw std::runtime_error("Cannot open patrol configuration: " + path.string());
        nlohmann::json document;
        stream >> document;
        if (document.at("formatVersion").get<int>() != 2)
            throw std::runtime_error("Unsupported patrol configuration version");
        return document;
    };
    auto const world = read(settings / "Environment.json").at("environment");
    auto const flight = read(settings / "Platforms/Mobile/defender_car.json")
        .at("parameters").at("FlightModel");
    Parameters parameters{};
    parameters.halfNorth = 0.5F * world.at("maxNorth").get<float>();
    parameters.halfEast = 0.5F * world.at("maxEast").get<float>();
    parameters.dt = world.at("timeStep").get<float>();
    parameters.maxSpeed = flight.at("MaxSpeed").get<float>();
    parameters.maxAcceleration = flight.at("MaxAcceleration").get<float>();
    parameters.maxRotationRate = flight.at("MaxRotAcceleration").get<float>();
    if (!Valid(parameters))
        throw std::invalid_argument("Invalid patrol world or vehicle parameters");
    return parameters;
}

} // namespace ppo::examples::twovstwo::defenders
