/**
* @file Serialization/data/serializer.hpp
* @brief Declarations for serializer.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <functional>
#include <cstdint>
#include <memory>

namespace ppo::serialization {

/**
* @brief JSON alias used across configuration services.
*/
using json = nlohmann::json;

/**
 * @brief Single supported application configuration format.
 */
constexpr int kConfigFormatVersion = 2;

/**
 * @brief Singleton service used to load JSON configuration files.
 */
class Serializer {
public:
    /**
    * @brief Returns the global serializer singleton.
    * @return Shared serializer instance.
    */
    static Serializer& GetInstance(void);
    
    /**
    * @brief Destroy the serializer implementation.
    */
    virtual ~Serializer(void);

    Serializer(Serializer&) = delete;
    Serializer(Serializer&&) = delete;
    Serializer& operator=(Serializer&) = delete;
    Serializer& operator=(Serializer&&) = delete;

    /**
    * @brief Sets the ordered folders searched for registered files.
    * @details An example keeps its own settings beside the shared ones: Load()
    *          resolves a registered relative file against the first folder that contains it,
    *          else against the first folder (so a missing file is reported there).
    * @param paths Search folders, highest priority first.
    */
    void SetSearchPaths(std::vector<std::filesystem::path> paths);

    /**
    * @brief Callback invoked when loading a JSON document.
    */
    using JsonLoadFunction = std::function<void(json& ar)>;

    /**
    * @brief Identifies one Register() call, to remove its callbacks with Unregister().
    */
    using Registration = std::uint64_t;

    /**
    * @brief Registers the load callback of one file.
    * @details A callback capturing an object must be removed with Unregister() before that
    *          object is destroyed: the singleton outlives it, and a process can build several
    *          environments in turn (a training environment, then an evaluation one).
    * @param path File path relative to the search folders (resolved by each Load).
    * @param load Load callback.
    * @return The handle of this registration (0, ignored by Unregister, for a null callback).
    */
    Registration Register(std::filesystem::path path, JsonLoadFunction load);

    /**
    * @brief Removes the callbacks of one registration; an unknown handle is ignored.
    * @param registration Handle returned by Register().
    */
    void Unregister(Registration registration) noexcept;

    /**
    * @brief Loads every registered configuration file.
    */
    void Load(void);

    /**
     * @brief Parse a JSON file with robust error handling.
     * @param file Path to the JSON file.
     * @return Parsed JSON object, or an empty object when parsing fails.
     */
    static json Parse(std::filesystem::path file);

private:
    /**
    * @brief Creates the serializer singleton instance.
    */
    Serializer(void);

    /**
     * @brief Require formatVersion 2 without mutating the input.
     */
    static bool ValidateFormatVersion(std::filesystem::path const& file, json const& content);

    /**
    * @brief Hidden implementation holding the callbacks and the search folders.
    */
    struct Impl;
    /**
    * @brief Pointer to the serializer implementation.
    */
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::serialization */
