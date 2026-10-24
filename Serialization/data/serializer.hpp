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
#include <string>
#include <functional>
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
 * @brief Singleton service used to load/save JSON configuration files.
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
    * @brief Sets the base directory used for registered files.
    * @param path Root configuration folder.
    */
    void SetPath(std::filesystem::path path);

    /**
    * @brief Callback invoked when loading a JSON document.
    */
    using JsonLoadFunction = std::function<void(json& ar)>;
    /**
    * @brief Callback invoked when saving a JSON document.
    */
    using JsonSaveFunction = std::function<void(json& ar)>;

    /**
    * @brief Registers load and save callbacks for one file.
    * @param path File path relative to the configured root.
    * @param load Load callback, optional.
    * @param save Save callback, optional.
    */
    void Register(std::filesystem::path path, JsonLoadFunction load, JsonSaveFunction save);

    /**
    * @brief Loads every registered configuration file.
    */
    void Load(void);
    /**
    * @brief Saves every registered configuration file.
    */
    void Save(void);

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
    * @brief Hidden implementation holding callback maps and path state.
    */
    struct Impl;
    /**
    * @brief Pointer to the serializer implementation.
    */
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::serialization */
