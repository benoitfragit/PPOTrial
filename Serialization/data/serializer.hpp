#pragma once
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <string>
#include <functional>
#include <memory>

namespace ppo::serialization {

using json = nlohmann::json;

/**
 * @brief Global schema version expected for configuration files.
 */
constexpr int kConfigSchemaVersion = 1;

/**
 * @brief Singleton service used to load/save JSON configuration files.
 */
class Serializer {
public:

    static Serializer& GetInstance(void);
    
    virtual ~Serializer(void);

    Serializer(Serializer&) = delete;
    Serializer(Serializer&&) = delete;
    Serializer& operator=(Serializer&) = delete;
    Serializer& operator=(Serializer&&) = delete;

    void SetPath(std::filesystem::path path);

    using JsonLoadFunction = std::function<void(json& ar)>;
    using JsonSaveFunction = std::function<void(json& ar)>;

    void Register(std::filesystem::path path, JsonLoadFunction load, JsonSaveFunction save);

    void Load(void);
    void Save(void);

    /**
     * @brief Parse a JSON file with robust error handling.
     * @param file Path to the JSON file.
     * @return Parsed JSON object, or an empty object when parsing fails.
     */
    static json Parse(std::filesystem::path file);

private:
    Serializer(void);

    /**
     * @brief Validate the `schemaVersion` field of a configuration payload.
     * @param file Source file path.
     * @param content JSON content to validate.
     * @return true when the payload is accepted, false otherwise.
     */
    static bool ValidateSchemaVersion(std::filesystem::path const& file, json& content);

    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::serialization */
