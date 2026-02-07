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

    static json Parse(std::filesystem::path file);

private:
    Serializer(void);

    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::serialization */