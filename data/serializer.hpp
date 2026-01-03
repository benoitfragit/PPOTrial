#pragma once
#include "serialization.hpp"

#include <filesystem>
#include <functional>
#include <memory>

namespace ppo::serialization {

class Serializer {
public:
    static Serializer& GetInstance(void);
    
    virtual ~Serializer(void);

    Serializer(Serializer&) = delete;
    Serializer(Serializer&&) = delete;
    Serializer& operator=(Serializer&) = delete;
    Serializer& operator=(Serializer&&) = delete;

    void SetPath(std::filesystem::path path);

    using TomlLoadFunction = std::function<void(TomlReadArchive& ar)>;
    using TomlSaveFunction = std::function<void(TomlWriteArchive& ar)>;

    void Register(TomlLoadFunction load, TomlSaveFunction save);

    void Load(void);
    void Save(void);

private:
    Serializer(void);

    struct Impl;
    std::unique_ptr<Impl> p_impl;
};

} /* ppo::serialization */