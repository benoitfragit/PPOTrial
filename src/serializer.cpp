#include "serializer.hpp"
#include "serialization.hpp"
#include <vector>

namespace ppo::serialization {

struct Serializer::Impl 
{
    std::filesystem::path         m_path{};
    std::vector<TomlLoadFunction> m_loadfunctions{};
    std::vector<TomlSaveFunction> m_savefunctions{};
};

Serializer::Serializer(void) :
    p_impl{ std::make_unique<Impl>() }
{
}

Serializer& 
Serializer::GetInstance(void)
{
    static Serializer instance{};
    return instance;
}

Serializer::~Serializer(void) = default;

void 
Serializer::SetPath(std::filesystem::path path)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_path = path;
}

void 
Serializer::Register(TomlLoadFunction load, TomlSaveFunction save)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_loadfunctions.push_back(load);
    p_impl->m_savefunctions.push_back(save);
}

void
Serializer::Load(void)
{
    if (p_impl == nullptr
    ||  p_impl->m_path.empty()
    || !std::filesystem::exists(p_impl->m_path))
        return;

    auto tbl = toml::parse_file(p_impl->m_path.c_str());
    TomlReadArchive ar{ tbl };

    for (auto& load : p_impl->m_loadfunctions)
        load(ar);
}

void
Serializer::Save(void)
{
    if (p_impl == nullptr
    ||  p_impl->m_path.empty())
        return;

    if (!std::filesystem::exists(p_impl->m_path))
    {
        std::filesystem::path parent_path{ p_impl->m_path.parent_path() };

        if (!std::filesystem::exists(parent_path))
            std::filesystem::create_directories(parent_path);
    }

    // crearte an archive
    toml::table tbl;
    TomlWriteArchive ar{ tbl };

    for (auto& save : p_impl->m_savefunctions)
        save(ar);

    std::ofstream(p_impl->m_path) << tbl;
}

}