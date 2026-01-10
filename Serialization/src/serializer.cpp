#include "serializer.hpp"
#include "serialization.hpp"
#include <unordered_map>
#include <vector>

namespace ppo::serialization {

struct Serializer::Impl 
{
    std::filesystem::path                                                    m_path{};
    std::unordered_map<std::filesystem::path, std::vector<TomlLoadFunction>> m_loadfunctions{};
    std::unordered_map<std::filesystem::path, std::vector<TomlSaveFunction>> m_savefunctions{};
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
Serializer::Register(std::filesystem::path path, TomlLoadFunction load, TomlSaveFunction save)
{
    if (p_impl == nullptr)
        return;

    std::filesystem::path p{ p_impl->m_path / path };
    p.make_preferred();

    if (load != nullptr)
    {
        if (auto Iter = p_impl->m_loadfunctions.find(p);
            Iter == p_impl->m_loadfunctions.end())
        {
            p_impl->m_loadfunctions[p] = { load };
        }
        else
        {
            Iter->second.push_back(load);
        }
    }

    if (save != nullptr)
    {
        if (auto Iter = p_impl->m_savefunctions.find(p);
            Iter == p_impl->m_savefunctions.end())
        {
            p_impl->m_savefunctions[p] = { save };
        }
        else
        {
            Iter->second.push_back(save);
        }
    }
}

void
Serializer::Load(void)
{
    if (p_impl == nullptr
    ||  p_impl->m_path.empty()
    || !std::filesystem::exists(p_impl->m_path))
        return;

    for (auto& [filepath, loads] : p_impl->m_loadfunctions)
    {
        if (!std::filesystem::exists(filepath))
            continue;

        auto tbl = toml::parse_file(filepath.c_str());
        TomlReadArchive ar{ tbl };

        for (auto& load : loads)
            load(ar);
    }
}

void
Serializer::Save(void)
{
    if (p_impl == nullptr
    ||  p_impl->m_path.empty())
        return;

    if (!std::filesystem::exists(p_impl->m_path))
        std::filesystem::create_directories(p_impl->m_path);

    // crearte an archive
    for (auto& [filepath, saves] : p_impl->m_savefunctions)
    {
        toml::table tbl;
        TomlWriteArchive ar{ tbl };

        for (auto& save : saves)
            save(ar);

        std::ofstream(filepath) << tbl;
    }
}

}