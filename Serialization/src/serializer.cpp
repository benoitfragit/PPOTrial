/**
* @file Serialization/src/serializer.cpp
* @brief Implementation for serializer.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "serializer.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <sstream>

namespace ppo::serialization {

namespace {

/**
 * @brief Emit a structured configuration diagnostic log line.
 * @param code Machine-friendly event code.
 * @param file File concerned by the event.
 * @param message Human-readable details.
 */
void
LogConfigEvent(std::string const& code, std::filesystem::path const& file, std::string const& message)
{
    std::cerr << "[" << code << "] file='" << file.string() << "' message='" << message << "'" << std::endl;
}

} // namespace

/**
* @brief Parses one JSON file with robust error handling.
*/
json 
Serializer::Parse(std::filesystem::path file)
{
    std::ifstream stream(file);

    if (!stream.is_open())
    {
        LogConfigEvent("CFG_PARSE_FAILURE", file, "unable to open file");
        return json::object();
    }

    try
    {
        return json::parse(stream);
    }
    catch (std::exception const& exception)
    {
        LogConfigEvent("CFG_PARSE_FAILURE", file, exception.what());
        return json::object();
    }
}

/**
* @brief Rejects configurations outside the sole supported format.
*/
bool
Serializer::ValidateFormatVersion(std::filesystem::path const& file, json const& content)
{
    if (!content.is_object()
        || !content.contains("formatVersion")
        || !content.at("formatVersion").is_number_integer()
        || content.at("formatVersion").get<int>() != kConfigFormatVersion)
    {
        LogConfigEvent("CFG_FORMAT_MISMATCH", file,
                       "configuration requires integer formatVersion = 2");
        return false;
    }
    return true;
}

/**
* @brief Hidden serializer implementation storing root path and callback registries.
*/
struct Serializer::Impl 
{
    std::filesystem::path                                                    m_path{};
    std::unordered_map<std::filesystem::path, std::vector<JsonLoadFunction>> m_loadfunctions{};
    std::unordered_map<std::filesystem::path, std::vector<JsonSaveFunction>> m_savefunctions{};
    // A malformed input must not be replaced with defaults at shutdown.
    std::unordered_set<std::filesystem::path> m_failedLoads{};
};

/**
* @brief Creates the serializer implementation.
*/
Serializer::Serializer(void) :
    p_impl{ std::make_unique<Impl>() }
{
}

/**
* @brief Returns the serializer singleton.
*/
Serializer& 
Serializer::GetInstance(void)
{
    static Serializer instance{};
    return instance;
}

/**
* @brief Destroys the serializer implementation.
*/
Serializer::~Serializer(void) = default;

/**
* @brief Sets the root path used to resolve registered files.
*/
void 
Serializer::SetPath(std::filesystem::path path)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_path = path;
}

/**
* @brief Registers callbacks associated with one JSON file.
*/
void 
Serializer::Register(std::filesystem::path path, JsonLoadFunction load, JsonSaveFunction save)
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

/**
* @brief Loads all registered files and dispatches the parsed JSON to callbacks.
*/
void
Serializer::Load(void)
{
    if (p_impl == nullptr
    ||  p_impl->m_path.empty()
    || !std::filesystem::exists(p_impl->m_path))
        return;

    p_impl->m_failedLoads.clear();

    for (auto& [filepath, loads] : p_impl->m_loadfunctions)
    {
        if (!std::filesystem::exists(filepath))
        {
            LogConfigEvent("CFG_MISSING_FILE", filepath, "file not found");
            p_impl->m_failedLoads.insert(filepath);
            continue;
        }

        auto content = Parse(filepath.string());

        if (!ValidateFormatVersion(filepath, content))
        {
            p_impl->m_failedLoads.insert(filepath);
            continue;
        }

        for (auto& load : loads)
        {
            try
            {
                load(content);
            }
            catch (std::exception const& exception)
            {
                p_impl->m_failedLoads.insert(filepath);
                LogConfigEvent("CFG_LOAD_EXCEPTION", filepath, exception.what());
            }
        }
    }
}

/**
* @brief Builds JSON documents from callbacks and writes them to disk.
*/
void
Serializer::Save(void)
{
    if (p_impl == nullptr
    ||  p_impl->m_path.empty())
        return;

    if (!std::filesystem::exists(p_impl->m_path))
        std::filesystem::create_directories(p_impl->m_path);

    // Create one fresh document per registered save target.
    for (auto& [filepath, saves] : p_impl->m_savefunctions)
    {
        if (p_impl->m_failedLoads.count(filepath) != 0U)
        {
            LogConfigEvent("CFG_SAVE_SKIPPED", filepath,
                           "input configuration failed to load; original file preserved");
            continue;
        }

        json doc{};

        for (auto& save : saves)
            save(doc);

        doc["formatVersion"] = kConfigFormatVersion;

        std::ofstream(filepath) << doc.dump(4);
    }
}

}
