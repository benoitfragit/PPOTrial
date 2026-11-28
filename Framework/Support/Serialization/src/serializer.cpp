/**
* @file Serialization/src/serializer.cpp
* @brief Implementation for serializer.
* @author PPOTrial
* @date 2026-03-29
* @copyright Copyright (c) 2026 PPOTrial
* @license MIT
*/
#include "serializer.hpp"
#include <algorithm>
#include <unordered_map>
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
/**
 * @brief Lit et vérifie des données depuis l'état ou les entrées de Serializer.
 * @details Utilise des fichiers ou chemins.
 */
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
* @brief Hidden serializer implementation storing search folders and callback registries.
*/
struct Serializer::Impl 
{
    /** @brief One callback and the registration it came from. */
    template<typename Function>
    struct Entry
    {
        Registration registration{ 0U };
        Function     function{};
    };
    using LoadEntries = std::vector<Entry<JsonLoadFunction>>;

    std::vector<std::filesystem::path>                        m_paths{};
    std::unordered_map<std::filesystem::path, LoadEntries>    m_loadfunctions{};
    Registration                                              m_nextRegistration{ 1U };
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
/**
 * @brief Retourne une vue ou une valeur calculée depuis l'état courant de Serializer.
 * @details Utilise des fichiers ou chemins.
 */
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
* @brief Sets the ordered folders used to resolve registered files.
*/
void
Serializer::SetSearchPaths(std::vector<std::filesystem::path> paths)
{
    if (p_impl == nullptr)
        return;

    p_impl->m_paths = std::move(paths);
}

namespace {

/**
 * @brief Resolves a registered file against the search folders.
 * @return First existing candidate, else the candidate in the first folder.
 */
std::filesystem::path
ResolveInSearchPaths(std::vector<std::filesystem::path> const& roots, std::filesystem::path const& path)
{
    if (path.is_absolute() || roots.empty())
        return path;
    for (auto const& root : roots)
    {
        auto candidate = root / path;
        if (std::filesystem::exists(candidate))
            return candidate;
    }
    return roots.front() / path;
}

/**
 * @brief True when at least one search folder exists on disk.
 */
bool
AnySearchPathExists(std::vector<std::filesystem::path> const& roots)
{
    for (auto const& root : roots)
    {
        if (!root.empty() && std::filesystem::exists(root))
            return true;
    }
    return false;
}

} // namespace

namespace {

/**
* @brief Removes the entries of one registration from a callback map.
*/
template<typename Map>
void
EraseRegistration(Map& map, Serializer::Registration registration) noexcept
{
    for (auto& [filepath, entries] : map)
    {
        entries.erase(std::remove_if(entries.begin(), entries.end(),
                                     [registration](auto const& entry)
                                     { return entry.registration == registration; }),
                      entries.end());
    }
}

} // namespace

/**
* @brief Registers callbacks associated with one JSON file.
*/
Serializer::Registration
Serializer::Register(std::filesystem::path path, JsonLoadFunction load)
{
    if (p_impl == nullptr || load == nullptr)
        return 0U;

    // Resolved at Load time, so registering before SetSearchPaths is fine.
    path.make_preferred();

    Registration const registration{ p_impl->m_nextRegistration++ };
    p_impl->m_loadfunctions[path].push_back({ registration, std::move(load) });
    return registration;
}

/**
* @brief Removes the callbacks of one registration.
*/
void
Serializer::Unregister(Registration registration) noexcept
{
    if (p_impl == nullptr || registration == 0U)
        return;

    EraseRegistration(p_impl->m_loadfunctions, registration);
}

/**
* @brief Loads all registered files and dispatches the parsed JSON to callbacks.
*/
void
Serializer::Load(void)
{
    if (p_impl == nullptr || !AnySearchPathExists(p_impl->m_paths))
        return;

    for (auto& [registered, loads] : p_impl->m_loadfunctions)
    {
        if (loads.empty())
            continue;

        std::filesystem::path const filepath{ ResolveInSearchPaths(p_impl->m_paths, registered) };
        if (!std::filesystem::exists(filepath))
        {
            LogConfigEvent("CFG_MISSING_FILE", filepath, "file not found");
            continue;
        }

        auto content = Parse(filepath.string());

        if (!ValidateFormatVersion(filepath, content))
            continue;

        for (auto& load : loads)
        {
            try
            {
                load.function(content);
            }
            catch (std::exception const& exception)
            {
                LogConfigEvent("CFG_LOAD_EXCEPTION", filepath, exception.what());
            }
        }
    }
}

}
