#include "serializer.hpp"
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

bool
Serializer::ValidateSchemaVersion(std::filesystem::path const& file, json& content)
{
    if (!content.is_object())
    {
        LogConfigEvent("CFG_SCHEMA_INVALID_TYPE", file, "root document is not an object");
        return false;
    }

    if (!content.contains("schemaVersion"))
    {
        std::ostringstream details;
        details << "missing schemaVersion, fallback=" << kConfigSchemaVersion;
        LogConfigEvent("CFG_SCHEMA_MISSING", file, details.str());
        content["schemaVersion"] = kConfigSchemaVersion;
        return true;
    }

    auto const& schemaVersion = content["schemaVersion"];

    if (!schemaVersion.is_number_integer())
    {
        LogConfigEvent("CFG_SCHEMA_INVALID_TYPE", file, "schemaVersion must be an integer");
        return false;
    }

    int const schemaVersionValue = schemaVersion.get<int>();
    if (schemaVersionValue != kConfigSchemaVersion)
    {
        std::ostringstream details;
        details << "expected=" << kConfigSchemaVersion << ", found=" << schemaVersionValue;
        LogConfigEvent("CFG_SCHEMA_MISMATCH", file, details.str());
        return false;
    }

    return true;
}

struct Serializer::Impl 
{
    std::filesystem::path                                                    m_path{};
    std::unordered_map<std::filesystem::path, std::vector<JsonLoadFunction>> m_loadfunctions{};
    std::unordered_map<std::filesystem::path, std::vector<JsonSaveFunction>> m_savefunctions{};
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
        {
            LogConfigEvent("CFG_MISSING_FILE", filepath, "file not found");
            continue;
        }

        auto content = Parse(filepath.string());

        if (!ValidateSchemaVersion(filepath, content))
            continue;

        for (auto& load : loads)
        {
            try
            {
                load(content);
            }
            catch (std::exception const& exception)
            {
                LogConfigEvent("CFG_LOAD_EXCEPTION", filepath, exception.what());
            }
        }
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
        json doc{};

        for (auto& save : saves)
            save(doc);

        if (!doc.contains("schemaVersion"))
            doc["schemaVersion"] = kConfigSchemaVersion;

        std::ofstream(filepath) << doc.dump(4);
    }
}

}
