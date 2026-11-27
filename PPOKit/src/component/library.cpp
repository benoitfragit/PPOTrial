/**
 * @file PPOKit/src/component/library.cpp
 * @brief Loading component DLLs, reading their manifests, validating configurations.
 */
#include "ppokit/component/library.hpp"
#include "ppokit/component/native_module.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <iterator>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <system_error>


namespace ppokit::component {
namespace {

using Json = nlohmann::ordered_json;

constexpr std::uint64_t FnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t FnvPrime = 1099511628211ULL;
constexpr std::size_t ErrorCapacity = 512;

std::uint64_t Fnv(std::uint64_t hash, void const* data, std::size_t size) noexcept
{
    auto const* bytes = static_cast<unsigned char const*>(data);
    for (std::size_t index = 0; index < size; ++index)
    {
        hash = (hash ^ bytes[index]) * FnvPrime;
    }
    return hash;
}

std::uint64_t HashFile(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    std::string const bytes((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    return Fnv(FnvOffset, bytes.data(), bytes.size());
}

[[noreturn]] void Fail(std::string const& source, std::string const& message)
{
    throw std::runtime_error(source + ": " + message);
}

Dtype ReadDtype(Json const& value, std::string const& source, std::string const& where)
{
    static std::map<std::string, Dtype> const names{
        {"f32", Dtype::F32}, {"f64", Dtype::F64}, {"i32", Dtype::I32}, {"bool", Dtype::Bool}};
    auto const found = value.is_string() ? names.find(value.get<std::string>()) : names.end();
    if (found == names.end())
    {
        Fail(source, where + ".dtype: expected f32, f64, i32 or bool");
    }
    return found->second;
}

std::vector<DimensionSpec> ReadShape(Json const& value, std::string const& source, std::string const& where)
{
    if (!value.is_array() || value.empty() || value.size() > 2)
    {
        Fail(source, where + ".shape: expected [n] or [capacity, n] (numbers or symbols such as \"N\")");
    }
    std::vector<DimensionSpec> shape;
    for (Json const& dimension : value)
    {
        if (dimension.is_string())
        {
            shape.push_back({dimension.get<std::string>(), 0});
        }
        else if (dimension.is_number_unsigned() && dimension.get<std::size_t>() > 0)
        {
            shape.push_back({{}, dimension.get<std::size_t>()});
        }
        else
        {
            Fail(source, where + ".shape: a dimension is a positive number or a symbol");
        }
    }
    return shape;
}

/** @brief Every symbol of an output and every mask must come from the inputs. */
/**
 * @brief An observer only watches: float inputs, no output, no gradient, no state, and
 *        every agent of a step in one call (batch).
 */
void CheckObserver(Manifest const& manifest, std::string const& source)
{
    Capabilities const& capabilities = manifest.capabilities;
    if (capabilities.differentiable || capabilities.trainable || capabilities.stateful)
    {
        Fail(source, "an observer cannot be differentiable, trainable or stateful: it only watches");
    }
    if (!capabilities.batch)
    {
        Fail(source, "an observer must declare batch: it receives every agent of a step in one call");
    }
    bool hasInput = false;
    for (PortSpec const& port : manifest.ports)
    {
        if (port.output)
        {
            Fail(source, "an observer has no output port ('" + port.name + "')");
        }
        if (port.dtype != Dtype::F32)
        {
            Fail(source, "observer port '" + port.name + "': only f32 inputs are supported");
        }
        hasInput = true;
    }
    if (!hasInput)
    {
        Fail(source, "an observer needs at least one input port");
    }
}

void CheckPortRelations(Manifest const& manifest, std::string const& source)
{
    std::set<std::string> inputSymbols;
    std::map<std::string, PortSpec const*> inputs;
    for (PortSpec const& port : manifest.ports)
    {
        if (!port.output)
        {
            inputs[port.name] = &port;
            for (DimensionSpec const& dimension : port.shape)
            {
                if (!dimension.symbol.empty())
                {
                    inputSymbols.insert(dimension.symbol);
                }
            }
        }
    }
    for (PortSpec const& port : manifest.ports)
    {
        std::string const where = "port '" + port.name + "'";
        if (!port.mask.empty())
        {
            auto const mask = inputs.find(port.mask);
            if (mask == inputs.end() || mask->second->dtype != Dtype::Bool)
            {
                Fail(source, where + ": its mask '" + port.mask + "' must be a bool input port");
            }
        }
        if (!port.output)
        {
            continue;
        }
        for (DimensionSpec const& dimension : port.shape)
        {
            if (!dimension.symbol.empty() && inputSymbols.count(dimension.symbol) == 0)
            {
                Fail(source, where + ": the dimension '" + dimension.symbol + "' appears in no input");
            }
        }
    }
}

SettingSpec ReadSetting(std::string const& name, Json const& value, std::string const& source)
{
    std::string const where = "config." + name;
    SettingSpec setting;
    setting.name = name;
    std::string const type = value.value("type", std::string());
    if (type == "float" || type == "int")
    {
        setting.type = type == "float" ? SettingSpec::Type::Float : SettingSpec::Type::Int;
        Json const range = value.value("range", Json::array());
        if (!range.is_array() || range.size() != 2 || !range[0].is_number() || !range[1].is_number())
        {
            Fail(source, where + ".range: expected [minimum, maximum]");
        }
        setting.minimum = range[0].get<double>();
        setting.maximum = range[1].get<double>();
        setting.defaultValue = value.value("default", setting.minimum);
    }
    else if (type == "bool")
    {
        setting.type = SettingSpec::Type::Bool;
        setting.maximum = 1.0;
        setting.defaultValue = value.value("default", false) ? 1.0 : 0.0;
    }
    else
    {
        Fail(source, where + ".type: expected float, int or bool");
    }
    if (setting.defaultValue < setting.minimum || setting.defaultValue > setting.maximum)
    {
        Fail(source, where + ": the default is outside the range");
    }
    return setting;
}

} // namespace

Manifest ParseManifest(std::string_view json, std::string const& source)
{
    Json const root = Json::parse(json, nullptr, false);
    if (!root.is_object() || root.value("schema", std::string()) != "ppokit.component/1")
    {
        Fail(source, "the manifest is not a \"ppokit.component/1\" object");
    }
    Manifest manifest;
    manifest.name = root.value("name", std::string());
    manifest.version = root.value("version", 0U);
    if (manifest.name.empty() || manifest.version == 0)
    {
        Fail(source, "the manifest needs a name and a version above 0");
    }
    std::set<std::string> names;
    Json const ports = root.value("ports", Json::array());
    for (Json const& entry : ports)
    {
        PortSpec port;
        port.name = entry.value("name", std::string());
        std::string const where = "port '" + port.name + "'";
        if (port.name.empty() || !names.insert(port.name).second)
        {
            Fail(source, where + ": every port needs a unique name");
        }
        std::string const direction = entry.value("direction", std::string());
        if (direction != "in" && direction != "out")
        {
            Fail(source, where + ".direction: expected in or out");
        }
        port.output = direction == "out";
        port.dtype = ReadDtype(entry.value("dtype", Json()), source, where);
        port.shape = ReadShape(entry.value("shape", Json()), source, where);
        port.mask = entry.value("mask", std::string());
        manifest.ports.push_back(std::move(port));
    }
    CheckPortRelations(manifest, source);
    Json const capabilities = root.value("capabilities", Json::object());
    manifest.capabilities.differentiable = capabilities.value("differentiable", false);
    manifest.capabilities.trainable = capabilities.value("trainable", false);
    manifest.capabilities.stateful = capabilities.value("stateful", false);
    manifest.capabilities.deterministic = capabilities.value("deterministic", true);
    manifest.capabilities.threadSafe = capabilities.value("threadSafe", false);
    manifest.capabilities.batch = capabilities.value("batch", false);
    manifest.capabilities.observer = capabilities.value("observer", false);
    if (manifest.capabilities.trainable && !manifest.capabilities.differentiable)
    {
        Fail(source, "a trainable component must be differentiable");
    }
    if (manifest.capabilities.observer)
    {
        CheckObserver(manifest, source);
    }
    Json const config = root.value("config", Json::object());
    for (auto const& [name, value] : config.items())
    {
        manifest.config.push_back(ReadSetting(name, value, source));
    }
    return manifest;
}

std::string ComponentRef::Id() const
{
    return manifest->name + "@" + std::to_string(manifest->version);
}

std::shared_ptr<Library const> Library::Open(std::string const& reference, std::filesystem::path const& directory)
{
    std::filesystem::path const path = ResolveNativeModule(reference, directory, "component library");
    // One Library per file: a DLL is loaded once, whoever asks.
    static std::mutex mutex;
    static std::map<std::string, std::weak_ptr<Library const>> open;
    std::lock_guard<std::mutex> const lock(mutex);
    if (auto existing = open[path.string()].lock())
    {
        return existing;
    }
    std::shared_ptr<Library> library(new Library());
    library->path_ = path;
    library->handle_ = OpenNativeModule(path);
    std::string const source = path.filename().string();
    auto const entry = reinterpret_cast<PpokitComponentsFunction>(NativeSymbol(library->handle_, PPOKIT_COMPONENTS_ENTRY));
    if (entry == nullptr)
    {
        Fail(source, std::string("not a PPOKit component library (no ") + PPOKIT_COMPONENTS_ENTRY
                + " function; write components with ppokit/component/sdk.hpp)");
    }
    PpokitComponentApi const* const* components = nullptr;
    std::size_t count = 0;
    if (entry(PPOKIT_COMPONENT_ABI, &components, &count) != 0 || components == nullptr)
    {
        Fail(source, "refuses component ABI " + std::to_string(PPOKIT_COMPONENT_ABI)
                + ": it was built with another PPOKit, rebuild it with this one");
    }
    for (std::size_t index = 0; index < count; ++index)
    {
        PpokitComponentApi const* const api = components[index];
        if (api == nullptr || api->abi != PPOKIT_COMPONENT_ABI || api->create == nullptr || api->forward == nullptr)
        {
            Fail(source, "component " + std::to_string(index) + " has an incomplete or foreign table");
        }
        library->apis_.push_back(api);
        library->manifests_.push_back(ParseManifest(api->manifestJson, source + " (" + api->name + ")"));
        Manifest const& manifest = library->manifests_.back();
        bool const consistent = (api->backward != nullptr) == manifest.capabilities.differentiable
            && (api->parameters != nullptr) == manifest.capabilities.trainable
            && (api->reset != nullptr) == manifest.capabilities.stateful;
        if (!consistent)
        {
            Fail(source, std::string("component '") + api->name
                    + "': its functions do not match the capabilities of its manifest");
        }
    }
    library->binaryHash_ = HashFile(path);
    open[path.string()] = library;
    return library;
}

Library::~Library()
{
    if (handle_ != nullptr)
    {
        CloseNativeModule(handle_);
    }
}

ComponentRef Library::At(std::size_t index) const
{
    return {shared_from_this(), apis_.at(index), &manifests_.at(index)};
}

ComponentRef Library::Find(std::string const& name, std::uint32_t version) const
{
    std::string available;
    for (std::size_t index = 0; index < apis_.size(); ++index)
    {
        if (manifests_[index].name == name && manifests_[index].version == version)
        {
            return At(index);
        }
        available += (available.empty() ? "" : ", ") + manifests_[index].name + "@"
            + std::to_string(manifests_[index].version);
    }
    Fail(path_.filename().string(), "no component " + name + "@" + std::to_string(version) + " (it provides: "
            + (available.empty() ? std::string("none") : available) + ")");
}

std::string ValidateConfig(Manifest const& manifest, std::string const& configJson)
{
    Json const given = Json::parse(configJson.empty() ? "{}" : configJson, nullptr, false);
    std::string const source = "component " + manifest.name;
    if (!given.is_object())
    {
        Fail(source, "\"config\" must be an object");
    }
    for (auto const& item : given.items())
    {
        bool const known = std::any_of(manifest.config.begin(), manifest.config.end(),
            [&item](SettingSpec const& setting) { return setting.name == item.key(); });
        if (!known)
        {
            std::string settings;
            for (SettingSpec const& setting : manifest.config)
            {
                settings += (settings.empty() ? "" : ", ") + setting.name;
            }
            Fail(source, "config." + item.key() + ": unknown setting (settings: "
                    + (settings.empty() ? std::string("none") : settings) + ")");
        }
    }
    Json complete = Json::object();
    for (SettingSpec const& setting : manifest.config)
    {
        std::string const where = "config." + setting.name;
        if (!given.contains(setting.name))
        {
            complete[setting.name] = setting.type == SettingSpec::Type::Bool ? Json(setting.defaultValue != 0.0)
                                                                             : Json(setting.defaultValue);
            continue;
        }
        Json const& value = given.at(setting.name);
        if (setting.type == SettingSpec::Type::Bool)
        {
            if (!value.is_boolean())
            {
                Fail(source, where + ": expected true or false");
            }
            complete[setting.name] = value;
            continue;
        }
        if (!value.is_number() || !std::isfinite(value.get<double>()))
        {
            Fail(source, where + ": expected a number");
        }
        double const number = value.get<double>();
        if (number < setting.minimum || number > setting.maximum)
        {
            Fail(source, where + ": " + value.dump() + " is outside [" + Json(setting.minimum).dump() + ", "
                    + Json(setting.maximum).dump() + "]");
        }
        if (setting.type == SettingSpec::Type::Int && std::floor(number) != number)
        {
            Fail(source, where + ": expected a whole number");
        }
        complete[setting.name] = number;
    }
    return complete.dump();
}

std::uint64_t ComponentHash(ComponentRef const& component, std::string const& configJson)
{
    std::uint64_t const binary = component.library->BinaryHash();
    std::uint64_t hash = Fnv(FnvOffset, &binary, sizeof(binary));
    std::string const manifest = component.api->manifestJson;
    hash = Fnv(hash, manifest.data(), manifest.size());
    return Fnv(hash, configJson.data(), configJson.size());
}

Instance::Instance(ComponentRef component, std::string const& configJson, std::string const& shapesJson,
    std::uint64_t seed)
    : component_(std::move(component))
{
    std::array<char, ErrorCapacity> error{};
    handle_ = component_.api->create(configJson.c_str(), shapesJson.c_str(), seed, error.data(), error.size());
    if (handle_ == nullptr)
    {
        Fail("component " + component_.Id(), std::string("cannot be created: ") + error.data());
    }
}

Instance::~Instance()
{
    component_.api->destroy(handle_);
}

void Instance::Require(int code, char const* call) const
{
    if (code == 0)
    {
        return;
    }
    std::array<char, ErrorCapacity> error{};
    component_.api->lastError(handle_, error.data(), error.size());
    Fail("component " + component_.Id(), std::string(call) + " failed: " + error.data());
}

void Instance::Forward(std::uint32_t batch, PpokitBuffer const* inputs, PpokitBuffer* outputs, void* cache)
{
    Require(component_.api->forward(handle_, batch, inputs, outputs, cache), "forward");
}

void Instance::Backward(std::uint32_t batch, PpokitBuffer const* outputGradients, void const* cache,
    PpokitBuffer* inputGradients, float* parameterGradients)
{
    Require(component_.api->backward(handle_, batch, outputGradients, cache, inputGradients, parameterGradients),
        "backward");
}

std::size_t Instance::CacheSize(std::uint32_t batch) const
{
    return component_.api->cacheSize != nullptr ? component_.api->cacheSize(handle_, batch) : 0;
}

float* Instance::Parameters(std::size_t& count)
{
    float* data = nullptr;
    Require(component_.api->parameters(handle_, &data, &count), "parameters");
    return data;
}

} // namespace ppokit::component
