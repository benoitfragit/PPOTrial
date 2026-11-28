/**
 * @file PPOKit/include/ppokit/component/sdk.hpp
 * @brief C++ SDK to write PPOKit components: a class in, the C ABI table out.
 * @details Header-only, C++17, no dependency. Write a class deriving from Component, with
 * a static Describe() giving its ports, capabilities and configuration, then list the
 * classes in PPOKIT_COMPONENTS. The SDK writes the manifest from Describe() (so manifest
 * and code cannot disagree), catches every exception before the C boundary and turns it
 * into an error message, and implements only the optional functions the capabilities
 * declare.
 *
 * @code
 * class Scale final : public ppokit::sdk::Component
 * {
 * public:
 *     static ppokit::sdk::Manifest Describe()
 *     {
 *         ppokit::sdk::Manifest manifest("scale", 1);
 *         manifest.Input("x", {"N"}).Output("y", {"N"});
 *         manifest.Differentiable().Trainable();
 *         manifest.Number("initial", 1.0, 0.0, 10.0);
 *         return manifest;
 *     }
 *     Scale(ppokit::sdk::Config const& config, ppokit::sdk::Shapes const& shapes, std::uint64_t seed)
 *         : weights_(shapes["N"], static_cast<float>(config.Number("initial"))) {}
 *     void Forward(ppokit::sdk::Call const& call) override
 *     {
 *         for (std::size_t i = 0; i < weights_.size(); ++i) call.Output(0)[i] = weights_[i] * call.Input(0)[i];
 *     }
 *     ... Backward and Parameters, since it declares them.
 * };
 * PPOKIT_COMPONENTS(Scale)
 * @endcode
 *
 * Calls are made with batch = 1 by the graph engine unless the component declares Batch().
 */
#pragma once

#include "ppokit/component/abi.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ppokit::sdk {

/** @brief One dimension of a port: a number, or a symbol (such as "N") fixed by the wiring. */
struct Dimension final
{
    Dimension(std::size_t value) : size(value) {} // NOLINT: written as plain numbers
    Dimension(char const* name) : symbol(name) {} // NOLINT: written as "N"
    std::string symbol;
    std::size_t size = 0;
};

/** @brief Ports, capabilities and configuration of a component. */
class Manifest final
{
public:
    Manifest(std::string name, std::uint32_t version) : name_(std::move(name)), version_(version) {}

    /** @brief A float input of shape `shape`, e.g. {"N", 13}; `mask` names its presence mask port. */
    Manifest& Input(std::string name, std::vector<Dimension> shape, std::string mask = {})
    {
        ports_.push_back({std::move(name), false, PPOKIT_DTYPE_F32, std::move(shape), std::move(mask)});
        return *this;
    }
    /** @brief A presence mask input (one byte per entity), named by the "mask" of a set input. */
    Manifest& Mask(std::string name, std::vector<Dimension> shape)
    {
        ports_.push_back({std::move(name), false, PPOKIT_DTYPE_BOOL, std::move(shape), {}});
        return *this;
    }
    /** @brief A float output; `mask` names the input mask that its entities follow. */
    Manifest& Output(std::string name, std::vector<Dimension> shape, std::string mask = {})
    {
        ports_.push_back({std::move(name), true, PPOKIT_DTYPE_F32, std::move(shape), std::move(mask)});
        return *this;
    }

    Manifest& Differentiable() { differentiable_ = true; return *this; }
    /** @brief Exposes parameters to PPOKit's optimizer; implies Differentiable(). */
    Manifest& Trainable() { trainable_ = differentiable_ = true; return *this; }
    Manifest& Stateful() { stateful_ = true; return *this; }
    /** @brief Uses its seed for random draws; outputs then depend on the seed. */
    Manifest& NonDeterministic() { deterministic_ = false; return *this; }
    Manifest& ThreadSafe() { threadSafe_ = true; return *this; }
    Manifest& Batch() { batch_ = true; return *this; }
    /**
     * @brief Only watches (a Visualiseur): inputs only, called after each environment step
     *        with every agent at once, so it implies Batch().
     */
    Manifest& Observer() { observer_ = batch_ = true; return *this; }

    /** @brief A float setting of graph.json, with its default and range. */
    Manifest& Number(std::string name, double defaultValue, double minimum, double maximum)
    {
        config_.push_back({std::move(name), "float", defaultValue, minimum, maximum});
        return *this;
    }
    /** @brief A whole-number setting. */
    Manifest& Integer(std::string name, long long defaultValue, long long minimum, long long maximum)
    {
        config_.push_back({std::move(name), "int", static_cast<double>(defaultValue), static_cast<double>(minimum),
            static_cast<double>(maximum)});
        return *this;
    }
    /** @brief A true/false setting. */
    Manifest& Flag(std::string name, bool defaultValue)
    {
        config_.push_back({std::move(name), "bool", defaultValue ? 1.0 : 0.0, 0.0, 1.0});
        return *this;
    }

    std::string const& Name() const noexcept { return name_; }
    std::uint32_t Version() const noexcept { return version_; }
    bool IsDifferentiable() const noexcept { return differentiable_; }
    bool IsTrainable() const noexcept { return trainable_; }
    bool IsStateful() const noexcept { return stateful_; }

    /** @brief The manifest JSON ("ppokit.component/1") PPOKit reads. */
    std::string ToJson() const
    {
        std::string json = "{\"schema\":\"ppokit.component/1\",\"name\":" + Quoted(name_)
            + ",\"version\":" + std::to_string(version_) + ",\"ports\":[";
        for (std::size_t index = 0; index < ports_.size(); ++index)
        {
            Port const& port = ports_[index];
            json += (index == 0 ? "" : ",");
            json += "{\"name\":" + Quoted(port.name) + ",\"direction\":" + Quoted(port.output ? "out" : "in")
                + ",\"dtype\":" + Quoted(port.dtype == PPOKIT_DTYPE_BOOL ? "bool" : "f32") + ",\"shape\":[";
            for (std::size_t axis = 0; axis < port.shape.size(); ++axis)
            {
                Dimension const& dimension = port.shape[axis];
                json += (axis == 0 ? "" : ",");
                json += dimension.symbol.empty() ? std::to_string(dimension.size) : Quoted(dimension.symbol);
            }
            json += "]";
            if (!port.mask.empty())
            {
                json += ",\"mask\":" + Quoted(port.mask);
            }
            json += "}";
        }
        json += "],\"capabilities\":{\"differentiable\":" + Bool(differentiable_)
            + ",\"trainable\":" + Bool(trainable_) + ",\"stateful\":" + Bool(stateful_)
            + ",\"deterministic\":" + Bool(deterministic_) + ",\"threadSafe\":" + Bool(threadSafe_)
            + ",\"batch\":" + Bool(batch_) + ",\"observer\":" + Bool(observer_) + "},\"config\":{";
        for (std::size_t index = 0; index < config_.size(); ++index)
        {
            Setting const& setting = config_[index];
            json += (index == 0 ? "" : ",");
            json += Quoted(setting.name) + ":{\"type\":" + Quoted(setting.type) + ",\"default\":"
                + Number(setting.type, setting.defaultValue);
            if (std::string(setting.type) != "bool")
            {
                json += ",\"range\":[" + Number(setting.type, setting.minimum) + ","
                    + Number(setting.type, setting.maximum) + "]";
            }
            json += "}";
        }
        return json + "}}";
    }

private:
    struct Port final
    {
        std::string name;
        bool output;
        std::uint32_t dtype;
        std::vector<Dimension> shape;
        std::string mask;
    };
    struct Setting final
    {
        std::string name;
        char const* type;
        double defaultValue;
        double minimum;
        double maximum;
    };

    static std::string Quoted(std::string const& text)
    {
        std::string quoted = "\"";
        for (char const character : text)
        {
            if (character == '"' || character == '\\')
            {
                quoted += '\\';
            }
            quoted += character;
        }
        return quoted + "\"";
    }
    static std::string Bool(bool value) { return value ? "true" : "false"; }
    static std::string Number(char const* type, double value)
    {
        if (std::string(type) == "bool")
        {
            return value != 0.0 ? "true" : "false";
        }
        char text[64];
        std::snprintf(text, sizeof(text), std::string(type) == "int" ? "%.0f" : "%.17g", value);
        return text;
    }

    std::string name_;
    std::uint32_t version_;
    std::vector<Port> ports_;
    std::vector<Setting> config_;
    bool differentiable_ = false;
    bool trainable_ = false;
    bool stateful_ = false;
    bool deterministic_ = true;
    bool threadSafe_ = false;
    bool batch_ = false;
    bool observer_ = false;
};

namespace detail {

/** @brief Values of a flat JSON object of numbers and booleans, as PPOKit writes them. */
class FlatValues
{
public:
    explicit FlatValues(char const* json)
    {
        std::string const text = json != nullptr ? json : "{}";
        std::size_t position = 0;
        while ((position = text.find('"', position)) != std::string::npos)
        {
            std::size_t const end = text.find('"', position + 1);
            std::string const key = text.substr(position + 1, end - position - 1);
            std::size_t const colon = text.find(':', end);
            std::size_t valueStart = text.find_first_not_of(" \t\n", colon + 1);
            double value = 0.0;
            if (text.compare(valueStart, 4, "true") == 0)
            {
                value = 1.0;
            }
            else if (text.compare(valueStart, 5, "false") != 0)
            {
                value = std::strtod(text.c_str() + valueStart, nullptr);
            }
            entries_.emplace_back(key, value);
            position = text.find_first_of(",}", valueStart);
        }
    }

protected:
    double Find(std::string const& key) const
    {
        for (auto const& [name, value] : entries_)
        {
            if (name == key)
            {
                return value;
            }
        }
        throw std::invalid_argument("unknown name '" + key + "'");
    }

private:
    std::vector<std::pair<std::string, double>> entries_;
};

} // namespace detail

/** @brief Validated configuration: every setting of the manifest, with the graph.json value or its default. */
class Config final : public detail::FlatValues
{
public:
    using FlatValues::FlatValues;
    double Number(std::string const& name) const { return Find(name); }
    long long Integer(std::string const& name) const { return static_cast<long long>(Find(name)); }
    bool Flag(std::string const& name) const { return Find(name) != 0.0; }
};

/** @brief Symbolic dimensions fixed by the wiring, e.g. shapes["N"] == 10. */
class Shapes final : public detail::FlatValues
{
public:
    using FlatValues::FlatValues;
    std::size_t operator[](std::string const& symbol) const { return static_cast<std::size_t>(Find(symbol)); }
};

/** @brief The buffers of one forward or backward call (PPOKit owns them all). */
class Call final
{
public:
    Call(std::uint32_t batch, PpokitBuffer const* inputs, PpokitBuffer* outputs, void* cache) noexcept
        : batch_(batch), inputs_(inputs), outputs_(outputs), cache_(cache)
    {
    }
    std::uint32_t Batch() const noexcept { return batch_; }
    /** @brief Input `port`, in the order of the "in" ports of the manifest (masks included). */
    float const* Input(std::size_t port) const noexcept { return static_cast<float const*>(inputs_[port].data); }
    std::uint8_t const* Mask(std::size_t port) const noexcept
    {
        return static_cast<std::uint8_t const*>(inputs_[port].data);
    }
    float* Output(std::size_t port) const noexcept { return static_cast<float*>(outputs_[port].data); }
    void* Cache() const noexcept { return cache_; }

private:
    std::uint32_t batch_;
    PpokitBuffer const* inputs_;
    PpokitBuffer* outputs_;
    void* cache_;
};

/** @brief The buffers of a backward call. */
class GradientCall final
{
public:
    GradientCall(std::uint32_t batch, PpokitBuffer const* outputGradients, void const* cache,
        PpokitBuffer* inputGradients, float* parameterGradients) noexcept
        : batch_(batch), outputGradients_(outputGradients), cache_(cache), inputGradients_(inputGradients),
          parameterGradients_(parameterGradients)
    {
    }
    std::uint32_t Batch() const noexcept { return batch_; }
    float const* OutputGradient(std::size_t port) const noexcept
    {
        return static_cast<float const*>(outputGradients_[port].data);
    }
    /** @brief Where to ADD d loss / d input; null for a mask port. */
    float* InputGradient(std::size_t port) const noexcept { return static_cast<float*>(inputGradients_[port].data); }
    /** @brief Where to ADD d loss / d parameter (trainable components). */
    float* ParameterGradients() const noexcept { return parameterGradients_; }
    void const* Cache() const noexcept { return cache_; }

private:
    std::uint32_t batch_;
    PpokitBuffer const* outputGradients_;
    void const* cache_;
    PpokitBuffer* inputGradients_;
    float* parameterGradients_;
};

/** @brief Base class of a component; override what its manifest declares. */
class Component
{
public:
    virtual ~Component() = default;
    virtual void Forward(Call const& call) = 0;
    /** @brief Bytes Forward may fill in Call::Cache() for Backward. */
    virtual std::size_t CacheSize(std::uint32_t /*batch*/) const { return 0; }
    virtual void Backward(GradientCall const& /*call*/)
    {
        throw std::logic_error("Backward is not implemented although the manifest declares Differentiable()");
    }
    /** @brief Trainable: the parameters, which PPOKit's optimizer updates in place. */
    virtual float* Parameters(std::size_t& /*count*/)
    {
        throw std::logic_error("Parameters is not implemented although the manifest declares Trainable()");
    }
    virtual void Reset(std::uint32_t /*agentSlot*/, std::uint64_t /*seed*/) {}
    virtual std::size_t StateSize() const { return 0; }
    virtual void SaveState(void* /*buffer*/, std::size_t /*size*/) const {}
    virtual void LoadState(void const* /*buffer*/, std::size_t /*size*/) {}
};

namespace detail {

/** @brief What the C table points to: the component and the text of its last error. */
struct Instance final
{
    std::unique_ptr<Component> component;
    std::string error;
};

inline void CopyError(std::string const& message, char* error, std::size_t errorSize) noexcept
{
    if (error != nullptr && errorSize > 0)
    {
        std::snprintf(error, errorSize, "%s", message.c_str());
    }
}

/** @brief Runs `body` on an instance; an exception becomes return code 1 and the instance's error. */
template<typename Body>
int Guarded(void* handle, Body&& body) noexcept
{
    auto* const instance = static_cast<Instance*>(handle);
    try
    {
        body(*instance->component);
        return 0;
    }
    catch (std::exception const& failure)
    {
        instance->error = failure.what();
    }
    catch (...)
    {
        instance->error = "unknown exception";
    }
    return 1;
}

/** @brief The C table of component class T, built once. */
template<typename T>
PpokitComponentApi const* Table()
{
    static Manifest const manifest = T::Describe();
    static std::string const manifestJson = manifest.ToJson();
    static PpokitComponentApi const table = [] {
        PpokitComponentApi api{};
        api.abi = PPOKIT_COMPONENT_ABI;
        api.name = manifest.Name().c_str();
        api.version = manifest.Version();
        api.manifestJson = manifestJson.c_str();
        api.create = [](char const* config, char const* shapes, std::uint64_t seed, char* error,
                         std::size_t errorSize) -> void* {
            try
            {
                auto instance = std::make_unique<Instance>();
                instance->component = std::make_unique<T>(Config(config), Shapes(shapes), seed);
                return instance.release();
            }
            catch (std::exception const& failure)
            {
                CopyError(failure.what(), error, errorSize);
            }
            catch (...)
            {
                CopyError("unknown exception", error, errorSize);
            }
            return nullptr;
        };
        api.destroy = [](void* handle) { delete static_cast<Instance*>(handle); };
        api.forward = [](void* handle, std::uint32_t batch, PpokitBuffer const* inputs, PpokitBuffer* outputs,
                          void* cache) {
            return Guarded(handle, [&](Component& component) { component.Forward(Call(batch, inputs, outputs, cache)); });
        };
        api.cacheSize = [](void* handle, std::uint32_t batch) -> std::size_t {
            try
            {
                return static_cast<Instance*>(handle)->component->CacheSize(batch);
            }
            catch (...)
            {
                return 0;
            }
        };
        if (manifest.IsDifferentiable())
        {
            api.backward = [](void* handle, std::uint32_t batch, PpokitBuffer const* outputGradients,
                               void const* cache, PpokitBuffer* inputGradients, float* parameterGradients) {
                return Guarded(handle, [&](Component& component) {
                    component.Backward(GradientCall(batch, outputGradients, cache, inputGradients, parameterGradients));
                });
            };
        }
        if (manifest.IsTrainable())
        {
            api.parameters = [](void* handle, float** data, std::size_t* count) {
                return Guarded(handle, [&](Component& component) { *data = component.Parameters(*count); });
            };
        }
        if (manifest.IsStateful())
        {
            api.reset = [](void* handle, std::uint32_t slot, std::uint64_t seed) {
                return Guarded(handle, [&](Component& component) { component.Reset(slot, seed); });
            };
            api.stateSize = [](void* handle) -> std::size_t {
                try
                {
                    return static_cast<Instance*>(handle)->component->StateSize();
                }
                catch (...)
                {
                    return 0;
                }
            };
            api.saveState = [](void* handle, void* buffer, std::size_t size) {
                return Guarded(handle, [&](Component& component) { component.SaveState(buffer, size); });
            };
            api.loadState = [](void* handle, void const* buffer, std::size_t size) {
                return Guarded(handle, [&](Component& component) { component.LoadState(buffer, size); });
            };
        }
        api.lastError = [](void* handle, char* error, std::size_t errorSize) {
            CopyError(static_cast<Instance*>(handle)->error, error, errorSize);
            return 0;
        };
        return api;
    }();
    return &table;
}

} // namespace detail
} // namespace ppokit::sdk

/** @brief Exports the components of this DLL: PPOKIT_COMPONENTS(Scale, TrackFilter) */
#define PPOKIT_COMPONENTS(...)                                                                                        \
    extern "C" PPOKIT_EXPORT int ppokit_components(                                                                  \
        uint32_t requestedAbi, const PpokitComponentApi* const** components, size_t* count)                        \
    {                                                                                                                 \
        if (requestedAbi != PPOKIT_COMPONENT_ABI)                                                                     \
        {                                                                                                             \
            return 1;                                                                                                 \
        }                                                                                                             \
        static const PpokitComponentApi* const table[] = {PPOKIT_SDK_TABLES(__VA_ARGS__)};                           \
        *components = table;                                                                                          \
        *count = sizeof(table) / sizeof(table[0]);                                                                    \
        return 0;                                                                                                     \
    }

/** @brief ppokit::sdk::detail::Table<A>(), ... for up to eight component classes. */
#define PPOKIT_SDK_TABLE(T) ppokit::sdk::detail::Table<T>()
#define PPOKIT_SDK_EXPAND(x) x
#define PPOKIT_SDK_COUNT(...) PPOKIT_SDK_EXPAND(PPOKIT_SDK_COUNT_(__VA_ARGS__, 8, 7, 6, 5, 4, 3, 2, 1))
#define PPOKIT_SDK_COUNT_(_1, _2, _3, _4, _5, _6, _7, _8, N, ...) N
#define PPOKIT_SDK_JOIN(a, b) PPOKIT_SDK_JOIN_(a, b)
#define PPOKIT_SDK_JOIN_(a, b) a##b
#define PPOKIT_SDK_TABLES(...) PPOKIT_SDK_EXPAND(PPOKIT_SDK_JOIN(PPOKIT_SDK_TABLES_, PPOKIT_SDK_COUNT(__VA_ARGS__))(__VA_ARGS__))
#define PPOKIT_SDK_TABLES_1(a) PPOKIT_SDK_TABLE(a)
#define PPOKIT_SDK_TABLES_2(a, ...) PPOKIT_SDK_TABLE(a), PPOKIT_SDK_EXPAND(PPOKIT_SDK_TABLES_1(__VA_ARGS__))
#define PPOKIT_SDK_TABLES_3(a, ...) PPOKIT_SDK_TABLE(a), PPOKIT_SDK_EXPAND(PPOKIT_SDK_TABLES_2(__VA_ARGS__))
#define PPOKIT_SDK_TABLES_4(a, ...) PPOKIT_SDK_TABLE(a), PPOKIT_SDK_EXPAND(PPOKIT_SDK_TABLES_3(__VA_ARGS__))
#define PPOKIT_SDK_TABLES_5(a, ...) PPOKIT_SDK_TABLE(a), PPOKIT_SDK_EXPAND(PPOKIT_SDK_TABLES_4(__VA_ARGS__))
#define PPOKIT_SDK_TABLES_6(a, ...) PPOKIT_SDK_TABLE(a), PPOKIT_SDK_EXPAND(PPOKIT_SDK_TABLES_5(__VA_ARGS__))
#define PPOKIT_SDK_TABLES_7(a, ...) PPOKIT_SDK_TABLE(a), PPOKIT_SDK_EXPAND(PPOKIT_SDK_TABLES_6(__VA_ARGS__))
#define PPOKIT_SDK_TABLES_8(a, ...) PPOKIT_SDK_TABLE(a), PPOKIT_SDK_EXPAND(PPOKIT_SDK_TABLES_7(__VA_ARGS__))
