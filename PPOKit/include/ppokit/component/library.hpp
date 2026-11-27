/**
 * @file PPOKit/include/ppokit/component/library.hpp
 * @brief Loading component DLLs: their manifests, configurations and instances.
 * @details A Library is a loaded DLL with the components it lists. Libraries are shared:
 * opening the same file twice gives the same Library, which stays loaded while a graph
 * uses one of its components. Every failure says which file, component or setting is at
 * fault and what to do, e.g. "trackfilter.dll: built for component ABI 2, PPOKit reads 1:
 * rebuild it with this PPOKit".
 */
#pragma once

#include "ppokit/component/abi.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ppokit::component {

enum class Dtype : std::uint8_t
{
    F32,
    F64,
    I32,
    Bool,
};

/** @brief One dimension of a port: a number, or a symbol fixed by the wiring ("N"). */
struct DimensionSpec final
{
    std::string symbol;
    std::size_t size = 0;
};

struct PortSpec final
{
    std::string name;
    bool output = false;
    Dtype dtype = Dtype::F32;
    std::vector<DimensionSpec> shape;
    /** @brief Presence mask port of a set input, or the input mask an output set follows. */
    std::string mask;
};

struct Capabilities final
{
    bool differentiable = false;
    bool trainable = false;
    bool stateful = false;
    bool deterministic = true;
    bool threadSafe = false;
    bool batch = false;
    /**
     * @brief Only watches: inputs, no output. Lives in the "observers" section of graph.json,
     *        never in a network, and is called after each environment step with every agent.
     */
    bool observer = false;
};

struct SettingSpec final
{
    enum class Type : std::uint8_t
    {
        Float,
        Int,
        Bool,
    };
    std::string name;
    Type type = Type::Float;
    double defaultValue = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
};

/** @brief A validated manifest ("ppokit.component/1"). */
struct Manifest final
{
    std::string name;
    std::uint32_t version = 0;
    std::vector<PortSpec> ports;
    Capabilities capabilities;
    std::vector<SettingSpec> config;
};

/**
 * @brief Parses and checks a manifest: known dtypes, shapes of rank 1 or 2, masks naming
 *        bool inputs, output dimensions defined by the inputs, settings in range.
 * @throws std::runtime_error naming `source` and the faulty entry.
 */
Manifest ParseManifest(std::string_view json, std::string const& source);

class Library;

/** @brief A component of a loaded library; the library stays loaded while this exists. */
struct ComponentRef final
{
    std::shared_ptr<Library const> library;
    PpokitComponentApi const* api = nullptr;
    Manifest const* manifest = nullptr;

    /** @brief "name@version", as written after "component:" in graph.json. */
    std::string Id() const;
};

/** @brief A loaded component DLL. */
class Library final : public std::enable_shared_from_this<Library>
{
public:
    /**
     * @brief Opens a DLL, or returns the one already open at that path.
     * @param reference A path, or a bare name ("trackfilter") looked up in `directory` as
     *        trackfilter.dll, libtrackfilter.so, trackfilter.so or libtrackfilter.dylib.
     * @param directory Folder of relative references (the folder of graph.json).
     * @throws std::runtime_error when the file is missing, is not a component DLL, or
     *         was built for another ABI.
     */
    static std::shared_ptr<Library const> Open(std::string const& reference, std::filesystem::path const& directory);

    ~Library();
    Library(Library const&) = delete;
    Library& operator=(Library const&) = delete;

    std::filesystem::path const& Path() const noexcept { return path_; }
    /** @brief FNV-1a hash of the DLL file, recorded with the networks that use it. */
    std::uint64_t BinaryHash() const noexcept { return binaryHash_; }
    std::size_t Count() const noexcept { return apis_.size(); }
    /** @brief Component `index`, in the DLL's order. */
    ComponentRef At(std::size_t index) const;
    /** @brief The component "name@version". @throws std::runtime_error listing the available ones. */
    ComponentRef Find(std::string const& name, std::uint32_t version) const;

private:
    Library() = default;

    std::filesystem::path path_;
    void* handle_ = nullptr;
    std::uint64_t binaryHash_ = 0;
    std::vector<PpokitComponentApi const*> apis_;
    std::vector<Manifest> manifests_;
};

/**
 * @brief Checks a graph.json "config" object against the manifest and fills the defaults.
 * @param configJson The object as JSON text ("{}" when absent).
 * @return The complete configuration as JSON text, the one handed to the DLL.
 * @throws std::runtime_error naming the setting: unknown, wrong type or out of range.
 */
std::string ValidateConfig(Manifest const& manifest, std::string const& configJson);

/** @brief Hash identifying a component as used: its DLL, its manifest and its configuration. */
std::uint64_t ComponentHash(ComponentRef const& component, std::string const& configJson);

/** @brief One instance of a component (RAII); calls fail with the component's own message. */
class Instance final
{
public:
    /** @throws std::runtime_error with the reason the component gives. */
    Instance(ComponentRef component, std::string const& configJson, std::string const& shapesJson, std::uint64_t seed);
    ~Instance();
    Instance(Instance const&) = delete;
    Instance& operator=(Instance const&) = delete;

    ComponentRef const& Component() const noexcept { return component_; }
    void Forward(std::uint32_t batch, PpokitBuffer const* inputs, PpokitBuffer* outputs, void* cache);
    void Backward(std::uint32_t batch, PpokitBuffer const* outputGradients, void const* cache,
        PpokitBuffer* inputGradients, float* parameterGradients);
    std::size_t CacheSize(std::uint32_t batch) const;
    /** @brief Trainable components: their parameters, owned by the component. */
    float* Parameters(std::size_t& count);

private:
    /** @brief Throws the component's last error when `code` is not 0. */
    void Require(int code, char const* call) const;

    ComponentRef component_;
    void* handle_ = nullptr;
};

} // namespace ppokit::component
