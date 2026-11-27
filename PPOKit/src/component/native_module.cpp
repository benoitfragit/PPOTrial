/**
 * @file PPOKit/src/component/native_module.cpp
 * @brief Loading a DLL on Windows (LoadLibrary) and POSIX (dlopen).
 */
#include "ppokit/component/native_module.hpp"
#include "ppokit/abi/task_contract.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace ppokit::component {

namespace {

[[noreturn]] void Fail(std::string const& source, std::string const& detail)
{
    throw std::runtime_error(source + ": " + detail);
}

std::vector<std::filesystem::path>& KnownModules()
{
    static std::vector<std::filesystem::path> modules;
    return modules;
}

/** @brief Name of a library file without its "lib" prefix and its extension. */
std::string ModuleName(std::filesystem::path const& file)
{
    std::string name = file.stem().string();
    if (name.rfind("lib", 0) == 0)
    {
        name.erase(0, 3);
    }
    return name;
}

/** @brief The known module named `name`, or an empty path. */
std::filesystem::path FindKnownModule(std::string const& name)
{
    std::error_code error;
    for (std::filesystem::path const& known : KnownModules())
    {
        if (ModuleName(known) == name && std::filesystem::is_regular_file(known, error))
        {
            return known;
        }
    }
    return {};
}

} // namespace

void SetKnownModules(std::vector<std::filesystem::path> modules)
{
    KnownModules() = std::move(modules);
}

std::filesystem::path ResolveNativeModule(std::string const& reference, std::filesystem::path const& directory,
    char const* what)
{
    std::filesystem::path const written(reference);
    std::filesystem::path const base = written.is_absolute() ? written : directory / written;
    std::vector<std::filesystem::path> candidates{base};
    std::string const stem = base.filename().string();
    for (char const* pattern : {"%s.dll", "lib%s.so", "%s.so", "lib%s.dylib", "%s.dylib"})
    {
        std::array<char, 512> name{};
        std::snprintf(name.data(), name.size(), pattern, stem.c_str());
        candidates.push_back(base.parent_path() / name.data());
    }
    std::error_code error;
    for (std::filesystem::path const& candidate : candidates)
    {
        if (std::filesystem::is_regular_file(candidate, error))
        {
            return std::filesystem::canonical(candidate, error);
        }
    }
    if (!written.has_parent_path())
    {
        std::filesystem::path const known = FindKnownModule(reference);
        if (!known.empty())
        {
            return std::filesystem::canonical(known, error);
        }
    }
    std::string tried;
    for (std::filesystem::path const& candidate : candidates)
    {
        tried += "\n  " + candidate.string();
    }
    if (!written.has_parent_path() && !KnownModules().empty())
    {
        tried += "\n  and " + std::to_string(KnownModules().size()) + " libraries of the build";
    }
    Fail(reference, std::string(what) + " not found; looked for:" + tried);
}

void* OpenNativeModule(std::filesystem::path const& path)
{
#if defined(_WIN32)
    HMODULE const module = LoadLibraryW(path.wstring().c_str());
    if (module == nullptr)
    {
        Fail(path.string(), "cannot be loaded (Windows error " + std::to_string(GetLastError())
                + "): a missing dependency, or a DLL built for another architecture");
    }
    return reinterpret_cast<void*>(module);
#else
    void* const module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (module == nullptr)
    {
        char const* const reason = dlerror();
        Fail(path.string(), std::string("cannot be loaded: ") + (reason != nullptr ? reason : "unknown reason"));
    }
    return module;
#endif
}

void CloseNativeModule(void* module) noexcept
{
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(module));
#else
    dlclose(module);
#endif
}

void* NativeSymbol(void* module, char const* name) noexcept
{
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(module), name));
#else
    return dlsym(module, name);
#endif
}

void RequireNativeTaskContract(void* module, std::filesystem::path const& path, std::uint64_t expected)
{
    auto const query = reinterpret_cast<decltype(&ppokit_task_semantic_hash)>(
        NativeSymbol(module, "ppokit_task_semantic_hash"));
    if (query == nullptr)
    {
        Fail(path.string(), "missing compiled task contract: rebuild with ppokit_bind_task_contract");
    }
    if (query() != expected)
    {
        Fail(path.string(), "compiled task contract differs from task.json");
    }
}

} // namespace ppokit::component
