/**
 * @file PPOKit/include/ppokit/component/native_module.hpp
 * @brief Loading a DLL (component, network or environment library) on Windows and POSIX.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ppokit::component {

/**
 * @brief Libraries the build knows (its program registry): a bare name not found in the
 *        folder of the file that names it is looked up among them, by file name without
 *        "lib" and extension. Lets a project of the repository name "flight_visualiser"
 *        without copying the DLL into its folder. Set once at start-up, before any load.
 */
void SetKnownModules(std::vector<std::filesystem::path> modules);

/**
 * @brief The file a reference names: a path, or a bare name tried as name, name.dll,
 *        libname.so, name.so, libname.dylib and name.dylib in `directory`, then among the
 *        known modules (SetKnownModules).
 * @param what What is looked for, for the message, e.g. "component library".
 * @throws std::runtime_error listing every file tried.
 */
std::filesystem::path ResolveNativeModule(std::string const& reference, std::filesystem::path const& directory,
    char const* what);

/** @brief Loads a DLL. @throws std::runtime_error with the system's reason. */
void* OpenNativeModule(std::filesystem::path const& path);

void CloseNativeModule(void* module) noexcept;

/** @brief Address of an exported function, or null. */
void* NativeSymbol(void* module, char const* name) noexcept;

/** @brief Reject a module whose compiled semantics are missing or differ from the project's task. */
void RequireNativeTaskContract(void* module, std::filesystem::path const& path, std::uint64_t expected);

} // namespace ppokit::component
