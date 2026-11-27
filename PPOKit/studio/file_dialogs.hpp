/**
 * @file PPOKit/studio/file_dialogs.hpp
 * @brief Native folder and file pickers (Windows); elsewhere the studio asks for a typed path.
 */
#pragma once

#include <filesystem>
#include <optional>

namespace ppokit::studio {

/** @brief Whether the pickers below open a native dialog on this system. */
bool NativeDialogsAvailable() noexcept;

/**
 * @brief Lets the user choose a folder.
 * @return The folder, or nothing when cancelled or when no native dialog exists.
 */
std::optional<std::filesystem::path> PickFolder(std::filesystem::path const& start, wchar_t const* title);

/**
 * @brief Lets the user choose a component library (.dll, .so, .dylib).
 * @return The file, or nothing when cancelled or when no native dialog exists.
 */
std::optional<std::filesystem::path> PickComponentLibrary(std::filesystem::path const& start);

} // namespace ppokit::studio
