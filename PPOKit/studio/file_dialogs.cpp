/**
 * @file PPOKit/studio/file_dialogs.cpp
 * @brief Windows common item dialogs (IFileOpenDialog), as in MFDStudio's editor.
 */
#include "file_dialogs.hpp"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <shobjidl.h>
#endif

namespace ppokit::studio {

#if defined(_WIN32)

namespace {

/** @brief COM for the lifetime of a dialog (the studio's thread may already have it). */
class ComScope final
{
public:
    ComScope()
    {
        HRESULT const result = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        initialized_ = result == S_OK || result == S_FALSE;
        ready_ = initialized_ || result == RPC_E_CHANGED_MODE;
    }
    ~ComScope()
    {
        if (initialized_)
        {
            CoUninitialize();
        }
    }
    ComScope(ComScope const&) = delete;
    ComScope& operator=(ComScope const&) = delete;
    bool Ready() const noexcept { return ready_; }

private:
    bool initialized_ = false;
    bool ready_ = false;
};

/** @brief Releases a COM interface when it goes out of scope. */
template<typename Interface>
class ComPointer final
{
public:
    ComPointer() = default;
    ~ComPointer()
    {
        if (pointer_ != nullptr)
        {
            pointer_->Release();
        }
    }
    ComPointer(ComPointer const&) = delete;
    ComPointer& operator=(ComPointer const&) = delete;
    Interface** Address() noexcept { return &pointer_; }
    Interface* operator->() const noexcept { return pointer_; }
    Interface* Get() const noexcept { return pointer_; }

private:
    Interface* pointer_ = nullptr;
};

/** @brief Shows `dialog` starting in `start`; the chosen path, or nothing when cancelled. */
std::optional<std::filesystem::path> Show(IFileOpenDialog& dialog, std::filesystem::path const& start)
{
    std::error_code error;
    if (!start.empty() && std::filesystem::is_directory(start, error))
    {
        ComPointer<IShellItem> folder;
        if (SUCCEEDED(SHCreateItemFromParsingName(start.wstring().c_str(), nullptr, IID_PPV_ARGS(folder.Address()))))
        {
            dialog.SetFolder(folder.Get());
        }
    }
    if (FAILED(dialog.Show(GetActiveWindow())))
    {
        return std::nullopt;
    }
    ComPointer<IShellItem> item;
    if (FAILED(dialog.GetResult(item.Address())))
    {
        return std::nullopt;
    }
    PWSTR path = nullptr;
    if (FAILED(item->GetDisplayName(SIGDN_FILESYSPATH, &path)) || path == nullptr)
    {
        return std::nullopt;
    }
    std::filesystem::path const chosen(path);
    CoTaskMemFree(path);
    return chosen;
}

} // namespace

bool NativeDialogsAvailable() noexcept
{
    return true;
}

std::optional<std::filesystem::path> PickFolder(std::filesystem::path const& start, wchar_t const* title)
{
    ComScope const com;
    ComPointer<IFileOpenDialog> dialog;
    if (!com.Ready()
        || FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(dialog.Address()))))
    {
        return std::nullopt;
    }
    FILEOPENDIALOGOPTIONS options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
    dialog->SetTitle(title);
    return Show(*dialog.Get(), start);
}

std::optional<std::filesystem::path> PickComponentLibrary(std::filesystem::path const& start)
{
    ComScope const com;
    ComPointer<IFileOpenDialog> dialog;
    if (!com.Ready()
        || FAILED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(dialog.Address()))))
    {
        return std::nullopt;
    }
    COMDLG_FILTERSPEC const filters[] = {{L"Component libraries (*.dll)", L"*.dll"}, {L"All files", L"*.*"}};
    dialog->SetFileTypes(2, filters);
    FILEOPENDIALOGOPTIONS options = 0;
    dialog->GetOptions(&options);
    dialog->SetOptions(options | FOS_FORCEFILESYSTEM | FOS_FILEMUSTEXIST);
    dialog->SetTitle(L"Add a component library");
    return Show(*dialog.Get(), start);
}

#else

bool NativeDialogsAvailable() noexcept
{
    return false;
}

std::optional<std::filesystem::path> PickFolder(std::filesystem::path const&, wchar_t const*)
{
    return std::nullopt;
}

std::optional<std::filesystem::path> PickComponentLibrary(std::filesystem::path const&)
{
    return std::nullopt;
}

#endif

} // namespace ppokit::studio
