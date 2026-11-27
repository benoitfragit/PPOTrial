/**
 * @file PPOKit/src/view/view_library.cpp
 * @brief Loading Visualiseur views and feeding them the frames received in UDP.
 */
#include "ppokit/view/view_library.hpp"

#include "ppokit/component/native_module.hpp"

#include <array>
#include <stdexcept>

namespace ppokit::view {

std::shared_ptr<ViewLibrary const> ViewLibrary::Open(std::string const& reference,
    std::filesystem::path const& directory)
{
    std::shared_ptr<ViewLibrary> library(new ViewLibrary());
    library->path_ = component::ResolveNativeModule(reference, directory, "view library");
    library->handle_ = component::OpenNativeModule(library->path_);
    auto const entry = reinterpret_cast<PpokitViewsFunction>(
        component::NativeSymbol(library->handle_, PPOKIT_VIEWS_ENTRY));
    std::string const source = library->path_.string();
    if (entry == nullptr)
    {
        throw std::runtime_error(source + ": not a Visualiseur library (no " + PPOKIT_VIEWS_ENTRY
            + "; export the views with PPOKIT_VIEWS)");
    }
    PpokitViewApi const* const* views = nullptr;
    std::size_t count = 0;
    if (entry(PPOKIT_VIEW_ABI, &views, &count) != 0 || views == nullptr || count == 0)
    {
        throw std::runtime_error(source + ": refuses view ABI " + std::to_string(PPOKIT_VIEW_ABI) + ", rebuild it");
    }
    library->views_.assign(views, views + count);
    return library;
}

ViewLibrary::~ViewLibrary()
{
    if (handle_ != nullptr)
    {
        component::CloseNativeModule(handle_);
    }
}

PpokitViewApi const& ViewLibrary::Find(std::string const& name) const
{
    if (name.empty())
    {
        return *views_.front();
    }
    std::string names;
    for (PpokitViewApi const* view : views_)
    {
        if (name == view->name)
        {
            return *view;
        }
        names += (names.empty() ? "" : ", ") + std::string(view->name);
    }
    throw std::runtime_error(path_.string() + " has no view '" + name + "' (views: " + names + ")");
}

ViewerSession::ViewerSession(std::shared_ptr<ViewLibrary const> library, std::string const& viewName,
    std::uint16_t port, std::string const& configJson)
    : library_(std::move(library)), receiver_(port), buffer_(live::MaximumDatagram)
{
    api_ = &library_->Find(viewName);
    std::array<char, 1024> error{};
    view_ = api_->create(configJson.c_str(), error.data(), error.size());
    if (view_ == nullptr)
    {
        throw std::runtime_error(std::string("view '") + api_->name + "' cannot be created: " + error.data());
    }
}

ViewerSession::~ViewerSession()
{
    api_->destroy(view_);
}

std::size_t ViewerSession::Poll()
{
    std::size_t count = 0;
    while (std::size_t const size = receiver_.Receive(buffer_.data()))
    {
        api_->receive(view_, buffer_.data(), size);
        ++count;
    }
    if (count > 0)
    {
        received_ += count;
        last_ = std::chrono::steady_clock::now();
    }
    return count;
}

void ViewerSession::Input(PpokitViewInput const& input)
{
    api_->input(view_, &input);
}

void ViewerSession::Render(PpokitDraw const& draw)
{
    api_->render(view_, &draw);
}

double ViewerSession::SecondsSinceLast() const noexcept
{
    if (received_ == 0)
    {
        return -1.0;
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - last_).count();
}

} // namespace ppokit::view
