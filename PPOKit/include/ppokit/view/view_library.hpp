/**
 * @file PPOKit/include/ppokit/view/view_library.hpp
 * @brief The viewer's side of a Visualiseur DLL: load its views, feed one with the frames
 *        received in UDP, and have it draw (ppokit/view/view_abi.h).
 * @details No graphics library here: ViewerSession is what ppokit-viewer runs each frame,
 * whatever draws the window; the tests run it with a recording PpokitDraw.
 */
#pragma once

#include "ppokit/live/datagram.hpp"
#include "ppokit/view/view_abi.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace ppokit::view {

/** @brief A loaded Visualiseur DLL and the views it lists. */
class ViewLibrary final
{
public:
    /**
     * @throws std::runtime_error when the file is missing, has no ppokit_views, or was built
     *         for another ABI.
     */
    static std::shared_ptr<ViewLibrary const> Open(std::string const& reference,
        std::filesystem::path const& directory = {});
    ~ViewLibrary();
    ViewLibrary(ViewLibrary const&) = delete;
    ViewLibrary& operator=(ViewLibrary const&) = delete;

    /** @brief The view named `name`, or the first one when `name` is empty. @throws std::runtime_error listing them. */
    PpokitViewApi const& Find(std::string const& name) const;
    std::filesystem::path const& Path() const noexcept { return path_; }

private:
    ViewLibrary() = default;

    std::filesystem::path path_;
    void* handle_ = nullptr;
    std::vector<PpokitViewApi const*> views_;
};

/** @brief One view attached to a UDP port: what the viewer's window shows. */
class ViewerSession final
{
public:
    /**
     * @param library The Visualiseur DLL.
     * @param viewName Its view to show (empty: the first).
     * @param port UDP port the Visualiseur sends to.
     * @param configJson Options handed to the view.
     * @throws std::runtime_error when the view cannot be created or the port is taken.
     */
    ViewerSession(std::shared_ptr<ViewLibrary const> library, std::string const& viewName, std::uint16_t port,
        std::string const& configJson = "{}");
    ~ViewerSession();
    ViewerSession(ViewerSession const&) = delete;
    ViewerSession& operator=(ViewerSession const&) = delete;

    /** @brief Hands every datagram waiting to the view. @return Datagrams received. */
    std::size_t Poll();
    void Input(PpokitViewInput const& input);
    void Render(PpokitDraw const& draw);

    std::uint16_t Port() const noexcept { return receiver_.Port(); }
    std::uint64_t Received() const noexcept { return received_; }
    /** @brief Seconds since the last datagram; negative before the first. */
    double SecondsSinceLast() const noexcept;
    char const* ViewName() const noexcept { return api_->name; }

private:
    std::shared_ptr<ViewLibrary const> library_;
    PpokitViewApi const* api_ = nullptr;
    void* view_ = nullptr;
    live::DatagramReceiver receiver_;
    std::vector<unsigned char> buffer_;
    std::uint64_t received_ = 0;
    std::chrono::steady_clock::time_point last_{};
};

} // namespace ppokit::view
