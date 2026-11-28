/**
 * @file PPOKit/include/ppokit/live/connection.hpp
 * @brief Local TCP connections carrying lines of text, without ever blocking the caller.
 * @details The live channel between a run and the studio is a socket on the loopback
 * interface (127.0.0.1): it never leaves the machine. Every call returns at once: sending
 * queues the line and writes what the socket accepts, receiving returns the complete lines
 * already arrived. A peer that disappears is reported by Open() becoming false.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::live {

/** @brief One connected socket exchanging newline-terminated lines. */
class Connection final
{
public:
    /**
     * @brief Connects to 127.0.0.1:`port`.
     * @return Nothing when no one listens there.
     */
    static std::optional<Connection> Connect(std::uint16_t port);

    Connection(Connection&& other) noexcept;
    Connection& operator=(Connection&& other) noexcept;
    Connection(Connection const&) = delete;
    Connection& operator=(Connection const&) = delete;
    ~Connection();

    /** @brief Queues `line` (a newline is added) and writes what the socket accepts. */
    void Send(std::string const& line);
    /** @brief Writes what is still queued. */
    void Flush();
    /** @brief Complete lines received since the last call, without their newline. */
    std::vector<std::string> Receive();
    /** @brief False once the peer closed, an error occurred, or too much is queued for it. */
    bool Open() const noexcept;
    /** @brief Bytes queued and not yet written. */
    std::size_t Pending() const noexcept { return outgoing_.size(); }

    /** @brief Largest queue kept for a slow peer before the connection is dropped. */
    static constexpr std::size_t MaximumPending = 16U * 1024U * 1024U;

private:
    friend class Listener;
    explicit Connection(std::intptr_t socket) noexcept;
    void Close() noexcept;

    std::intptr_t socket_;
    std::string outgoing_;
    std::string incoming_;
};

/** @brief A listening socket on 127.0.0.1. */
class Listener final
{
public:
    /**
     * @brief Listens on 127.0.0.1:`port`; 0 lets the system choose a free port.
     * @throws std::runtime_error when the port cannot be opened.
     */
    explicit Listener(std::uint16_t port = 0);
    Listener(Listener const&) = delete;
    Listener& operator=(Listener const&) = delete;
    ~Listener();

    /** @brief The port listened on. */
    std::uint16_t Port() const noexcept { return port_; }
    /** @brief A connection waiting to be accepted, if there is one. */
    std::optional<Connection> Accept();

private:
    std::intptr_t socket_;
    std::uint16_t port_ = 0;
};

} // namespace ppokit::live
