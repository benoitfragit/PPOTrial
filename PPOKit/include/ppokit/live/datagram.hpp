/**
 * @file PPOKit/include/ppokit/live/datagram.hpp
 * @brief UDP datagrams that never block: what a Visualiseur sends, what a viewer receives.
 * @details A lost datagram is not resent: a view shows the last frame it received, and the
 * run never waits for a viewer. Datagrams larger than MaximumSize are not sent.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ppokit::live {

/** @brief Largest datagram sent or received (well below the 64 KiB UDP limit). */
inline constexpr std::size_t MaximumDatagram = 60000;

/** @brief Sends datagrams to one host and port. */
class DatagramSender final
{
public:
    /**
     * @param host IPv4 address, e.g. "127.0.0.1".
     * @throws std::runtime_error when the address or the socket is not valid.
     */
    DatagramSender(std::string const& host, std::uint16_t port);
    ~DatagramSender();
    DatagramSender(DatagramSender const&) = delete;
    DatagramSender& operator=(DatagramSender const&) = delete;

    /** @brief Sends one datagram. @return False when it was not sent (too large, buffer full). */
    bool Send(void const* data, std::size_t size) noexcept;

private:
    std::intptr_t socket_;
    std::vector<unsigned char> address_;
};

/** @brief Receives datagrams on a local port. */
class DatagramReceiver final
{
public:
    /**
     * @param port Port to listen on; 0 lets the system choose (see Port()).
     * @throws std::runtime_error when the port is taken.
     */
    explicit DatagramReceiver(std::uint16_t port);
    ~DatagramReceiver();
    DatagramReceiver(DatagramReceiver const&) = delete;
    DatagramReceiver& operator=(DatagramReceiver const&) = delete;

    std::uint16_t Port() const noexcept { return port_; }
    /**
     * @brief Takes the next datagram waiting, if any.
     * @return Its size, or 0 when none is waiting. `buffer` must hold MaximumDatagram bytes.
     */
    std::size_t Receive(unsigned char* buffer) noexcept;

private:
    std::intptr_t socket_;
    std::uint16_t port_ = 0;
};

} // namespace ppokit::live
