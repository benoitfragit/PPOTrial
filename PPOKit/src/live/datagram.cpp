/**
 * @file PPOKit/src/live/datagram.cpp
 * @brief Non-blocking UDP sockets for the Visualiseur frames.
 */
#include "ppokit/live/datagram.hpp"

#include "socket_platform.hpp"

#include <cstring>
#include <stdexcept>

namespace ppokit::live {

DatagramSender::DatagramSender(std::string const& host, std::uint16_t port)
{
    StartSockets();
    sockaddr_in address = Loopback(port);
    if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1)
    {
        throw std::runtime_error("PPOKit datagram: '" + host + "' is not an IPv4 address");
    }
    socket_ = static_cast<std::intptr_t>(::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP));
    if (socket_ == NoSocket)
    {
        throw std::runtime_error("PPOKit datagram: cannot create a socket");
    }
    SetNonBlocking(socket_);
    address_.resize(sizeof(address));
    std::memcpy(address_.data(), &address, sizeof(address));
}

DatagramSender::~DatagramSender()
{
    CloseSocket(socket_);
}

bool DatagramSender::Send(void const* data, std::size_t size) noexcept
{
    if (size > MaximumDatagram)
    {
        return false;
    }
    auto const sent = sendto(Native(socket_), static_cast<char const*>(data), static_cast<int>(size), 0,
        reinterpret_cast<sockaddr const*>(address_.data()), static_cast<socklen_t>(address_.size()));
    return sent == static_cast<decltype(sent)>(size);
}

DatagramReceiver::DatagramReceiver(std::uint16_t port)
{
    StartSockets();
    socket_ = static_cast<std::intptr_t>(::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP));
    if (socket_ == NoSocket)
    {
        throw std::runtime_error("PPOKit datagram: cannot create a socket");
    }
    sockaddr_in address = Loopback(port);
    // Any interface: a viewer may watch a run on another machine.
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(Native(socket_), reinterpret_cast<sockaddr const*>(&address), static_cast<socklen_t>(sizeof(address)))
        != 0)
    {
        CloseSocket(socket_);
        throw std::runtime_error("PPOKit datagram: port " + std::to_string(port) + " is taken");
    }
    SetNonBlocking(socket_);
    auto length = static_cast<socklen_t>(sizeof(address));
    getsockname(Native(socket_), reinterpret_cast<sockaddr*>(&address), &length);
    port_ = ntohs(address.sin_port);
}

DatagramReceiver::~DatagramReceiver()
{
    CloseSocket(socket_);
}

std::size_t DatagramReceiver::Receive(unsigned char* buffer) noexcept
{
    auto const received = recvfrom(Native(socket_), reinterpret_cast<char*>(buffer), static_cast<int>(MaximumDatagram),
        0, nullptr, nullptr);
    return received > 0 ? static_cast<std::size_t>(received) : 0;
}

} // namespace ppokit::live
