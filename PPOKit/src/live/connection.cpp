/**
 * @file PPOKit/src/live/connection.cpp
 * @brief Non-blocking loopback sockets, for Windows (Winsock) and POSIX systems.
 */
#include "ppokit/live/connection.hpp"

#include <stdexcept>
#include <utility>

#include "socket_platform.hpp"

namespace ppokit::live {

namespace {

/** @brief Lines are small and frequent: send them at once rather than coalescing. */
void DisableDelay(std::intptr_t socket) noexcept
{
    int enabled = 1;
    setsockopt(Native(socket), IPPROTO_TCP, TCP_NODELAY, reinterpret_cast<char const*>(&enabled),
        static_cast<socklen_t>(sizeof(enabled)));
}

} // namespace

Connection::Connection(std::intptr_t socket) noexcept : socket_(socket)
{
    SetNonBlocking(socket_);
    DisableDelay(socket_);
}

std::optional<Connection> Connection::Connect(std::uint16_t port)
{
    StartSockets();
    std::intptr_t const socket = static_cast<std::intptr_t>(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (socket == NoSocket)
    {
        return std::nullopt;
    }
    // A blocking connect to the loopback answers at once: accepted or refused.
    sockaddr_in const address = Loopback(port);
    if (connect(Native(socket), reinterpret_cast<sockaddr const*>(&address), static_cast<socklen_t>(sizeof(address)))
        != 0)
    {
        CloseSocket(socket);
        return std::nullopt;
    }
    return Connection(socket);
}

Connection::Connection(Connection&& other) noexcept
    : socket_(std::exchange(other.socket_, NoSocket)),
      outgoing_(std::move(other.outgoing_)),
      incoming_(std::move(other.incoming_))
{
}

Connection& Connection::operator=(Connection&& other) noexcept
{
    if (this != &other)
    {
        Close();
        socket_ = std::exchange(other.socket_, NoSocket);
        outgoing_ = std::move(other.outgoing_);
        incoming_ = std::move(other.incoming_);
    }
    return *this;
}

Connection::~Connection()
{
    Close();
}

void Connection::Close() noexcept
{
    if (socket_ != NoSocket)
    {
        CloseSocket(socket_);
        socket_ = NoSocket;
    }
}

bool Connection::Open() const noexcept
{
    return socket_ != NoSocket;
}

void Connection::Send(std::string const& line)
{
    if (!Open())
    {
        return;
    }
    outgoing_ += line;
    outgoing_ += '\n';
    if (outgoing_.size() > MaximumPending)
    {
        // The peer does not read: drop it rather than grow without bound.
        Close();
        outgoing_.clear();
        return;
    }
    Flush();
}

void Connection::Flush()
{
    std::size_t written = 0;
    while (Open() && written < outgoing_.size())
    {
        int const count = SendBytes(socket_, outgoing_.data() + written, outgoing_.size() - written);
        if (count > 0)
        {
            written += static_cast<std::size_t>(count);
            continue;
        }
        if (count < 0 && WouldBlock())
        {
            break; // The rest goes at the next Flush.
        }
        Close();
    }
    outgoing_.erase(0, written);
}

std::vector<std::string> Connection::Receive()
{
    std::vector<std::string> lines;
    char buffer[4096];
    while (Open())
    {
        int const count = ReceiveBytes(socket_, buffer, sizeof(buffer));
        if (count > 0)
        {
            incoming_.append(buffer, static_cast<std::size_t>(count));
            continue;
        }
        if (count < 0 && WouldBlock())
        {
            break;
        }
        Close(); // 0: the peer closed; negative: an error.
    }
    std::size_t start = 0;
    for (std::size_t end = incoming_.find('\n'); end != std::string::npos; end = incoming_.find('\n', start))
    {
        lines.push_back(incoming_.substr(start, end - start));
        start = end + 1;
    }
    incoming_.erase(0, start);
    return lines;
}

Listener::Listener(std::uint16_t port)
{
    StartSockets();
    socket_ = static_cast<std::intptr_t>(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP));
    if (socket_ == NoSocket)
    {
        throw std::runtime_error("PPOKit live: cannot create a socket");
    }
    sockaddr_in address = Loopback(port);
    if (bind(Native(socket_), reinterpret_cast<sockaddr const*>(&address), static_cast<socklen_t>(sizeof(address))) != 0
        || listen(Native(socket_), 4) != 0)
    {
        CloseSocket(socket_);
        throw std::runtime_error("PPOKit live: cannot listen on 127.0.0.1:" + std::to_string(port));
    }
    SetNonBlocking(socket_);
    auto length = static_cast<socklen_t>(sizeof(address));
    getsockname(Native(socket_), reinterpret_cast<sockaddr*>(&address), &length);
    port_ = ntohs(address.sin_port);
}

Listener::~Listener()
{
    CloseSocket(socket_);
}

std::optional<Connection> Listener::Accept()
{
    auto const accepted = static_cast<std::intptr_t>(accept(Native(socket_), nullptr, nullptr));
    if (accepted == NoSocket)
    {
        return std::nullopt;
    }
    return Connection(accepted);
}

} // namespace ppokit::live
