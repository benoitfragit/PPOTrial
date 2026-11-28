/**
 * @file PPOKit/src/live/socket_platform.hpp
 * @brief Internal: the few socket calls PPOKit uses, for Windows (Winsock) and POSIX.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace ppokit::live {

#if defined(_WIN32)
inline constexpr std::intptr_t NoSocket = static_cast<std::intptr_t>(INVALID_SOCKET);

/** @brief Starts Winsock 2.2. @return False when it cannot start. */
inline bool StartWinsock() noexcept
{
    WSADATA data{};
    return WSAStartup(MAKEWORD(2, 2), &data) == 0;
}

/** @brief Winsock is started once per process, before the first socket. */
inline void StartSockets()
{
    static bool const started = StartWinsock();
    if (!started)
    {
        throw std::runtime_error("PPOKit live: cannot start Winsock");
    }
}

inline SOCKET Native(std::intptr_t socket) noexcept
{
    return static_cast<SOCKET>(socket);
}

inline void CloseSocket(std::intptr_t socket) noexcept
{
    closesocket(Native(socket));
}

inline bool SetNonBlocking(std::intptr_t socket) noexcept
{
    u_long enabled = 1;
    return ioctlsocket(Native(socket), FIONBIO, &enabled) == 0;
}

/** @brief True when the last call failed only because it would have blocked. */
inline bool WouldBlock() noexcept
{
    int const error = WSAGetLastError();
    return error == WSAEWOULDBLOCK || error == WSAEINPROGRESS;
}

inline int SendBytes(std::intptr_t socket, char const* data, std::size_t size) noexcept
{
    return send(Native(socket), data, static_cast<int>(size), 0);
}

inline int ReceiveBytes(std::intptr_t socket, char* data, std::size_t size) noexcept
{
    return recv(Native(socket), data, static_cast<int>(size), 0);
}
#else
inline constexpr std::intptr_t NoSocket = -1;

inline void StartSockets() noexcept {}

inline int Native(std::intptr_t socket) noexcept
{
    return static_cast<int>(socket);
}

inline void CloseSocket(std::intptr_t socket) noexcept
{
    close(Native(socket));
}

inline bool SetNonBlocking(std::intptr_t socket) noexcept
{
    int const flags = fcntl(Native(socket), F_GETFL, 0);
    return flags >= 0 && fcntl(Native(socket), F_SETFL, flags | O_NONBLOCK) == 0;
}

inline bool WouldBlock() noexcept
{
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINPROGRESS || errno == EINTR;
}

inline int SendBytes(std::intptr_t socket, char const* data, std::size_t size) noexcept
{
#if defined(MSG_NOSIGNAL)
    int const flags = MSG_NOSIGNAL; // A closed peer must not kill the run with SIGPIPE.
#else
    int const flags = 0;
#endif
    return static_cast<int>(send(Native(socket), data, size, flags));
}

inline int ReceiveBytes(std::intptr_t socket, char* data, std::size_t size) noexcept
{
    return static_cast<int>(recv(Native(socket), data, size, 0));
}
#endif

/** @brief The loopback address with `port`. */
inline sockaddr_in Loopback(std::uint16_t port) noexcept
{
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    return address;
}

} // namespace ppokit::live
