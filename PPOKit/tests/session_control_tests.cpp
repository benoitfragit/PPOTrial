/** @file session_control_tests.cpp
 * @brief Loopback protocol isolation and retry tests with an independent wire encoder.
 */
#include "ppokit/ipc/control.hpp"
#include "ppokit/live/datagram.hpp"

#include <array>
#include <chrono>
#include <future>
#include <iostream>
#include <stdexcept>
#include <thread>

namespace
{
using namespace ppokit;
using namespace std::chrono_literals;
void Check(bool ok, char const *why)
{
    if (!ok)
        throw std::runtime_error(why);
}

/** @brief Poll until the observable protocol event arrives at a bounded deadline. */
template <class Done> void Await(ipc::Supervisor &host, Done done)
{
    auto const deadline = std::chrono::steady_clock::now() + 4s;
    while (std::chrono::steady_clock::now() < deadline)
    {
        host.Tick();
        if (done())
            return;
        std::this_thread::sleep_for(2ms);
    }
    throw std::runtime_error("control event deadline");
}

/** @brief Independent HELLO fixture, in network byte order with sequence 1. */
void CheckWire()
{
    ipc::Supervisor host;
    auto const endpoint = host.Endpoint();
    auto const cut = endpoint.find('/');
    auto const port = static_cast<std::uint16_t>(std::stoul(endpoint.substr(0, cut)));
    live::DatagramSender peer("127.0.0.1", port), other("127.0.0.1", port);
    std::array<unsigned char, 45> bytes{'P', 'P', 'K', 'C', 0, 1, 0, 1};
    for (std::size_t i = 0; i < 16; ++i)
        bytes[8 + i] = static_cast<unsigned char>(std::stoul(endpoint.substr(cut + 1 + 2 * i, 2), nullptr, 16));
    bytes[31] = 1;
    bytes[5] = 2;
    peer.Send(bytes.data(), 44); // Unsupported version.
    bytes[5] = 1;
    bytes[43] = 1;
    peer.Send(bytes.data(), 44); // No payload byte.
    bytes[43] = 0;
    bytes[8] ^= 1;
    peer.Send(bytes.data(), 44); // Different session.
    bytes[8] ^= 1;
    peer.Send(bytes.data(), 17); // Truncated header.
    host.Tick();
    Check(!host.Connected(), "malformed HELLO established a session");
    peer.Send(bytes.data(), 44);
    Await(host, [&] { return host.Connected(); });
    bytes[7] = 4;
    bytes[31] = 3;
    bytes[43] = 1;
    bytes[44] = static_cast<unsigned char>(ipc::State::Paused);
    peer.Send(bytes.data(), 45);
    Await(host, [&] { return host.LastState() == ipc::State::Paused; });
    bytes[44] = static_cast<unsigned char>(ipc::State::Running);
    bytes[31] = 2;
    peer.Send(bytes.data(), 45); // Reordered earlier status.
    bytes[31] = 4;
    other.Send(bytes.data(), 45); // Correct capability, different source port.
    host.Tick();
    Check(host.LastState() == ipc::State::Paused, "stale or wrong-peer status was accepted");
}

void CheckRetries()
{
    ipc::Supervisor host, isolated;
    ipc::Worker worker(host.Endpoint(), true);
    ipc::Worker other(isolated.Endpoint(), false);
    Await(host, [&] { return host.Connected(); });
    Await(isolated, [&] { return isolated.Connected(); });
    Check(host.SetRewards({42, {{0, 0.0}, {1, -1.5}}}), "first request was refused");
    std::optional<runtime::WeightRequest> request;
    Await(host,
          [&]
          {
              request = worker.TakeWeights();
              return request.has_value();
          });
    Check(request->weights.size() == 2 && request->weights[1].second == -1.5, "reward payload changed");
    // Withhold the application ACK for several retry periods, like a long optimizer update.
    auto const delay = std::chrono::steady_clock::now() + 450ms;
    Await(host, [&] { return std::chrono::steady_clock::now() >= delay; });
    Check(!worker.TakeWeights() && !host.TakeAnswer(), "reward request was applied or confirmed twice");
    worker.Answer({request->id, true, "Applied"});
    std::optional<runtime::WeightAnswer> answer;
    Await(host,
          [&]
          {
              answer = host.TakeAnswer();
              return answer.has_value();
          });
    Check(answer->applied && answer->id == 42, "final reward ACK not correlated");
    host.Request(ipc::Command::Stop);
    Await(host, [&] { return worker.Stopping(); });
    Check(!worker.Continue() && other.Continue(), "stop escaped its session");
    Check(!host.Request(ipc::Command::Resume), "resume cleared a latched stop");
    worker.Saving();
    worker.Finish();
}
void CheckTerminalError()
{
    ipc::Supervisor supervisor;
    {
        ipc::Worker worker(supervisor.Endpoint(), false);
        Await(supervisor, [&] { return supervisor.Connected(); });
        worker.Fail("invalid configuration");
    }
    Await(supervisor, [&] { return supervisor.LastError() == "invalid configuration"; });
}
} // namespace

int main()
{
    try
    {
        CheckWire();
        CheckRetries();
        CheckTerminalError();
    }
    catch (std::exception const &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
