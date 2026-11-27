/** @file control.cpp
 * @brief Bounded, explicitly serialized control datagrams and a single outstanding command per session.
 */
#include "ppokit/ipc/control.hpp"
#include "../live/socket_platform.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace ppokit::ipc
{
namespace
{
using Clock = std::chrono::steady_clock;
using Session = std::array<std::uint8_t, 16>;
using namespace std::chrono_literals;
constexpr std::size_t Header = 44, Maximum = 1200, ReceiveBudget = 16;
constexpr auto TickInterval = 10ms;
constexpr auto Retry = 120ms;
constexpr auto Heartbeat = 250ms;
constexpr auto Silence = 3s;
constexpr auto CommandDeadline = 10s;
constexpr std::uint32_t Magic = 0x50504B43; // PPKC
enum class Type : std::uint16_t
{
    Hello = 1,
    HelloAck,
    Heartbeat,
    Status,
    Pause,
    Resume,
    Stop,
    SetRewards,
    Ack,
    Nack,
    Stopped,
    Error
};
struct Frame
{
    Type type = Type::Hello;
    std::uint64_t sequence = 0, correlation = 0;
    std::string payload;
};

/** @brief Serialize integers explicitly in network byte order. */
void Put(std::uint8_t *to, std::uint64_t value, std::size_t bytes) noexcept
{
    while (bytes)
    {
        to[--bytes] = static_cast<std::uint8_t>(value);
        value >>= 8;
    }
}
std::uint64_t Get(std::uint8_t const *from, std::size_t bytes) noexcept
{
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < bytes; ++i)
        result = (result << 8) | from[i];
    return result;
}

/** @brief One nonblocking socket, bound to an ephemeral loopback port for both send and receive. */
class Socket final
{
  public:
    Socket()
    {
        live::StartSockets();
        fd_ = static_cast<std::intptr_t>(::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP));
        if (fd_ == live::NoSocket)
            throw std::runtime_error("cannot create control socket");
        sockaddr_in address = live::Loopback(0);
        auto length = static_cast<socklen_t>(sizeof(address));
        if (!live::SetNonBlocking(fd_) ||
            bind(live::Native(fd_), reinterpret_cast<sockaddr *>(&address), length) != 0 ||
            getsockname(live::Native(fd_), reinterpret_cast<sockaddr *>(&address), &length) != 0)
        {
            live::CloseSocket(fd_);
            throw std::runtime_error("cannot bind a nonblocking loopback control socket");
        }
        port = ntohs(address.sin_port);
    }
    ~Socket()
    {
        live::CloseSocket(fd_);
    }
    Socket(Socket const &) = delete;
    Socket &operator=(Socket const &) = delete;

    void Send(std::uint16_t peer, Session const &session, Frame const &frame) const noexcept
    {
        if (!peer || frame.payload.size() > Maximum - Header)
            return;
        std::array<std::uint8_t, Maximum> bytes;
        Put(bytes.data(), Magic, 4);
        Put(bytes.data() + 4, 1, 2);
        Put(bytes.data() + 6, static_cast<std::uint16_t>(frame.type), 2);
        std::copy(session.begin(), session.end(), bytes.begin() + 8);
        Put(bytes.data() + 24, frame.sequence, 8);
        Put(bytes.data() + 32, frame.correlation, 8);
        Put(bytes.data() + 40, frame.payload.size(), 4);
        std::copy(frame.payload.begin(), frame.payload.end(), bytes.begin() + Header);
        auto const address = live::Loopback(peer);
        sendto(live::Native(fd_), reinterpret_cast<char const *>(bytes.data()),
               static_cast<int>(Header + frame.payload.size()), 0, reinterpret_cast<sockaddr const *>(&address),
               sizeof(address));
    }

    /** @brief Ignore malformed packets and cap both bytes and packet count per tick. */
    template <class Handler> void Drain(Session const &session, Handler handle)
    {
        std::array<std::uint8_t, Maximum + 1> bytes;
        for (std::size_t i = 0; i < ReceiveBudget; ++i)
        {
            sockaddr_in source{};
            auto length = static_cast<socklen_t>(sizeof(source));
            auto const n =
                recvfrom(live::Native(fd_), reinterpret_cast<char *>(bytes.data()), static_cast<int>(bytes.size()), 0,
                         reinterpret_cast<sockaddr *>(&source), &length);
            if (n < 0)
                break;
            auto const size = static_cast<std::size_t>(n);
            if (size < Header || size > Maximum || source.sin_addr.s_addr != htonl(INADDR_LOOPBACK) ||
                Get(bytes.data(), 4) != Magic || Get(bytes.data() + 4, 2) != 1 ||
                Get(bytes.data() + 40, 4) != size - Header ||
                !std::equal(session.begin(), session.end(), bytes.begin() + 8))
                continue;
            Frame frame;
            auto const wireType = Get(bytes.data() + 6, 2);
            if (wireType < static_cast<unsigned>(Type::Hello) || wireType > static_cast<unsigned>(Type::Error))
                continue;
            frame.type = static_cast<Type>(wireType);
            frame.sequence = Get(bytes.data() + 24, 8);
            frame.correlation = Get(bytes.data() + 32, 8);
            if (frame.sequence == 0)
                continue;
            frame.payload.assign(reinterpret_cast<char const *>(bytes.data() + Header), size - Header);
            handle(ntohs(source.sin_port), frame);
        }
    }
    std::uint16_t port = 0;

  private:
    std::intptr_t fd_;
};

struct Address
{
    std::uint16_t port{};
    Session session{};
};

/** @brief Parse PORT/32-hex-digits without echoing the capability in errors. */
Address ParseAddress(std::string const &text)
{
    auto const slash = text.find('/');
    auto const invalid = []
    { return std::invalid_argument("invalid control endpoint (expected port/32 hex digits)"); };
    if (slash == std::string::npos || text.size() - slash - 1 != 32)
        throw invalid();
    Address result;
    auto const count = std::from_chars(text.data(), text.data() + slash, result.port);
    if (!result.port || count.ec != std::errc{} || count.ptr != text.data() + slash)
        throw invalid();
    for (std::size_t i = 0; i < result.session.size(); ++i)
    {
        auto const begin = text.data() + slash + 1 + 2 * i;
        auto const hex = std::from_chars(begin, begin + 2, result.session[i], 16);
        if (hex.ec != std::errc{} || hex.ptr != begin + 2)
            throw invalid();
    }
    return result;
}

std::string StateText(State state)
{
    return std::string(1, static_cast<char>(state));
}

} // namespace

char const *Name(State state) noexcept
{
    switch (state)
    {
    case State::Starting:
        return "starting";
    case State::Running:
        return "running";
    case State::Paused:
        return "paused";
    case State::Stopping:
        return "stopping";
    case State::Saving:
        return "saving";
    case State::Stopped:
        return "stopped";
    case State::Error:
        return "error";
    }
    return "unknown";
}

struct Supervisor::Impl
{
    Socket socket;
    Session session{};
    std::uint16_t peer = 0;
    std::uint64_t sequence = 0, lastStatus = 0;
    State state = State::Starting;
    std::string error;
    bool stop = false, accepted = false, deadlineReported = false;
    std::optional<Frame> pending;
    std::optional<std::uint64_t> weightsId;
    std::optional<runtime::WeightAnswer> answer;
    std::optional<CommandResult> result;
    std::uint64_t resultCorrelation = 0;
    Clock::time_point seen{}, sent{}, issued{};

    Impl()
    {
        std::random_device random;
        std::uniform_int_distribution<unsigned> byte(0, 255);
        for (auto &value : session)
            value = static_cast<std::uint8_t>(byte(random));
    }
    bool Queue(Type type, Command command, std::string payload)
    {
        if (type == Type::Stop && stop)
            return true;
        if (stop || (pending && type != Type::Stop) || payload.size() > Maximum - Header)
            return false;
        if (type == Type::Stop)
        {
            stop = true;
            if (weightsId)
                answer = runtime::WeightAnswer{*weightsId, false, "Run is stopping"};
            weightsId.reset();
        }
        pending = Frame{type, ++sequence, 0, std::move(payload)};
        accepted = false;
        deadlineReported = false;
        issued = Clock::now();
        sent = {};
        result = CommandResult{command, CommandState::Pending, {}};
        resultCorrelation = pending->sequence;
        return true;
    }
    void Read(std::uint16_t from, Frame const &frame)
    {
        Clock::time_point const now = Clock::now();
        if (frame.type == Type::Hello && frame.payload.empty())
        {
            if (peer && peer != from && now - seen < Silence)
                return;
            if (peer != from)
                lastStatus = 0;
            peer = from;
            seen = now;
            socket.Send(peer, session, {Type::HelloAck, ++sequence, frame.sequence, {}});
        }
        if (!peer || peer != from)
            return;
        seen = now;
        if (frame.type == Type::Error && frame.sequence > lastStatus && !frame.payload.empty())
        {
            error = frame.payload;
            state = State::Error;
            lastStatus = frame.sequence;
        }
        if ((frame.type == Type::Status || frame.type == Type::Stopped) && frame.sequence > lastStatus &&
            frame.payload.size() == 1 &&
            static_cast<unsigned char>(frame.payload[0]) <= static_cast<unsigned>(State::Error))
        {
            state = static_cast<State>(frame.payload[0]);
            lastStatus = frame.sequence;
        }
        if (frame.type != Type::Ack && frame.type != Type::Nack)
            return;
        bool const current = pending && frame.correlation == pending->sequence;
        bool const late = result && result->state == CommandState::Indeterminate
            && frame.correlation == resultCorrelation;
        if (!current && !late)
            return;
        if (frame.type == Type::Ack && frame.payload == "received")
        {
            if (current)
            {
                accepted = true;
                if (!deadlineReported)
                {
                    result->state = CommandState::Received;
                    result->message = "Received; awaiting application";
                }
            }
            return;
        }
        if (current && weightsId)
            answer = runtime::WeightAnswer{*weightsId, frame.type == Type::Ack, frame.payload};
        if (current)
        {
            weightsId.reset();
            pending.reset();
        }
        result->state = frame.type == Type::Ack ? CommandState::Applied : CommandState::Rejected;
        result->message = frame.payload;
    }
};

Supervisor::Supervisor() : impl_(std::make_unique<Impl>())
{
}
Supervisor::~Supervisor() = default;
std::string Supervisor::Endpoint() const
{
    std::string text = std::to_string(impl_->socket.port) + "/";
    constexpr char hex[] = "0123456789abcdef";
    for (auto const byte : impl_->session)
    {
        text += hex[byte >> 4];
        text += hex[byte & 15];
    }
    return text;
}
void Supervisor::Tick()
{
    auto &s = *impl_;
    s.socket.Drain(s.session, [&](std::uint16_t from, Frame const &f) { s.Read(from, f); });
    auto const now = Clock::now();
    if (s.pending && now - s.issued >= CommandDeadline && !s.deadlineReported)
    {
        s.deadlineReported = true;
        s.result->state = CommandState::Indeterminate;
        s.result->message = s.accepted
            ? "Command received but its application was not confirmed"
            : "Command delivery was not confirmed";
        if (s.weightsId)
            s.answer = runtime::WeightAnswer{*s.weightsId, false, s.result->message};
        if (!s.stop)
        {
            s.weightsId.reset();
            s.pending.reset();
        }
    }
    if (s.peer && now - s.sent >= (s.pending ? Retry : Heartbeat))
    {
        if (s.pending)
            s.socket.Send(s.peer, s.session, *s.pending);
        else
            s.socket.Send(s.peer, s.session, {Type::Heartbeat, ++s.sequence, 0, {}});
        s.sent = now;
    }
}
bool Supervisor::Request(Command command)
{
    if (command == Command::SetRewards)
        return false;
    return impl_->Queue(static_cast<Type>(command), command, {});
}
bool Supervisor::SetRewards(runtime::WeightRequest const &request)
{
    if (!impl_->Queue(Type::SetRewards, Command::SetRewards, nlohmann::json(request.weights).dump()))
        return false;
    impl_->weightsId = request.id;
    return true;
}
std::optional<runtime::WeightAnswer> Supervisor::TakeAnswer()
{
    auto answer = std::move(impl_->answer);
    impl_->answer.reset();
    return answer;
}
std::optional<CommandResult> Supervisor::LastCommandResult() const
{
    return impl_->result;
}
State Supervisor::LastState() const noexcept
{
    return impl_->state;
}
bool Supervisor::Connected() const noexcept
{
    return impl_->peer && Clock::now() - impl_->seen < Silence;
}

std::string const &Supervisor::LastError() const noexcept
{
    return impl_->error;
}

struct Worker::Impl
{
    Address address;
    Socket socket;
    bool canSetRewards;
    std::atomic<bool> done{false}, stop{false}, pause{false};
    std::atomic<State> state{State::Running};
    std::mutex mutex;
    std::condition_variable wake;
    std::string terminalError;
    std::optional<runtime::WeightRequest> weights;
    std::optional<runtime::WeightAnswer> answer;
    std::uint64_t sequence = 0, lastCommand = 0;
    Frame reply;
    bool handshake = false, waitingForWeights = false;
    Clock::time_point lastSeen{}, lastSend{};
    std::thread thread;

    Impl(std::string const &endpoint, bool train)
        : address(ParseAddress(endpoint)), canSetRewards(train), thread(&Impl::Loop, this)
    {
    }
    ~Impl()
    {
        done = true;
        wake.notify_all();
        thread.join();
    }

    void Send(Type type, std::uint64_t correlation, std::string payload)
    {
        socket.Send(address.port, address.session, {type, ++sequence, correlation, std::move(payload)});
    }
    void Reply(Type type, std::string text)
    {
        reply = {type, ++sequence, lastCommand, std::move(text)};
        socket.Send(address.port, address.session, reply);
    }
    void Read(std::uint16_t from, Frame const &frame)
    {
        if (from != address.port)
            return;
        if (frame.type == Type::HelloAck)
            handshake = true;
        if (!handshake)
            return;
        lastSeen = Clock::now();
        if (frame.type < Type::Pause || frame.type > Type::SetRewards)
            return;
        if (frame.sequence == lastCommand)
        {
            socket.Send(address.port, address.session, reply);
            return;
        }
        if (frame.sequence < lastCommand)
            return;
        lastCommand = frame.sequence;
        waitingForWeights = frame.type == Type::SetRewards;
        if (frame.type != Type::SetRewards && !frame.payload.empty())
        {
            Reply(Type::Nack, "Unexpected payload");
            return;
        }
        if (stop && frame.type != Type::Stop)
        {
            Reply(Type::Nack, "Run is stopping");
            return;
        }
        if (frame.type == Type::SetRewards)
        {
            if (!canSetRewards)
            {
                Reply(Type::Nack, "Reward editing requires training");
                return;
            }
            try
            {
                auto entries = nlohmann::json::parse(frame.payload).get<std::vector<runtime::ChannelWeight>>();
                if (entries.empty())
                {
                    Reply(Type::Nack, "No reward weights");
                    return;
                }
                std::lock_guard<std::mutex> lock(mutex);
                weights = runtime::WeightRequest{lastCommand, std::move(entries)};
            }
            catch (nlohmann::json::exception const &)
            {
                Reply(Type::Nack, "Invalid reward values");
                return;
            }
        }
        else
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (frame.type == Type::Stop)
            {
                stop = true;
                weights.reset();
            }
            pause = frame.type == Type::Pause;
        }
        Reply(Type::Ack, "received");
        wake.notify_all();
    }
    void Loop()
    {
        while (!done)
        {
            socket.Drain(address.session, [&](std::uint16_t from, Frame const &f) { Read(from, f); });
            auto const now = Clock::now();
            if (handshake && now - lastSeen >= Silence)
            {
                handshake = false;
                // A detached observer must not leave an autonomous job permanently paused.
                std::lock_guard<std::mutex> lock(mutex);
                pause = false;
                wake.notify_all();
            }
            if (now - lastSend >= Heartbeat)
            {
                if (!handshake)
                    Send(Type::Hello, 0, {});
                else
                    Send(Type::Status, 0, StateText(state.load()));
                lastSend = now;
            }
            if (lastCommand && reply.payload == "received")
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (answer && answer->id == lastCommand)
                {
                    Reply(answer->applied ? Type::Ack : Type::Nack, answer->message.substr(0, Maximum - Header));
                    answer.reset();
                }
                else if (stop && state.load() >= State::Stopping)
                    Reply(Type::Ack, "Stop applied");
                else if (!canSetRewards || !waitingForWeights)
                {
                    if ((pause && state == State::Paused) || (!pause && !stop && state == State::Running))
                        Reply(Type::Ack, "Applied");
                }
            }
            std::this_thread::sleep_for(TickInterval);
        }
        std::lock_guard<std::mutex> lock(mutex);
        if (!terminalError.empty())
            Send(Type::Error, 0, terminalError);
        else if (state == State::Stopped)
            Send(Type::Stopped, 0, StateText(State::Stopped));
    }
};

Worker::Worker(std::string const &endpoint, bool train) : impl_(std::make_unique<Impl>(endpoint, train))
{
}
Worker::~Worker() = default;
bool Worker::Continue()
{
    auto &s = *impl_;
    if (s.pause && !s.stop)
    {
        std::unique_lock<std::mutex> lock(s.mutex);
        s.state = State::Paused;
        s.wake.wait(lock, [&] { return !s.pause || s.stop; });
        s.state = State::Running;
    }
    if (!s.stop)
        return true;
    s.state = State::Stopping;
    return false;
}
bool Worker::Stopping() const noexcept
{
    return impl_->stop.load();
}
void Worker::Saving() noexcept
{
    impl_->state = State::Saving;
}
std::optional<runtime::WeightRequest> Worker::TakeWeights()
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto request = std::move(impl_->weights);
    impl_->weights.reset();
    return request;
}
void Worker::Answer(runtime::WeightAnswer const &answer)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->answer = answer;
}
void Worker::Fail(std::string const &reason)
{
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->terminalError = reason.substr(0, Maximum - Header);
}
void Worker::Finish()
{
    impl_->state = State::Stopped;
}
} // namespace ppokit::ipc
