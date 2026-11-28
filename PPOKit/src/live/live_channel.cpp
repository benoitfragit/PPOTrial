/**
 * @file PPOKit/src/live/live_channel.cpp
 * @brief Live channel between a run and the studio: journal, statuses and commands.
 */
#include "ppokit/live/live_channel.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <system_error>

#include <nlohmann/json.hpp>

namespace ppokit::live {

namespace {

using Json = nlohmann::ordered_json;

constexpr char const* LiveSchemaVersion = "ppokit.live/1";

/** @brief The whole journal file, or empty when it does not exist yet. */
std::string ReadJournal(std::filesystem::path const& journal)
{
    std::ifstream file(journal, std::ios::binary);
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

std::string StatusLine(RunStatus const& status)
{
    Json line = Json::object();
    line["event"] = "status";
    line["state"] = RunStateName(status.state);
    line["update"] = status.update;
    line["steps"] = status.steps;
    line["episodes"] = status.episodes;
    line["window_steps"] = status.windowSteps;
    line["window_rewards"] = status.windowRewards;
    return line.dump();
}

std::optional<RunState> RunStateFromName(std::string const& name) noexcept
{
    for (RunState const state : {RunState::Running, RunState::Paused, RunState::Stopping, RunState::Finished})
    {
        if (name == RunStateName(state))
        {
            return state;
        }
    }
    return std::nullopt;
}

/**
 * @brief Reads a weights command: [[channel, weight], ...], every channel known and every
 *        weight finite.
 * @throws std::invalid_argument naming what is wrong.
 */
runtime::WeightRequest ReadWeights(Json const& json, std::size_t channels)
{
    runtime::WeightRequest request;
    request.id = json.at("id").get<std::uint64_t>();
    for (Json const& entry : json.at("weights"))
    {
        auto const channel = entry.at(0).get<std::size_t>();
        double const weight = entry.at(1).get<double>();
        if (channel >= channels || !std::isfinite(weight))
        {
            throw std::invalid_argument("weights: unknown channel " + std::to_string(channel) + " or non-finite weight");
        }
        request.weights.emplace_back(channel, weight);
    }
    return request;
}

bool IsClosed(Connection const& connection) noexcept
{
    return !connection.Open();
}

std::string CommandLine(char const* command)
{
    return Json{{"command", command}}.dump();
}

} // namespace

char const* RunStateName(RunState state) noexcept
{
    switch (state)
    {
    case RunState::Running:
        return "running";
    case RunState::Paused:
        return "paused";
    case RunState::Stopping:
        return "stopping";
    case RunState::Finished:
        return "finished";
    }
    return "running";
}

std::filesystem::path LiveFilePath(std::filesystem::path const& checkpoints)
{
    return checkpoints / "live.json";
}

std::optional<std::uint16_t> AnnouncedPort(std::filesystem::path const& checkpoints)
{
    std::ifstream file(LiveFilePath(checkpoints), std::ios::binary);
    if (!file)
    {
        return std::nullopt;
    }
    Json const announcement = Json::parse(file, nullptr, false);
    if (!announcement.is_object() || announcement.value("schema", std::string()) != LiveSchemaVersion
        || !announcement.contains("port") || !announcement.at("port").is_number_unsigned())
    {
        return std::nullopt;
    }
    return announcement.at("port").get<std::uint16_t>();
}

LiveServer::LiveServer(std::filesystem::path checkpoints, std::filesystem::path journal, std::size_t channels)
    : checkpoints_(std::move(checkpoints)), journal_(std::move(journal)), channels_(channels)
{
    std::filesystem::create_directories(checkpoints_);
    Json const announcement{{"schema", LiveSchemaVersion}, {"port", listener_.Port()}};
    std::ofstream(LiveFilePath(checkpoints_), std::ios::binary) << announcement.dump() << '\n';
}

LiveServer::~LiveServer()
{
    std::error_code ignored;
    std::filesystem::remove(LiveFilePath(checkpoints_), ignored);
}

void LiveServer::Poll(RunStatus const& status)
{
    if (!Due())
    {
        return;
    }
    auto const now = std::chrono::steady_clock::now();
    lastPoll_ = now;
    AcceptClients();
    ReadCommands();
    if (!clients_.empty() && now - lastStatus_ >= StatusInterval)
    {
        SendStatus(status);
    }
    for (Connection& client : clients_)
    {
        client.Flush();
    }
}

void LiveServer::SendStatus(RunStatus const& status)
{
    lastStatus_ = std::chrono::steady_clock::now();
    if (!clients_.empty())
    {
        Broadcast(StatusLine(status));
    }
}

void LiveServer::AcceptClients()
{
    while (std::optional<Connection> client = listener_.Accept())
    {
        // A studio that attaches late first receives the whole journal, then follows it.
        std::string const journal = ReadJournal(journal_);
        std::size_t start = 0;
        for (std::size_t end = journal.find('\n'); end != std::string::npos; end = journal.find('\n', start))
        {
            client->Send(journal.substr(start, end - start));
            start = end + 1;
        }
        clients_.push_back(std::move(*client));
    }
}

void LiveServer::ReadCommands()
{
    for (Connection& client : clients_)
    {
        for (std::string const& line : client.Receive())
        {
            Json const json = Json::parse(line, nullptr, false);
            if (!json.is_object() || !json.contains("command") || !json.at("command").is_string())
            {
                continue; // Not a command: ignored, the run must not stop for a bad line.
            }
            std::string const name = json.at("command").get<std::string>();
            Command command;
            if (name == "pause")
            {
                command.kind = Command::Kind::Pause;
            }
            else if (name == "play")
            {
                command.kind = Command::Kind::Play;
            }
            else if (name == "stop")
            {
                command.kind = Command::Kind::Stop;
            }
            else if (name == "weights")
            {
                command.kind = Command::Kind::Weights;
                try
                {
                    command.weights = ReadWeights(json, channels_);
                }
                catch (std::exception const& error)
                {
                    Answer({json.value("id", std::uint64_t{0}), false, error.what()});
                    continue;
                }
            }
            else
            {
                continue;
            }
            commands_.push_back(std::move(command));
        }
    }
    // Forget the studios that left.
    clients_.erase(std::remove_if(clients_.begin(), clients_.end(), IsClosed), clients_.end());
}

std::optional<Command> LiveServer::TakeCommand()
{
    if (commands_.empty())
    {
        return std::nullopt;
    }
    Command command = std::move(commands_.front());
    commands_.pop_front();
    return command;
}

void LiveServer::Answer(runtime::WeightAnswer const& answer)
{
    Json line = Json::object();
    line["event"] = "weights_answer";
    line["id"] = answer.id;
    line["applied"] = answer.applied;
    line["message"] = answer.message;
    Broadcast(line.dump());
}

void LiveServer::JournalLine(std::string const& line)
{
    Broadcast(line);
}

void LiveServer::Broadcast(std::string const& line)
{
    for (Connection& client : clients_)
    {
        client.Send(line);
    }
}

std::optional<LiveClient> LiveClient::Attach(std::filesystem::path const& checkpoints)
{
    std::optional<std::uint16_t> const port = AnnouncedPort(checkpoints);
    if (!port)
    {
        return std::nullopt;
    }
    std::optional<Connection> connection = Connection::Connect(*port);
    if (!connection)
    {
        return std::nullopt;
    }
    return LiveClient(std::move(*connection));
}

bool LiveClient::Poll()
{
    connection_.Flush();
    std::vector<std::string> const lines = connection_.Receive();
    for (std::string const& line : lines)
    {
        Read(line);
    }
    return !lines.empty();
}

void LiveClient::Read(std::string const& line)
{
    try
    {
        Json const json = Json::parse(line);
        std::string const event = json.at("event").get<std::string>();
        if (event == "status")
        {
            RunStatus status;
            status.state = RunStateFromName(json.at("state").get<std::string>()).value_or(RunState::Running);
            status.update = json.at("update").get<std::uint64_t>();
            status.steps = json.at("steps").get<std::uint64_t>();
            status.episodes = json.at("episodes").get<std::uint64_t>();
            status.windowSteps = json.at("window_steps").get<std::uint64_t>();
            status.windowRewards = json.at("window_rewards").get<std::vector<double>>();
            status_ = std::move(status);
            return;
        }
        if (event == "weights_answer")
        {
            answers_.push_back({json.at("id").get<std::uint64_t>(), json.at("applied").get<bool>(),
                json.at("message").get<std::string>()});
            return;
        }
        runtime::AppendRunLogLine(log_, line);
    }
    catch (std::exception const& error)
    {
        error_ = error.what();
    }
}

std::optional<runtime::WeightAnswer> LiveClient::TakeAnswer()
{
    if (answers_.empty())
    {
        return std::nullopt;
    }
    runtime::WeightAnswer answer = std::move(answers_.front());
    answers_.pop_front();
    return answer;
}

void LiveClient::Pause()
{
    connection_.Send(CommandLine("pause"));
}

void LiveClient::Play()
{
    connection_.Send(CommandLine("play"));
}

void LiveClient::Stop()
{
    connection_.Send(CommandLine("stop"));
}

void LiveClient::SendWeights(runtime::WeightRequest const& request)
{
    Json weights = Json::array();
    for (auto const& [channel, weight] : request.weights)
    {
        weights.push_back(Json::array({channel, weight}));
    }
    connection_.Send(Json{{"command", "weights"}, {"id", request.id}, {"weights", std::move(weights)}}.dump());
}

} // namespace ppokit::live
