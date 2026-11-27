/**
 * @file PPOKit/tests/live_channel_tests.cpp
 * @brief The live channel: a studio attaching late receives the whole journal, then each
 *        new event and the statuses; its commands reach the run; a bad command is refused
 *        without stopping it; a studio may leave and come back.
 */
#include "ppokit/live/live_channel.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string>
#include <thread>

namespace {

using namespace ppokit;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/** @brief Polls both sides until `done` holds or two seconds pass. */
bool PumpUntil(live::LiveServer& server, live::LiveClient& client, live::RunStatus const& status,
    std::function<bool()> const& done)
{
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (std::chrono::steady_clock::now() < deadline)
    {
        server.Poll(status);
        client.Poll();
        if (done())
        {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
}

runtime::RunHeader Header()
{
    runtime::RunHeader header;
    header.task = "corridor";
    header.channels = {{"Progress", "Progress", 1.0}, {"Crash", "Crash", -5.0}};
    header.settings = {{"gamma", 0.99}};
    header.acceptsWeightRequests = true;
    return header;
}

runtime::UpdateRecord Update(std::uint64_t number)
{
    runtime::UpdateRecord record;
    record.update = number;
    record.elapsedSeconds = static_cast<double>(number);
    record.ppo = {{"policy_loss", 0.1 * static_cast<double>(number)}, {"entropy", 1.0}};
    record.rewards.steps = 100;
    record.rewards.episodes = 2;
    record.rewards.episodeReturnMean = 3.0;
    record.rewards.channels.resize(2);
    return record;
}

} // namespace

int main()
{
    std::filesystem::path const folder = std::filesystem::temp_directory_path()
        / ("ppokit_live_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(folder);
    std::filesystem::path const journal = folder / "metrics.jsonl";
    try
    {
        Check(!live::LiveClient::Attach(folder), "no run announced: nothing to attach to");
        runtime::RunLogWriter writer(journal, Header());
        writer.WriteUpdate(Update(1));
        live::LiveServer server(folder, journal, 2);
        writer.SetSink(&server);
        Check(live::AnnouncedPort(folder) == server.Port(), "the run announces its port in live.json");

        // A studio attaching after update 1 receives it, then update 2 as it is written.
        std::optional<live::LiveClient> client = live::LiveClient::Attach(folder);
        Check(client.has_value(), "the studio attaches to the announced run");
        if (!client)
        {
            return EXIT_FAILURE;
        }
        live::RunStatus status;
        status.update = 1;
        status.steps = 120;
        status.windowRewards = {0.5, -5.0};
        status.windowSteps = 20;
        Check(PumpUntil(server, *client, status, [&] { return client->HasLog() && client->Log().updates.size() == 1; }),
            "the journal written before attaching is received");
        writer.WriteUpdate(Update(2));
        Check(PumpUntil(server, *client, status, [&] { return client->Log().updates.size() == 2; }),
            "a new update arrives live");
        Check(client->Log().header.channels.size() == 2 && client->Log().updates[1].ppo[0].second == 0.2,
            "the received journal is the run's");
        Check(PumpUntil(server, *client, status,
                  [&] { return client->Status() && client->Status()->windowRewards.size() == 2; }),
            "the status arrives a few times per second");
        Check(client->Status()->steps == 120 && client->Status()->windowRewards[1] == -5.0,
            "the status carries the window's rewards");

        // Commands, in order.
        client->Pause();
        client->Play();
        runtime::WeightRequest request;
        request.id = 7;
        request.weights = {{1, -2.0}};
        client->SendWeights(request);
        client->Stop();
        std::vector<live::Command> received;
        PumpUntil(server, *client, status, [&] {
            while (std::optional<live::Command> command = server.TakeCommand())
            {
                received.push_back(*command);
            }
            return received.size() == 4;
        });
        Check(received.size() == 4 && received[0].kind == live::Command::Kind::Pause
                && received[1].kind == live::Command::Kind::Play && received[2].kind == live::Command::Kind::Weights
                && received[2].weights.id == 7 && received[2].weights.weights[0].second == -2.0
                && received[3].kind == live::Command::Kind::Stop,
            "pause, play, weights and stop reach the run in order");

        server.Answer({7, true, "applied from update 3"});
        Check(PumpUntil(server, *client, status, [&] { return client->TakeAnswer().has_value(); }),
            "the studio receives the answer to its weights");

        // A weights command naming an unknown channel is refused, not applied.
        runtime::WeightRequest unknown;
        unknown.id = 8;
        unknown.weights = {{5, 1.0}};
        client->SendWeights(unknown);
        std::optional<runtime::WeightAnswer> refused;
        PumpUntil(server, *client, status, [&] {
            refused = client->TakeAnswer();
            return refused.has_value();
        });
        Check(refused && refused->id == 8 && !refused->applied && !server.TakeCommand(),
            "an unknown channel is refused without reaching the run");

        // The studio leaves; the run goes on; another studio attaches and gets everything.
        client.reset();
        writer.WriteUpdate(Update(3));
        std::optional<live::LiveClient> again = live::LiveClient::Attach(folder);
        Check(again && PumpUntil(server, *again, status, [&] { return again->Log().updates.size() == 3; }),
            "a studio that comes back receives the whole journal");
        Check(server.Clients() == 1, "the studio that left is forgotten");
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    Check(!std::filesystem::exists(folder / "live.json"), "the announcement goes with the run");
    std::error_code ignored;
    std::filesystem::remove_all(folder, ignored);
    if (failures != 0)
    {
        std::cerr << failures << " live channel check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit live channel tests passed\n";
    return EXIT_SUCCESS;
}
