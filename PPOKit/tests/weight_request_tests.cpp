/**
 * @file PPOKit/tests/weight_request_tests.cpp
 * @brief Live reward weights: the studio's draft and request, the run's inbox and
 *        answer, refusals, stale answers of an earlier session, and the journal flag.
 */
#include "ppokit/editor/live_weights.hpp"
#include "ppokit/runtime/run_log.hpp"
#include "ppokit/runtime/weight_request.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

using namespace ppokit;
using State = editor::LiveWeights::State;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

runtime::RunHeader Header()
{
    runtime::RunHeader header;
    header.task = "flight";
    header.channels = {{"Path", "Path", 1.0}, {"Crash", "Crash", -10.0}};
    header.acceptsWeightRequests = true;
    return header;
}

void RoundTrip(std::filesystem::path const& journal)
{
    runtime::RunLogWriter writer(journal, Header());
    Check(runtime::LoadRunLog(journal).header.acceptsWeightRequests, "the journal says the run reads requests");

    runtime::WeightRequestInbox inbox(journal, {"Path", "Crash"});
    Check(!inbox.Take().has_value(), "no request: nothing taken");

    editor::LiveWeights weights(journal);
    weights.Sync(runtime::LoadRunLog(journal));
    Check(weights.Channels() == 2 && weights.Current(1) == -10.0 && !weights.AnyEdited(), "starts from the journal");
    weights.SetDraft(1, -20.0);
    weights.SetDraft(0, 1.0);
    Check(weights.Edited(1) && !weights.Edited(0), "only a changed weight counts as edited");
    Check(weights.Send() && weights.Status() == State::Pending, "sent, waiting for the run");

    // The run, at its next update boundary.
    auto const request = inbox.Take();
    Check(request && request->weights.size() == 1 && request->weights[0].first == 1
            && request->weights[0].second == -20.0,
        "the run takes only the edited channel");
    Check(!std::filesystem::exists(runtime::WeightRequestPath(journal)), "a taken request is removed");
    writer.WriteWeights({4, {1.0, -10.0}, {1.0, -20.0}});
    inbox.Answer({request->id, true, "Applied from update 4."});

    weights.Poll();
    weights.Sync(runtime::LoadRunLog(journal));
    Check(weights.Status() == State::Applied && weights.Message() == "Applied from update 4.", "the answer is read");
    Check(weights.Current(1) == -20.0 && !weights.AnyEdited(), "the applied weight is now the one in force");

    // A request the run cannot apply.
    {
        std::ofstream(runtime::WeightRequestPath(journal))
            << R"({ "schema": "ppokit.weights/1", "id": 99, "weights": { "Speed": 2 } })";
    }
    Check(!inbox.Take().has_value(), "an unknown channel is not returned");
    auto const refused = runtime::ReadWeightAnswer(journal);
    Check(refused && refused->id == 99 && !refused->applied
            && refused->message.find("'Speed' is not a reward channel") != std::string::npos,
        "and it is refused with the reason");
    Check(!std::filesystem::exists(runtime::WeightRequestPath(journal)), "a refused request is consumed too");

    // A studio opened later must not take the old answer for its own.
    editor::LiveWeights later(journal);
    later.Sync(runtime::LoadRunLog(journal));
    later.SetDraft(0, 3.0);
    later.Send();
    later.Poll();
    Check(later.Status() == State::Pending, "an earlier answer does not answer a new request");
    Check(inbox.Take()->id == 100, "request ids continue after the last answer");
}

} // namespace

int main()
{
    std::filesystem::path const folder = std::filesystem::temp_directory_path() / "ppokit_weight_request_tests";
    std::error_code ignored;
    std::filesystem::remove_all(folder, ignored);
    std::filesystem::create_directories(folder);
    try
    {
        RoundTrip(folder / "metrics.jsonl");
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    std::filesystem::remove_all(folder, ignored);
    if (failures != 0)
    {
        std::cerr << failures << " weight request check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit weight request tests passed\n";
    return EXIT_SUCCESS;
}
