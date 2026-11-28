/**
 * @file PPOKit/src/runtime/weight_request.cpp
 * @brief Reading and writing the weight request and answer files of a run folder.
 */
#include "ppokit/runtime/weight_request.hpp"

#include "ppokit/artifacts/atomic_file.hpp"

#include <nlohmann/json.hpp>

#include <cmath>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <system_error>

namespace ppokit::runtime {
namespace {

using Json = nlohmann::ordered_json;

constexpr char const* WeightRequestSchema = "ppokit.weights/1";

std::string ReadWholeFile(std::filesystem::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        throw std::runtime_error("cannot open " + path.string());
    }
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void WriteJsonAtomically(std::filesystem::path const& path, Json const& json, char const* context)
{
    std::string const text = json.dump(2) + "\n";
    artifacts::WriteFileAtomically(
        path, [&text](std::ostream& out) { out << text; }, artifacts::AtomicWriteOptions{}, context);
}

/** @brief Index of `name` in `names`, if present. */
std::optional<std::size_t> ChannelIndex(std::vector<std::string> const& names, std::string const& name)
{
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        if (names[index] == name)
        {
            return index;
        }
    }
    return std::nullopt;
}

/**
 * @brief The request of `json`, resolved against the run's channels.
 * @throws std::runtime_error with a message for the user when it cannot be applied.
 */
WeightRequest ResolveRequest(Json const& json, std::vector<std::string> const& names)
{
    if (!json.is_object() || json.value("schema", std::string()) != WeightRequestSchema)
    {
        throw std::runtime_error(std::string("not a ") + WeightRequestSchema + " request");
    }
    if (!json.contains("weights") || !json.at("weights").is_object() || json.at("weights").empty())
    {
        throw std::runtime_error("the request lists no weight");
    }
    WeightRequest request;
    request.id = json.at("id").get<std::uint64_t>();
    for (auto const& [name, value] : json.at("weights").items())
    {
        auto const index = ChannelIndex(names, name);
        if (!index)
        {
            throw std::runtime_error("'" + name + "' is not a reward channel of this run");
        }
        if (!value.is_number() || !std::isfinite(value.get<double>()))
        {
            throw std::runtime_error("the weight of '" + name + "' is not a finite number");
        }
        request.weights.emplace_back(*index, value.get<double>());
    }
    return request;
}

} // namespace

std::filesystem::path WeightRequestPath(std::filesystem::path const& journal)
{
    return journal.parent_path() / "weights.request.json";
}

std::filesystem::path WeightAnswerPath(std::filesystem::path const& journal)
{
    return journal.parent_path() / "weights.answer.json";
}

void WriteWeightRequest(std::filesystem::path const& journal, std::uint64_t id,
    std::vector<std::pair<std::string, double>> const& weights)
{
    Json request = Json::object();
    request["schema"] = WeightRequestSchema;
    request["id"] = id;
    Json values = Json::object();
    for (auto const& [name, weight] : weights)
    {
        values[name] = weight;
    }
    request["weights"] = std::move(values);
    WriteJsonAtomically(WeightRequestPath(journal), request, "PPOKit weight request");
}

std::optional<WeightAnswer> ReadWeightAnswer(std::filesystem::path const& journal)
{
    std::filesystem::path const path = WeightAnswerPath(journal);
    std::error_code error;
    if (!std::filesystem::exists(path, error))
    {
        return std::nullopt;
    }
    try
    {
        Json const json = Json::parse(ReadWholeFile(path));
        WeightAnswer answer;
        answer.id = json.at("id").get<std::uint64_t>();
        answer.applied = json.at("applied").get<bool>();
        answer.message = json.value("message", std::string());
        return answer;
    }
    catch (std::exception const&)
    {
        return std::nullopt; // being replaced, or written by hand: read it next time
    }
}

WeightRequestInbox::WeightRequestInbox(std::filesystem::path journal, std::vector<std::string> channelNames)
    : journal_(std::move(journal)), channelNames_(std::move(channelNames))
{
}

std::optional<WeightRequest> WeightRequestInbox::Take()
{
    std::filesystem::path const path = WeightRequestPath(journal_);
    std::error_code error;
    if (!std::filesystem::exists(path, error))
    {
        return std::nullopt;
    }
    std::uint64_t id = 0;
    try
    {
        Json const json = Json::parse(ReadWholeFile(path));
        std::filesystem::remove(path, error);
        if (json.is_object() && json.contains("id") && json.at("id").is_number_unsigned())
        {
            id = json.at("id").get<std::uint64_t>();
        }
        return ResolveRequest(json, channelNames_);
    }
    catch (std::exception const& failure)
    {
        // Consumed even when refused: the same bad request must not be answered forever.
        std::filesystem::remove(path, error);
        Answer({id, false, std::string("Refused: ") + failure.what()});
        return std::nullopt;
    }
}

void WeightRequestInbox::Answer(WeightAnswer const& answer) const
{
    Json json = Json::object();
    json["id"] = answer.id;
    json["applied"] = answer.applied;
    json["message"] = answer.message;
    WriteJsonAtomically(WeightAnswerPath(journal_), json, "PPOKit weight answer");
}

} // namespace ppokit::runtime
