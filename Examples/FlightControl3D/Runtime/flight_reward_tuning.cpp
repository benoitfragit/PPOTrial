/**
 * @file flight_reward_tuning.cpp
 * @brief Validation of a weight edit, signature update and atomic rewrite of the scenario JSON.
 *
 * The scenario is rewritten through "<path>.reward-ui.tmp" and "<path>.reward-ui.backup":
 * an existing temporary or backup file aborts the edit, so two editors never race, and a
 * failed rename restores the original file.
 *
 * Read next: flight_reward_tuning.hpp, Core/flight_config.cpp (Signature).
 */
#include "flight_reward_tuning.hpp"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace ppo::examples::flight {
namespace {
using Json = nlohmann::json;
/** @brief Indentation of the rewritten scenario JSON. */
constexpr int ScenarioIndent = 2;

/** @brief Update an existing JSON reward weight. */
void SetJsonValue(Json& reward, std::size_t index, double value)
{
    auto& weights = reward.at("weights");
    weights.at(RewardNames[index]) = value;
}

/** @brief Writes the document to the temporary file, then swaps it with the scenario through a backup. */
void ReplaceScenario(std::filesystem::path const& path, Json const& document,
    std::filesystem::path const& temporary,
    std::filesystem::path const& backup)
{
    {
        std::ofstream output(temporary, std::ios::trunc);
        output << document.dump(ScenarioIndent) << '\n';
        output.flush();
        if (!output)
        {
            throw std::runtime_error("Cannot write reward configuration");
        }
    }
    std::filesystem::rename(path, backup);
    try
    {
        std::filesystem::rename(temporary, path);
    }
    catch (...)
    {
        std::filesystem::rename(backup, path);
        throw;
    }
    std::error_code ignored;
    std::filesystem::remove(backup, ignored);
}
} // namespace

double RewardTuningValue(RewardSettings const& settings, std::size_t index) noexcept
{
    if (index < RewardCount)
    {
        return settings.weights[index];
    }
    return 0;
}

Config WithRewardTuning(Config const& current, std::size_t index, double value)
{
    if (index >= RewardTuningCount || !std::isfinite(value))
    {
        throw std::invalid_argument("Invalid reward setting");
    }
    Config next = current;
    next.reward.weights[index] = value;
    Validate(next);
    // The signature carries the reward section, so a weight edit changes the contract.
    auto signature = Json::parse(next.checkpointSignature);
    SetJsonValue(signature.at("reward"), index, value);
    next.checkpointSignature = signature.dump();
    return next;
}

void SaveRewardTuning(std::filesystem::path const& path, std::size_t index, double value)
{
    if (path.empty() || index >= RewardTuningCount || !std::isfinite(value))
    {
        throw std::invalid_argument("Invalid reward configuration path or value");
    }
    std::ifstream input(path);
    if (!input)
    {
        throw std::runtime_error("Cannot read reward configuration");
    }
    Json document;
    input >> document;
    input.close();
    SetJsonValue(document.at("reward"), index, value);
    auto temporary = path;
    temporary += ".reward-ui.tmp";
    auto backup = path;
    backup += ".reward-ui.backup";
    if (std::filesystem::exists(temporary) || std::filesystem::exists(backup))
    {
        throw std::runtime_error("Reward configuration temporary file already exists");
    }
    try
    {
        ReplaceScenario(path, document, temporary, backup);
    }
    catch (...)
    {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}
} // namespace ppo::examples::flight
