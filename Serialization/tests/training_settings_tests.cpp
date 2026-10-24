/**
 * @file training_settings_tests.cpp
 * @brief V2 training settings invariants and known-good numerical parameters.
 */
#include "training_settings.hpp"
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using ppo::serialization::TrainingSettings;
using ppo::serialization::ReadTrainingSettings;
using ppo::serialization::WriteTrainingSettings;
using ppo::serialization::Serializer;
using Json = ppo::serialization::json;

namespace {
void Check(bool valid, std::string const& message)
{
    if (!valid)
        throw std::runtime_error(message);
}

void Same(TrainingSettings const& a, TrainingSettings const& b, std::string const& name)
{
    Check(a.rolloutSteps == b.rolloutSteps
        && a.checkpointEvery == b.checkpointEvery
        && a.modelVersion == b.modelVersion
        && a.rewardClamp == b.rewardClamp
        && a.rewardWeights == b.rewardWeights
        && a.gamma == b.gamma && a.gaeLambda == b.gaeLambda
        && a.clipEpsilon == b.clipEpsilon
        && a.valueCoefficient == b.valueCoefficient
        && a.entropyCoefficient == b.entropyCoefficient
        && a.learningRate == b.learningRate
        && a.epochs == b.epochs && a.batchSize == b.batchSize
        && a.targetKL == b.targetKL,
        name + ": typed settings are not stable across serialization");
}

void Test(std::string const& name,
          std::vector<std::string> const& rewardNames,
          std::vector<float> const& expectedRewards,
          long long rollout, unsigned int version,
          float clamp, float kl)
{
    auto const path = std::filesystem::path(PPO_SETTINGS_ROOT)
                    / "Training" / (name + ".json");
    Json const source = Serializer::Parse(path);
    auto const actual = ReadTrainingSettings(source, rewardNames);
    Check(actual.rolloutSteps == rollout, name + ": rollout changed");
    Check(actual.checkpointEvery == 5U, name + ": checkpoint cadence changed");
    Check(actual.modelVersion == version, name + ": model version changed");
    Check(actual.rewardClamp == clamp, name + ": reward clamp changed");
    Check(actual.rewardWeights == expectedRewards, name + ": reward weights changed");
    Check(actual.gamma == 0.99F && actual.gaeLambda == 0.95F
        && actual.clipEpsilon == 0.2F
        && actual.valueCoefficient == 0.5F
        && actual.entropyCoefficient == 0.02F
        && actual.learningRate == 0.0003F
        && actual.epochs == 4 && actual.batchSize == 64
        && actual.targetKL == kl,
        name + ": PPO hyperparameters changed");

    Json written{};
    WriteTrainingSettings(written, actual, rewardNames);
    Same(actual, ReadTrainingSettings(written, rewardNames), name);

    // No alternate grammar is accepted. Fail closed for missing or extra
    // numerical settings and reward channels.
    for (auto const& field : {"rolloutSteps", "checkpointEvery"})
    {
        Json invalid = source;
        invalid["training"].erase(field);
        bool rejected = false;
        try { (void)ReadTrainingSettings(invalid, rewardNames); }
        catch (std::exception const&) { rejected = true; }
        Check(rejected, name + ": missing required setting was accepted");
    }

    for (bool const remove : {false, true})
    {
        Json invalid = source;
        if (remove)
            invalid["rewards"]["weights"].erase(rewardNames.front());
        else
            invalid["rewards"]["weights"]["unexpected"] = 0.0;
        bool rejected = false;
        try { (void)ReadTrainingSettings(invalid, rewardNames); }
        catch (std::exception const&) { rejected = true; }
        Check(rejected, name + ": invalid reward contract was accepted");
    }

    Json noVersion = source;
    noVersion.erase("formatVersion");
    bool rejected = false;
    try { (void)ReadTrainingSettings(noVersion, rewardNames); }
    catch (std::exception const&) { rejected = true; }
    Check(rejected, name + ": missing format version was accepted");
}
}

int main()
{
    try
    {
        Test("RLSNA",
             {"Target destruction", "Target in EZ", "Ownship destruction",
              "Closing", "NavPN", "NavAPN", "Fuel", "MissionFailure"},
             {1.F, 0.2160000056028366F, 1.F, 0.07999999821186066F,
              0.7910000085830688F, 0.5529999732971191F,
              0.007000000216066837F, 1.F},
             800, 888U, 10.F, 0.029999999329447746F);
        Test("RLMission",
             {"Main target destruction", "Secondary target destruction",
              "Ownship destruction", "Wingmen destruction",
              "Civilian destruction", "MissionFailure", "Intercept",
              "Danger", "Out Of Area", "Range", "Fuel", "NearDanger"},
             {1.F, 0.15000000596046448F, 1.100000023841858F,
              0.800000011920929F, 0.8999999761581421F, 1.F,
              1.F, 0.699999988079071F, 0.25F, 0.8999999761581421F,
              0.019999999552965164F, 0.4000000059604645F},
             2021, 125U, 60.F, 0.025999998673796654F);
        std::cout << "PASS: V2 training settings and exact numerical invariants\n";
        return 0;
    }
    catch (std::exception const& error)
    {
        std::cerr << "Training settings regression: " << error.what() << '\n';
        return 1;
    }
}
