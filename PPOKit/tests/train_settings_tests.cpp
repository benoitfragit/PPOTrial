/**
 * @file PPOKit/tests/train_settings_tests.cpp
 * @brief PPO parameters of train.json: the catalog, the "ppo" section and its messages,
 *        the studio document (kept sections, reset, save) and the plain-language advice.
 */
#include "ppokit/editor/ppo_help.hpp"
#include "ppokit/editor/train_settings_document.hpp"
#include "ppokit/schema/train_schema.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>

namespace {

using ppokit::core::PpoParameter;
using ppokit::core::PpoParameterValues;
using ppokit::editor::AdviceLevel;
using ppokit::editor::TrainSettingsDocument;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

/** @brief The C++ entry point rejects invalid floating-point inputs even without JSON. */
void NativeSettings()
{
    using namespace ppokit::core;
    for (double PpoSettings::*field : {&PpoSettings::actorLearningRate, &PpoSettings::criticLearningRate,
             &PpoSettings::entropyCoefficient, &PpoSettings::targetKl, &PpoSettings::minimumLogStd,
             &PpoSettings::maximumLogStd, &PpoSettings::initialLogStd, &PpoSettings::valueCoefficient,
             &PpoSettings::maxGradientNorm, &PpoSettings::clip, &PpoSettings::gamma, &PpoSettings::lambda})
    {
        for (double bad : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
        {
            PpoSettings settings;
            settings.*field = bad;
            bool refused = false;
            try { ValidatePpoSettings(settings); }
            catch (std::invalid_argument const&) { refused = true; }
            Check(refused, "native non-finite PPO setting is rejected");
        }
    }
}

/** @brief The message of the SchemaError raised by `text`, or empty when it parses. */
std::string ErrorOf(std::string const& text)
{
    try
    {
        ppokit::schema::ParsePpoParameters(text, "train.json");
    }
    catch (ppokit::schema::SchemaError const& error)
    {
        return error.what();
    }
    return {};
}

bool Contains(std::string const& text, std::string const& part)
{
    return text.find(part) != std::string::npos;
}

void CatalogMatchesTheLearners()
{
    PpoParameterValues const defaults;
    ppokit::core::PpoSettings const reference;
    auto const settings = defaults.Settings();
    Check(settings.gamma == reference.gamma && settings.lambda == reference.lambda && settings.clip == reference.clip
            && settings.actorLearningRate == reference.actorLearningRate
            && settings.criticLearningRate == reference.criticLearningRate
            && settings.entropyCoefficient == reference.entropyCoefficient && settings.epochs == reference.epochs
            && settings.minibatch == reference.minibatch && settings.initialLogStd == reference.initialLogStd
            && settings.seed == reference.seed,
        "catalog defaults are the PpoSettings defaults");
    for (auto const& spec : ppokit::core::PpoParameterSpecs)
    {
        Check(ppokit::editor::FindHelp(spec.name) != nullptr, std::string("help text for ") + spec.name);
        Check(spec.minimum <= spec.defaultValue && spec.defaultValue <= spec.maximum,
            std::string("default in range for ") + spec.name);
    }
    PpoParameterValues values;
    values.Set(PpoParameter::Epochs, 3.6);
    values.Set(PpoParameter::Gamma, 7.0);
    Check(values[PpoParameter::Epochs] == 4.0 && values[PpoParameter::Gamma] == 1.0,
        "Set rounds whole-number parameters and clamps to the range");
}

void ReadsThePpoSection()
{
    auto const values = ppokit::schema::ParsePpoParameters(
        R"({ "schema": "ppokit.train/1", "ppo": { "gamma": 0.995, "rollout_steps": 4096 } })", "train.json");
    Check(values[PpoParameter::Gamma] == 0.995 && values.RolloutSteps() == 4096
            && values.IsDefault(PpoParameter::Clip),
        "set values are read, the others keep their default");
    Check(Contains(ErrorOf(R"({ "schema": "ppokit.train/1", "ppo": { "gama": 0.9 } })"), "did you mean 'gamma'"),
        "a typo suggests the closest name");
    Check(Contains(ErrorOf(R"({ "schema": "ppokit.train/1", "ppo": { "zzzzzzzzzz": 1 } })"), "unknown PPO parameter")
            && !Contains(ErrorOf(R"({ "schema": "ppokit.train/1", "ppo": { "zzzzzzzzzz": 1 } })"), "did you mean"),
        "an unrelated name is not given a far-fetched suggestion");
    Check(Contains(ErrorOf(R"({ "schema": "ppokit.train/1", "ppo": { "clip": 3 } })"), "ppo.clip: 3 is outside [0.01, 1]"),
        "an out-of-range value names the key and the range");
    Check(Contains(ErrorOf(R"({ "schema": "ppokit.train/1", "ppo": { "epochs": 2.5 } })"), "expected a whole number"),
        "whole-number parameters refuse fractions");
}

void EditsAndKeepsTheRestOfTheFile()
{
    std::string const original = R"({
  "schema": "ppokit.train/1",
  "curriculum": { "stages": [ { "name": "calm", "sample": {} } ] },
  "ppo": { "epochs": 8 }
})";
    auto document = TrainSettingsDocument::Open(original, "train.json");
    Check(!document.Dirty() && document.IsSet(PpoParameter::Epochs) && !document.IsSet(PpoParameter::Gamma),
        "opening a file: clean, with the parameters it sets");
    document.Set(PpoParameter::Gamma, 0.995);
    document.Reset(PpoParameter::Epochs);
    Check(document.Dirty() && document.Values()[PpoParameter::Epochs] == 4.0, "a reset returns to the default");
    auto const text = document.Text();
    Check(Contains(text, "\"gamma\": 0.995") && !Contains(text, "epochs") && Contains(text, "\"calm\""),
        "written: the set values only, and the curriculum unchanged");
    Check(text.find("\"ppo\"") < text.find("\"curriculum\""), "the ppo section comes right after the schema");
    document.Set(PpoParameter::Gamma, 0.99);
    document.Reset(PpoParameter::Gamma);
    document.Set(PpoParameter::Epochs, 8);
    Check(!document.Dirty(), "undoing every change by hand leaves the document clean");

    // A drag is one change: undone at once, then redone.
    document.Set(PpoParameter::Clip, 0.25);
    document.Set(PpoParameter::Clip, 0.3, true);
    document.Set(PpoParameter::Clip, 0.35, true);
    document.Undo();
    Check(!document.IsSet(PpoParameter::Clip) && !document.Dirty(), "undo reverts a whole drag");
    document.Redo();
    Check(document.Values()[PpoParameter::Clip] == 0.35 && document.Dirty(), "redo replays it");
    document.Undo();
    Check(document.CanUndo() && document.CanRedo(), "both histories are kept");

    // Probes: a section of their own, absent when off.
    document.SetProbeEvery(10);
    Check(Contains(document.Text(), "\"probes\": {\n    \"every\": 10"), "probes on: written with their interval");
    Check(TrainSettingsDocument::Open(document.Text(), "train.json").ProbeEvery() == 10, "probes read back");
    document.Undo();
    Check(document.ProbeEvery() == 0 && !Contains(document.Text(), "probes"), "probes off: the section leaves the file");

    auto const reread = TrainSettingsDocument::Open(document.Text(), "train.json");
    Check(reread.Values()[PpoParameter::Epochs] == 8.0, "what is written reads back");

    std::filesystem::path const path = std::filesystem::temp_directory_path() / "ppokit_train_settings_test.json";
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    auto created = TrainSettingsDocument::Load(path);
    Check(created.Dirty() && Contains(created.Text(), "ppokit.train/1"), "a missing file starts a new document");
    created.Set(PpoParameter::RolloutSteps, 4096);
    created.Save(path);
    Check(!created.Dirty() && TrainSettingsDocument::Load(path).Values().RolloutSteps() == 4096,
        "saving writes the file and cleans the document");
    std::ifstream stream(path);
    std::string const saved((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    Check(Contains(saved, "\"rollout_steps\": 4096") && !Contains(saved, "4096.0"), "whole numbers are written as such");
    stream.close();
    created.Set(PpoParameter::InitialLogStd, 1.0);
    created.Set(PpoParameter::MaximumLogStd, 0.0);
    bool refused = false;
    try
    {
        created.Save(path);
    }
    catch (ppokit::schema::SchemaError const&)
    {
        refused = true;
    }
    std::ifstream unchanged(path);
    std::string const after((std::istreambuf_iterator<char>(unchanged)), std::istreambuf_iterator<char>());
    Check(refused && created.Dirty() && after == saved, "invalid PPO relationships do not overwrite the file");
    unchanged.close();
    std::filesystem::remove(path, ignored);
}

bool HasAdvice(std::vector<ppokit::editor::PpoAdvice> const& advice, AdviceLevel level, std::string const& part)
{
    for (auto const& entry : advice)
    {
        if (entry.level == level && Contains(entry.message, part))
        {
            return true;
        }
    }
    return false;
}

void ExplainsTheValues()
{
    PpoParameterValues values;
    auto advice = ppokit::editor::CheckPpoParameters(values);
    Check(HasAdvice(advice, AdviceLevel::Info, "about 100 steps ahead"), "gamma 0.99 means about 100 steps");
    Check(HasAdvice(advice, AdviceLevel::Info, "collects 2048 steps, then makes 128 optimizer steps"),
        "the work of one update is spelled out");
    for (auto const& entry : advice)
    {
        Check(entry.level == AdviceLevel::Info, "the defaults raise no warning: " + entry.message);
    }
    values.Set(PpoParameter::Minibatch, 4096);
    values.Set(PpoParameter::ActorLearningRate, 1e-2);
    values.Set(PpoParameter::Gamma, 1.0);
    advice = ppokit::editor::CheckPpoParameters(values);
    Check(HasAdvice(advice, AdviceLevel::Warning, "larger than the rollout"), "minibatch larger than the rollout");
    Check(HasAdvice(advice, AdviceLevel::Warning, "large for PPO"), "large actor learning rate");
    Check(HasAdvice(advice, AdviceLevel::Warning, "never discounts"), "gamma 1");
    values.Set(PpoParameter::Minibatch, 1000);
    Check(HasAdvice(ppokit::editor::CheckPpoParameters(values), AdviceLevel::Warning, "only 48 samples"),
        "an uneven split names the size of the last minibatch");
}

} // namespace

int main()
{
    try
    {
        NativeSettings();
        Check(Contains(ErrorOf(R"({"schema":"ppokit.train/1","ppo":{"initial_log_std":1,"maximum_log_std":0}})"),
                  "initial_log_std"), "cross-field settings rejected by the JSON loader");
        CatalogMatchesTheLearners();
        ReadsThePpoSection();
        EditsAndKeepsTheRestOfTheFile();
        ExplainsTheValues();
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0)
    {
        std::cerr << failures << " train settings check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit train settings tests passed\n";
    return EXIT_SUCCESS;
}
