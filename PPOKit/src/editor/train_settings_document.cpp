/**
 * @file PPOKit/src/editor/train_settings_document.cpp
 * @brief Editing the PPO section of train.json, and the plain-language advice on it.
 */
#include "ppokit/editor/train_settings_document.hpp"

#include "ppokit/artifacts/atomic_file.hpp"
#include "ppokit/schema/train_schema.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ppokit::editor {
namespace {

using Json = nlohmann::ordered_json;
using core::PpoParameter;

constexpr char const* TrainingSchemaVersion = "ppokit.train/1";

/** @brief A value as JSON: an integer for whole-number parameters, so the file reads 4096, not 4096.0. */
Json ValueJson(PpoParameter parameter, double value)
{
    if (core::SpecOf(parameter).integral)
    {
        return Json(static_cast<std::int64_t>(value));
    }
    return Json(value);
}

/** @brief A number as a user would read it (0.0003, 4096, 0.99). */
std::string Shown(double value)
{
    std::ostringstream text;
    text << std::setprecision(10) << value;
    return text.str();
}

std::string NameOf(PpoParameter parameter)
{
    return core::SpecOf(parameter).name;
}

} // namespace

TrainSettingsDocument TrainSettingsDocument::Open(std::string_view text, std::string_view sourceName)
{
    TrainSettingsDocument document;
    if (text.empty())
    {
        document.others_ = Json::object();
        document.others_["schema"] = TrainingSchemaVersion;
    }
    else
    {
        // Validates the schema key and every PPO value, with a message naming the key.
        document.values_ = schema::ParsePpoParameters(text, sourceName);
        document.probeEvery_ = schema::ParseProbeEvery(text, sourceName);
        document.others_ = Json::parse(text);
        document.others_.erase("probes");
        if (document.others_.contains("ppo"))
        {
            for (auto const& item : document.others_.at("ppo").items())
            {
                document.set_[Index(*core::FindPpoParameter(item.key()))] = true;
            }
            document.others_.erase("ppo");
        }
    }
    document.Render();
    document.savedText_ = text.empty() ? std::string() : document.text_;
    return document;
}

TrainSettingsDocument TrainSettingsDocument::Load(std::filesystem::path const& path)
{
    if (!std::filesystem::exists(path))
    {
        return Open({}, path.filename().string());
    }
    std::ifstream stream(path, std::ios::binary);
    std::string const text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    return Open(text, path.filename().string());
}

void TrainSettingsDocument::Set(PpoParameter parameter, double value, bool continuing)
{
    if (!continuing || undo_.empty())
    {
        Remember();
    }
    values_.Set(parameter, value);
    set_[Index(parameter)] = true;
    Render();
}

void TrainSettingsDocument::Reset(PpoParameter parameter)
{
    Remember();
    values_.Set(parameter, core::SpecOf(parameter).defaultValue);
    set_[Index(parameter)] = false;
    Render();
}

void TrainSettingsDocument::ResetAll()
{
    Remember();
    values_ = core::PpoParameterValues{};
    set_.fill(false);
    Render();
}

void TrainSettingsDocument::SetProbeEvery(std::size_t every)
{
    if (every == probeEvery_)
    {
        return;
    }
    Remember();
    probeEvery_ = every;
    Render();
}

void TrainSettingsDocument::AttachTask(schema::TaskSchema task)
{
    bool const clean = !Dirty();
    // Validate the original document before the editor can normalize or discard any fields.
    schema::ParseTrainingPlan(text_, "train.json", task);
    if (others_.contains("curriculum"))
    {
        curriculum_ = ReadCurriculum(others_.at("curriculum"), task);
        others_.erase("curriculum");
    }
    task_ = std::move(task);
    Render();
    if (clean)
    {
        savedText_ = text_;
    }
}

void TrainSettingsDocument::SetCurriculum(std::optional<EditableCurriculum> curriculum, bool continuing)
{
    if (!task_)
    {
        throw std::logic_error("PPOKit train settings: attach the task before editing the curriculum");
    }
    if (!continuing || undo_.empty())
    {
        Remember();
    }
    curriculum_ = std::move(curriculum);
    Render();
}

std::string TrainSettingsDocument::CurriculumProblem() const
{
    if (!task_)
    {
        return {};
    }
    return CheckCurriculum(text_, *task_);
}

void TrainSettingsDocument::Undo()
{
    if (undo_.empty())
    {
        return;
    }
    redo_.push_back({values_, set_, probeEvery_, curriculum_});
    Restore(undo_.back());
    undo_.pop_back();
}

void TrainSettingsDocument::Redo()
{
    if (redo_.empty())
    {
        return;
    }
    undo_.push_back({values_, set_, probeEvery_, curriculum_});
    Restore(redo_.back());
    redo_.pop_back();
}

void TrainSettingsDocument::Remember()
{
    undo_.push_back({values_, set_, probeEvery_, curriculum_});
    redo_.clear();
}

void TrainSettingsDocument::Restore(State const& state)
{
    values_ = state.values;
    set_ = state.set;
    probeEvery_ = state.probeEvery;
    curriculum_ = state.curriculum;
    Render();
}

void TrainSettingsDocument::Save(std::filesystem::path const& path)
{
    // Use the same validation as the learner, including relationships between PPO settings.
    if (task_)
    {
        schema::ParseTrainingPlan(text_, path.filename().string(), *task_);
    }
    else
    {
        // The standalone PPO editor keeps a curriculum opaque until its task is attached.
        // Validate every other section without interpreting the preserved scenario parameters.
        auto root = Json::parse(text_);
        root.erase("curriculum");
        schema::ParseTrainingPlan(root.dump(), path.filename().string(), schema::TaskSchema{});
    }
    artifacts::WriteFileAtomically(
        path, [this](std::ostream& out) { out << text_; }, artifacts::AtomicWriteOptions{}, "PPOKit train.json");
    savedText_ = text_;
}

void TrainSettingsDocument::Render()
{
    // Order of the file: schema, then ppo, then the other sections as they were.
    Json root = Json::object();
    root["schema"] = others_.contains("schema") ? others_.at("schema") : Json(TrainingSchemaVersion);
    Json ppo = Json::object();
    for (std::size_t index = 0; index < core::PpoParameterCount; ++index)
    {
        if (set_[index])
        {
            auto const parameter = static_cast<PpoParameter>(index);
            ppo[core::PpoParameterSpecs[index].name] = ValueJson(parameter, values_[parameter]);
        }
    }
    if (!ppo.empty())
    {
        root["ppo"] = std::move(ppo);
    }
    if (probeEvery_ != 0)
    {
        root["probes"] = Json{{"every", probeEvery_}};
    }
    for (auto const& item : others_.items())
    {
        if (item.key() != "schema")
        {
            root[item.key()] = item.value();
        }
    }
    if (curriculum_ && task_)
    {
        root["curriculum"] = WriteCurriculum(*curriculum_, *task_);
    }
    text_ = root.dump(2) + "\n";
}

std::vector<PpoAdvice> CheckPpoParameters(core::PpoParameterValues const& values)
{
    std::vector<PpoAdvice> advice;
    double const rollout = values[PpoParameter::RolloutSteps];
    double const minibatch = values[PpoParameter::Minibatch];
    double const epochs = values[PpoParameter::Epochs];

    if (minibatch > rollout)
    {
        advice.push_back({AdviceLevel::Warning, {PpoParameter::Minibatch, PpoParameter::RolloutSteps},
            "The minibatch (" + Shown(minibatch) + ") is larger than the rollout (" + Shown(rollout)
                + "): each epoch is a single step on the whole rollout. Lower " + NameOf(PpoParameter::Minibatch)
                + " or raise " + NameOf(PpoParameter::RolloutSteps) + "."});
    }
    else if (std::fmod(rollout, minibatch) != 0.0)
    {
        double const remainder = std::fmod(rollout, minibatch);
        advice.push_back({AdviceLevel::Warning, {PpoParameter::Minibatch, PpoParameter::RolloutSteps},
            Shown(rollout) + " steps do not split evenly into minibatches of " + Shown(minibatch)
                + ": the last one of each epoch has only " + Shown(remainder)
                + " samples and a noisier step. A divisor of the rollout is better."});
    }

    double const steps = epochs * std::ceil(rollout / minibatch);
    if (steps > 20000.0)
    {
        advice.push_back({AdviceLevel::Warning,
            {PpoParameter::Epochs, PpoParameter::Minibatch, PpoParameter::RolloutSteps},
            Shown(steps) + " optimizer steps per update: updates will be slow and move far from the data. "
                "Fewer epochs or a larger minibatch."});
    }

    double const actorRate = values[PpoParameter::ActorLearningRate];
    if (actorRate > 3e-3)
    {
        advice.push_back({AdviceLevel::Warning, {PpoParameter::ActorLearningRate},
            "An actor learning rate of " + Shown(actorRate)
                + " is large for PPO: watch approx_kl, and lower it if the returns collapse."});
    }
    double const criticRate = values[PpoParameter::CriticLearningRate];
    if (criticRate > 1e-2)
    {
        advice.push_back({AdviceLevel::Warning, {PpoParameter::CriticLearningRate},
            "A critic learning rate of " + Shown(criticRate)
                + " is large: the value loss may oscillate instead of decreasing."});
    }
    if (values[PpoParameter::Clip] > 0.4)
    {
        advice.push_back({AdviceLevel::Warning, {PpoParameter::Clip},
            "A clip range above 0.4 barely limits the updates; 0.1 to 0.3 is the usual range."});
    }
    if (values[PpoParameter::EntropyCoefficient] > 0.05)
    {
        advice.push_back({AdviceLevel::Warning, {PpoParameter::EntropyCoefficient},
            "An entropy bonus above 0.05 can keep the policy random: the returns may never settle."});
    }

    // What the values mean, for a reader learning PPO.
    double const gamma = values[PpoParameter::Gamma];
    if (gamma >= 1.0)
    {
        advice.push_back({AdviceLevel::Warning, {PpoParameter::Gamma},
            "gamma = 1 never discounts: returns of long episodes grow without bound and the critic struggles. "
            "0.99 to 0.999 is safer."});
    }
    else
    {
        double const horizon = 1.0 / (1.0 - gamma);
        advice.push_back({AdviceLevel::Info, {PpoParameter::Gamma},
            "With gamma = " + Shown(gamma) + ", the policy looks about " + Shown(std::round(horizon))
                + " steps ahead: rewards further away count little."});
    }
    advice.push_back({AdviceLevel::Info, {PpoParameter::RolloutSteps, PpoParameter::Minibatch, PpoParameter::Epochs},
        "Each update collects " + Shown(rollout) + " steps, then makes " + Shown(steps) + " optimizer steps ("
            + Shown(epochs) + " epochs)."});
    return advice;
}

} // namespace ppokit::editor
