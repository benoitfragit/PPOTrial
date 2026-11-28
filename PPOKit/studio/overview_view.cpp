/**
 * @file PPOKit/studio/overview_view.cpp
 * @brief Drawing of the agent overview: boxes laid out on a grid, arrows between them.
 */
#include "overview_view.hpp"

#include "studio_theme.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace ppokit::studio {
namespace {

using Box = OverviewView::Box;

/** @brief Grid of the diagram: 6 columns, 3 rows. */
constexpr float Columns = 6.F;
constexpr float Gap = 34.F;
constexpr float LinesPerRow = 4.5F;
/** @brief Room above the boxes for two arrow lanes: actions to environment, update to actor. */
constexpr float TopLanes = Gap * 1.8F;
constexpr float MinimumColumnWidth = 150.F;
constexpr float BoxPadding = 12.F;
constexpr float BoxRounding = 12.F;
constexpr float AccentHeight = 4.F;
constexpr float ArrowHead = 9.F;
/** @brief Lines of detail a box shows at most. */
constexpr std::size_t MaximumLines = 6;

/** @brief Grid cell of a box: column, row and spans. */
struct Cell final
{
    float column;
    float row;
    float columns;
    float rows;
};

Cell CellOf(Box box) noexcept
{
    switch (box)
    {
    case Box::Curriculum:
        return {0.F, 2.F, 1.F, 1.F};
    case Box::Environment:
        return {0.F, 0.F, 1.F, 2.F};
    case Box::Observation:
        return {1.F, 0.F, 1.F, 2.F};
    case Box::Actor:
        return {2.F, 0.F, 1.F, 1.F};
    case Box::Actions:
        return {3.F, 0.F, 1.F, 1.F};
    case Box::Critic:
        return {2.F, 1.F, 1.F, 1.F};
    case Box::Value:
        return {3.F, 1.F, 1.F, 1.F};
    case Box::Rewards:
        return {3.F, 2.F, 1.F, 1.F};
    case Box::Gae:
        return {4.F, 1.F, 1.F, 1.F};
    case Box::Update:
        return {5.F, 0.F, 1.F, 2.F};
    case Box::Count:
        break;
    }
    return {0.F, 0.F, 1.F, 1.F};
}

OverviewTarget TargetOf(Box box) noexcept
{
    switch (box)
    {
    case Box::Curriculum:
        return OverviewTarget::Curriculum;
    case Box::Actor:
        return OverviewTarget::Actor;
    case Box::Critic:
    case Box::Value:
        return OverviewTarget::Critic;
    case Box::Rewards:
        return OverviewTarget::Rewards;
    case Box::Gae:
    case Box::Update:
        return OverviewTarget::Training;
    case Box::Environment:
        return OverviewTarget::Environment;
    case Box::Observation:
    case Box::Actions:
    case Box::Count:
        break;
    }
    return OverviewTarget::Task;
}

/** @brief Color of the family of a box: task (teal), actor (blue), critic (violet), training (amber). */
ImVec4 AccentOf(Box box) noexcept
{
    switch (box)
    {
    case Box::Actor:
    case Box::Actions:
        return PrimaryAccent;
    case Box::Critic:
    case Box::Value:
        return ImVec4(0.69F, 0.58F, 1.F, 1.F);
    case Box::Gae:
    case Box::Update:
    case Box::Rewards:
        return ImVec4(0.957F, 0.765F, 0.42F, 1.F);
    case Box::Curriculum:
    case Box::Environment:
    case Box::Observation:
    case Box::Count:
        break;
    }
    return SecondaryAccent;
}

char const* TitleOf(Box box) noexcept
{
    switch (box)
    {
    case Box::Curriculum:
        return "Curriculum";
    case Box::Environment:
        return "Environment";
    case Box::Observation:
        return "Observation";
    case Box::Actor:
        return "Actor";
    case Box::Actions:
        return "Actions";
    case Box::Critic:
        return "Critic";
    case Box::Value:
        return "Value V(s)";
    case Box::Rewards:
        return "Rewards";
    case Box::Gae:
        return "GAE";
    case Box::Update:
        return "PPO update";
    case Box::Count:
        break;
    }
    return "";
}

char const* HelpOf(Box box) noexcept
{
    switch (box)
    {
    case Box::Curriculum:
        return "Open the declared curriculum source and stages, or see why this project has none.";
    case Box::Environment:
        return "The environment module: it starts episodes, applies the actions, and returns the next "
               "observation and the rewards. Its scenario parameters are what the curriculum varies.";
    case Box::Observation:
        return "What the agent sees at each step (task.json \"observation\"): named groups of values, and sets "
               "of entities (tracks, team mates) of varying size. Both networks read it.";
    case Box::Actor:
        return "The policy network: reads the observation, outputs the action distribution. Click to edit its "
               "blocks.";
    case Box::Actions:
        return "What the actor decides (task.json \"actions\"): continuous (Gaussian) or on/off (Bernoulli) heads, mapped to the task's bounds.";
    case Box::Critic:
        return "The value network: estimates how good a state is. Used only for training; it may read more than "
               "the actor (the centralized critic of MAPPO). Click to edit its blocks.";
    case Box::Value:
        return "The critic's estimate of the discounted return from the state. Training compares it with what "
               "really happened.";
    case Box::Rewards:
        return "Reward channels of the task and their weights. The weighted sum is the reward GAE uses. Click to "
               "change the weights of a running training.";
    case Box::Gae:
        return "Generalized Advantage Estimation: how much better an action did than the critic expected, from "
               "the rewards and the values. Gamma discounts the future, lambda trades bias for variance. Click to "
               "edit them.";
    case Box::Update:
        return "The PPO update: after each rollout, a few epochs of minibatch steps move the actor towards the "
               "actions with positive advantage (clipped so that it stays close) and the critic towards the "
               "returns. Click to edit the parameters.";
    case Box::Count:
        break;
    }
    return "";
}

std::string Number(double value)
{
    char text[32];
    std::snprintf(text, sizeof text, "%g", value);
    return text;
}

std::string GroupLine(schema::ObservationGroup const& group)
{
    if (group.isSet)
    {
        return group.name + ": up to " + std::to_string(group.capacity) + " x " + std::to_string(group.features);
    }
    return group.name + ": " + std::to_string(group.features);
}

char const* HeadName(schema::ActionHead head) noexcept
{
    switch (head)
    {
    case schema::ActionHead::Gaussian:
        return "continuous";
    case schema::ActionHead::Bernoulli:
        return "on/off";
    }
    return "?";
}

std::vector<std::string> ObservationLines(schema::TaskSchema const& task)
{
    std::vector<std::string> lines;
    for (schema::ObservationGroup const& group : task.observation.groups)
    {
        lines.push_back(GroupLine(group));
    }
    if (task.critic)
    {
        lines.push_back("critic also reads " + std::to_string(task.critic->size) + " values");
    }
    return lines;
}

std::vector<std::string> CurriculumLines(OverviewModel const& model)
{
    if (!model.nativeCurriculum.empty()) return model.nativeCurriculum;
    if (!model.curriculum || model.curriculum->stages.empty())
    {
        return {"No PPOKit curriculum", "Click to inspect settings"};
    }
    std::vector<std::string> lines{std::to_string(model.curriculum->stages.size()) + " stages"};
    for (editor::EditableStage const& stage : model.curriculum->stages)
    {
        lines.push_back("  " + stage.name);
    }
    lines.push_back("next at " + Number(100.0 * model.curriculum->promoteAt) + " % success");
    return lines;
}

std::vector<std::string> LinesOf(Box box, OverviewModel const& model)
{
    using core::PpoParameter;
    if (!model.task && box != Box::Gae && box != Box::Update && box != Box::Curriculum && box != Box::Environment)
    {
        return {"task.json missing"};
    }
    // A program with its own settings file: its PPO and curriculum values are not PPOKit's.
    bool const settingsBox = box == Box::Gae || box == Box::Update;
    auto const& learnerSource = model.learnerSettings.empty() ? model.settings : model.learnerSettings;
    if (settingsBox && !model.hasTraining)
    {
        if (learnerSource.empty()) return {"No training settings", "Click for diagnostics"};
        return {"Native learner settings", learnerSource, "Click to inspect"};
    }
    switch (box)
    {
    case Box::Curriculum:
        return CurriculumLines(model);
    case Box::Environment:
        return {model.task ? model.task->name : "task.json missing", model.environment.empty() ? "program-owned environment" : model.environment,
            model.settings.empty() ? "No scenario settings file" : model.settings,
            "Open environment settings"};
    case Box::Observation:
        return ObservationLines(*model.task);
    case Box::Actor:
        if (!model.hasNetwork)
        {
            return {"network.json missing"};
        }
        if (!model.networkEditable)
        {
            return {std::to_string(model.actorBlocks) + " blocks", "compiled into the program"};
        }
        return {std::to_string(model.actorBlocks) + " blocks", "edit the network"};
    case Box::Actions:
    {
        std::vector<std::string> lines;
        for (schema::ActionGroup const& group : model.task->actions)
        {
            lines.push_back(group.name + ": " + HeadName(group.head) + " x" + std::to_string(group.componentNames.size()));
        }
        return lines;
    }
    case Box::Critic:
        if (!model.hasNetwork)
        {
            return {"network.json missing"};
        }
        return {std::to_string(model.criticBlocks) + " blocks",
            model.task->critic ? "centralized input (MAPPO)" : "reads the observation"};
    case Box::Value:
        return {"one number per state", "trained on the returns"};
    case Box::Rewards:
    {
        std::vector<std::string> lines;
        for (schema::RewardChannel const& channel : model.task->rewards)
        {
            // A program with its own settings file weighs the channels there, not with task.json.
            if (!model.settings.empty())
            {
                lines.push_back(channel.label);
                continue;
            }
            // The weight first: a long label is clipped at its end, never the weight.
            lines.push_back("x" + Number(channel.weight) + "  " + channel.label);
        }
        return lines;
    }
    case Box::Gae:
        return {"gamma " + Number(model.ppo[PpoParameter::Gamma]), "lambda " + Number(model.ppo[PpoParameter::Lambda])};
    case Box::Update:
        return {"rollout " + Number(model.ppo[PpoParameter::RolloutSteps]) + " steps",
            "epochs " + Number(model.ppo[PpoParameter::Epochs]) + ", batch " + Number(model.ppo[PpoParameter::Minibatch]),
            "clip " + Number(model.ppo[PpoParameter::Clip]) + ", KL " + Number(model.ppo[PpoParameter::TargetKl]),
            "actor rate " + Number(model.ppo[PpoParameter::ActorLearningRate]),
            "critic rate " + Number(model.ppo[PpoParameter::CriticLearningRate]),
            model.probeEvery > 0 ? "probes every " + std::to_string(model.probeEvery) : "no probes"};
    case Box::Count:
        break;
    }
    return {};
}

ImU32 Color(ImVec4 color, float alpha = 1.F)
{
    return ImGui::ColorConvertFloat4ToU32(ImVec4(color.x, color.y, color.z, color.w * alpha));
}

void Arrow(ImDrawList& drawList, std::vector<ImVec2> const& points, ImU32 color)
{
    drawList.AddPolyline(points.data(), static_cast<int>(points.size()), color, ImDrawFlags_None, 2.F);
    ImVec2 const tip = points.back();
    ImVec2 const from = points[points.size() - 2];
    float const dx = tip.x - from.x;
    float const dy = tip.y - from.y;
    float const length = std::max(std::sqrt(dx * dx + dy * dy), 1.F);
    ImVec2 const direction(dx / length, dy / length);
    ImVec2 const normal(-direction.y, direction.x);
    ImVec2 const back(tip.x - direction.x * ArrowHead, tip.y - direction.y * ArrowHead);
    drawList.AddTriangleFilled(tip, ImVec2(back.x + normal.x * ArrowHead * 0.55F, back.y + normal.y * ArrowHead * 0.55F),
        ImVec2(back.x - normal.x * ArrowHead * 0.55F, back.y - normal.y * ArrowHead * 0.55F), color);
}

} // namespace

void OverviewView::Layout(ImVec2 origin, float width)
{
    float const column = std::max(MinimumColumnWidth, (width - (Columns - 1.F) * Gap) / Columns);
    // A row is as tall as a title and a few lines of text: the diagram keeps its proportions on tall windows.
    float const line = ImGui::GetTextLineHeightWithSpacing();
    float const height = 2.F * BoxPadding + line * (1.25F + LinesPerRow);
    for (std::size_t index = 0; index < shapes_.size(); ++index)
    {
        Cell const cell = CellOf(static_cast<Box>(index));
        ImVec2 const minimum(origin.x + cell.column * (column + Gap), origin.y + TopLanes + cell.row * (height + Gap));
        shapes_[index] = {minimum, ImVec2(minimum.x + cell.columns * column + (cell.columns - 1.F) * Gap,
                                       minimum.y + cell.rows * height + (cell.rows - 1.F) * Gap)};
    }
}

OverviewView::Shape const& OverviewView::ShapeOf(Box box) const noexcept
{
    return shapes_[static_cast<std::size_t>(box)];
}

float OverviewView::MiddleX(Shape const& shape) noexcept
{
    return (shape.minimum.x + shape.maximum.x) * 0.5F;
}

float OverviewView::MiddleY(Shape const& shape) noexcept
{
    return (shape.minimum.y + shape.maximum.y) * 0.5F;
}

ImVec2 OverviewView::BoxCenter(Box box) const noexcept
{
    Shape const& shape = shapes_[static_cast<std::size_t>(box)];
    return ImVec2((shape.minimum.x + shape.maximum.x) * 0.5F, (shape.minimum.y + shape.maximum.y) * 0.5F);
}

void OverviewView::DrawArrows(ImDrawList& drawList) const
{
    ImU32 const flow = Color(MutedText, 0.75F);
    ImU32 const training = Color(ImVec4(0.957F, 0.765F, 0.42F, 1.F), 0.8F);
    Shape const& environment = ShapeOf(Box::Environment);
    Shape const& observation = ShapeOf(Box::Observation);
    Shape const& actor = ShapeOf(Box::Actor);
    Shape const& actions = ShapeOf(Box::Actions);
    Shape const& critic = ShapeOf(Box::Critic);
    Shape const& value = ShapeOf(Box::Value);
    Shape const& gae = ShapeOf(Box::Gae);
    Shape const& update = ShapeOf(Box::Update);
    Shape const& rewards = ShapeOf(Box::Rewards);
    Shape const& curriculum = ShapeOf(Box::Curriculum);
    Arrow(drawList, {ImVec2(MiddleX(curriculum), curriculum.minimum.y), ImVec2(MiddleX(curriculum), environment.maximum.y)},
        flow);
    Arrow(drawList, {ImVec2(environment.maximum.x, MiddleY(environment)), ImVec2(observation.minimum.x, MiddleY(environment))},
        flow);
    Arrow(drawList, {ImVec2(observation.maximum.x, MiddleY(actor)), ImVec2(actor.minimum.x, MiddleY(actor))}, flow);
    Arrow(drawList, {ImVec2(observation.maximum.x, MiddleY(critic)), ImVec2(critic.minimum.x, MiddleY(critic))}, flow);
    Arrow(drawList, {ImVec2(actor.maximum.x, MiddleY(actor)), ImVec2(actions.minimum.x, MiddleY(actor))}, flow);
    // The actions go back to the environment, over the top.
    float const top = actions.minimum.y - Gap * 0.55F;
    Arrow(drawList, {ImVec2(MiddleX(actions), actions.minimum.y), ImVec2(MiddleX(actions), top),
                        ImVec2(MiddleX(environment), top), ImVec2(MiddleX(environment), environment.minimum.y)},
        flow);
    Arrow(drawList, {ImVec2(critic.maximum.x, MiddleY(critic)), ImVec2(value.minimum.x, MiddleY(critic))}, flow);
    Arrow(drawList, {ImVec2(value.maximum.x, MiddleY(value)), ImVec2(gae.minimum.x, MiddleY(value))}, training);
    Arrow(drawList, {ImVec2(rewards.maximum.x, MiddleY(rewards)), ImVec2(MiddleX(gae), MiddleY(rewards)),
                        ImVec2(MiddleX(gae), gae.maximum.y)},
        training);
    Arrow(drawList, {ImVec2(gae.maximum.x, MiddleY(gae)), ImVec2(update.minimum.x, MiddleY(gae))}, training);
    // The update trains both networks: under the critic row, back to the actor and the critic.
    float const bottom = rewards.maximum.y + Gap * 0.45F;
    Arrow(drawList, {ImVec2(MiddleX(update), update.maximum.y), ImVec2(MiddleX(update), bottom),
                        ImVec2(critic.minimum.x + 18.F, bottom), ImVec2(critic.minimum.x + 18.F, critic.maximum.y)},
        training);
    // ... and over the top, in a lane above the actions loop, back to the actor.
    float const high = actor.minimum.y - Gap * 1.35F;
    Arrow(drawList, {ImVec2(MiddleX(update), update.minimum.y), ImVec2(MiddleX(update), high),
                        ImVec2(actor.maximum.x - 18.F, high), ImVec2(actor.maximum.x - 18.F, actor.minimum.y)},
        training);
}

void OverviewView::DrawBox(ImDrawList& drawList, Box box, OverviewModel const& model)
{
    Shape const& shape = shapes_[static_cast<std::size_t>(box)];
    ImVec4 const accent = AccentOf(box);
    ImGui::SetCursorScreenPos(shape.minimum);
    ImGui::PushID(static_cast<int>(box));
    bool const clicked = ImGui::InvisibleButton("box", ImVec2(shape.maximum.x - shape.minimum.x,
                                                            shape.maximum.y - shape.minimum.y));
    bool const hovered = ImGui::IsItemHovered();
    // Diagram cards provide immediate help without relying on the shared hover-delay timer.
    ItemTooltip(HelpOf(box), ImGuiHoveredFlags_None);
    ImGui::PopID();
    if (clicked)
    {
        choice_ = TargetOf(box);
    }
    drawList.AddRectFilled(shape.minimum, shape.maximum, Color(hovered ? ImVec4(0.12F, 0.16F, 0.22F, 1.F) : RaisedSurface),
        BoxRounding);
    drawList.AddRectFilled(shape.minimum, ImVec2(shape.maximum.x, shape.minimum.y + AccentHeight), Color(accent),
        BoxRounding, ImDrawFlags_RoundCornersTop);
    if (hovered)
    {
        drawList.AddRect(shape.minimum, shape.maximum, Color(accent, 0.7F), BoxRounding, ImDrawFlags_None, 1.5F);
    }
    drawList.PushClipRect(shape.minimum, shape.maximum, true);
    float const line = ImGui::GetTextLineHeightWithSpacing();
    ImVec2 cursor(shape.minimum.x + BoxPadding, shape.minimum.y + BoxPadding);
    drawList.AddText(cursor, Color(accent), TitleOf(box));
    cursor.y += line * 1.25F;
    std::vector<std::string> const lines = LinesOf(box, model);
    for (std::size_t index = 0; index < lines.size() && index < MaximumLines; ++index)
    {
        bool const more = index + 1 == MaximumLines && lines.size() > MaximumLines;
        drawList.AddText(cursor, Color(more ? MutedText : ImGui::GetStyleColorVec4(ImGuiCol_Text)),
            more ? "..." : lines[index].c_str());
        cursor.y += line;
    }
    drawList.PopClipRect();
}

void OverviewView::Draw(OverviewModel const& model)
{
    Heading("Overview", "How your agent learns. Click a box to open it; hover it for an explanation.");
    ImVec2 const origin = ImGui::GetCursorScreenPos();
    float const width = ImGui::GetContentRegionAvail().x;
    Layout(origin, width);
    ImDrawList& drawList = *ImGui::GetWindowDrawList();
    DrawArrows(drawList);
    for (std::size_t index = 0; index < shapes_.size(); ++index)
    {
        DrawBox(drawList, static_cast<Box>(index), model);
    }
    // Reserve the full diagram in both directions, including columns beyond a narrow viewport.
    ImGui::SetCursorScreenPos(origin);
    float const diagramWidth = shapes_[static_cast<std::size_t>(Box::Update)].maximum.x - origin.x;
    ImGui::Dummy(ImVec2(std::max(width, diagramWidth),
        shapes_[static_cast<std::size_t>(Box::Curriculum)].maximum.y - origin.y + Gap));
}

std::optional<OverviewTarget> OverviewView::TakeChoice()
{
    std::optional<OverviewTarget> const choice = choice_;
    choice_.reset();
    return choice;
}

} // namespace ppokit::studio
