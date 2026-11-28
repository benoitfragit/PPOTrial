/**
 * @file PPOKit/studio/overview_view.hpp
 * @brief The whole agent on one screen: how the task, the actor, the critic, GAE and the
 *        PPO update connect. Every box opens what it summarizes.
 * @details
 * @verbatim
 *   Curriculum --> Environment --> Observation --+--> Actor  --> Actions --> (back to the environment)
 *                                                |
 *                                                +--> Critic --> Value --> GAE --> PPO update
 *                                     Rewards ----------------------------^           |
 *                                                    updates the actor and the critic <+
 * @endverbatim
 * The actor and the critic are two networks reading the same observation (the critic may
 * read more, for MAPPO): GAE turns the critic's values and the rewards into advantages, and
 * the PPO update trains both from them. Needs a current ImGui context.
 */
#pragma once

#include "ppokit/core/ppo_parameters.hpp"
#include "ppokit/editor/curriculum_editing.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <imgui.h>

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::studio {

/** @brief What a click on a box of the overview opens. */
enum class OverviewTarget
{
    Task,
    Environment,
    Actor,
    Critic,
    Training,
    Curriculum,
    Rewards,
};

/** @brief What the overview summarizes, read from the project's documents. */
struct OverviewModel final
{
    std::optional<schema::TaskSchema> task;
    /** @brief Whether network.json exists and opens. */
    bool hasNetwork = false;
    std::size_t actorBlocks = 0;
    std::size_t criticBlocks = 0;
    /** @brief False when the program compiles its network (shown read-only). */
    bool networkEditable = true;
    core::PpoParameterValues ppo;
    bool hasTraining = false;
    /** @brief The program's own settings file (project.json "settings"): PPO and curriculum live there. */
    std::string settings;
    /** @brief Separate native learner settings, when declared by the project. */
    std::string learnerSettings;
    std::string environment;
    std::vector<std::string> nativeCurriculum;
    std::optional<editor::EditableCurriculum> curriculum;
    std::size_t probeEvery = 0;
};

class OverviewView final
{
public:
    /** @brief Boxes of the diagram, in drawing order. */
    enum class Box : std::size_t
    {
        Curriculum,
        Environment,
        Observation,
        Actor,
        Actions,
        Critic,
        Value,
        Rewards,
        Gae,
        Update,
        Count
    };

    /** @brief Draws the diagram in the remaining space of the current window. */
    void Draw(OverviewModel const& model);

    /** @brief What was clicked since the last call. */
    std::optional<OverviewTarget> TakeChoice();

    /** @brief Screen center of a box in the last Draw (for tests: where to click). */
    ImVec2 BoxCenter(Box box) const noexcept;

private:
    struct Shape final
    {
        ImVec2 minimum{};
        ImVec2 maximum{};
    };

    Shape const& ShapeOf(Box box) const noexcept;
    static float MiddleX(Shape const& shape) noexcept;
    static float MiddleY(Shape const& shape) noexcept;
    /// @brief Places the boxes for a diagram of the given width.
    void Layout(ImVec2 origin, float width);
    void DrawArrows(ImDrawList& drawList) const;
    void DrawBox(ImDrawList& drawList, Box box, OverviewModel const& model);

    std::array<Shape, static_cast<std::size_t>(Box::Count)> shapes_{};
    std::optional<OverviewTarget> choice_;
};

} // namespace ppokit::studio
