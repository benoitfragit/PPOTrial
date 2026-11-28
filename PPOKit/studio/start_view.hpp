/**
 * @file PPOKit/studio/start_view.hpp
 * @brief First screen of the studio: open a project folder, create one, or reopen one of
 *        the folders the user opened before. The studio never searches the disk itself.
 */
#pragma once

#include <array>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::studio {

/** @brief What the user asked for on the start screen. */
struct StartChoice final
{
    enum class Kind
    {
        /** @brief Open the project in `folder`. */
        Open,
        /** @brief Create project `name` in `folder`, then open it. */
        Create,
    };
    Kind kind = Kind::Open;
    std::filesystem::path folder;
    std::string name;
};

class StartView final
{
public:
    /** @param recentFile Where the recent folders are kept (editor/recent_projects.hpp). */
    explicit StartView(std::filesystem::path recentFile);

    void Draw();

    /** @brief The choice made since the last call. */
    std::optional<StartChoice> TakeChoice();

    /** @brief Shows why the last choice failed. */
    void ShowError(std::string message) { error_ = std::move(message); }

    /** @brief Reads the recent folders again (after a project was opened). */
    void Refresh();

    std::vector<std::filesystem::path> const& Recent() const noexcept { return recent_; }

private:
    /** @brief Height of the content, to center it in the window. */
    float EstimatedHeight() const;
    void DrawBrand(float width);
    void DrawActions(float width);
    void OpenFolder();
    void DrawOpenByPath(float width);
    void DrawCreate(float width);
    void DrawRecent(float width);
    bool DrawRecentRow(std::filesystem::path const& folder, float width);

    std::filesystem::path recentFile_;
    std::vector<std::filesystem::path> recent_;
    std::optional<StartChoice> choice_;
    std::string error_;
    bool creating_ = false;
    /** @brief Without a native folder picker, "Open project" shows a path field. */
    bool typingPath_ = false;
    std::array<char, 512> pathBuffer_{};
    std::array<char, 64> nameBuffer_{};
    std::array<char, 512> locationBuffer_{};
};

} // namespace ppokit::studio
