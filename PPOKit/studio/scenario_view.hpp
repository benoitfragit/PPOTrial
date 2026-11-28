/** @file scenario_view.hpp
 * @brief Inspection and validated editing of a project's native scenario and curriculum.
 */
#pragma once

#include "ppokit/schema/project.hpp"
#include "ppokit/environment/environment_library.hpp"
#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::studio {

class ScenarioView final
{
public:
    explicit ScenarioView(schema::Project const& project,
        std::vector<std::filesystem::path> const& programs = {});
    /** @brief True when the native provider can validate scenario changes. */
    bool Editable() const noexcept;
    /** @brief The provider authorizes this scalar path, and editing is currently enabled. */
    bool CanEdit(std::string const& pointer) const noexcept;
    bool Dirty() const noexcept { return dirty_ || externalDirty_; }
    void SetEditingAllowed(bool allowed) noexcept { editingAllowed_ = allowed; }
    /** @brief Change an existing scalar in an editable section; no structural or contract changes. */
    bool SetScalar(std::string const& pointer, nlohmann::ordered_json value);
    /**
     * @brief Change a supported adaptive-gate setting after checking its runtime limits.
     * @param pointer JSON pointer to the existing gate setting.
     * @param value New value of the same JSON type.
     * @return True when the value was accepted into the draft.
     */
    bool SetExternalScalar(std::string const& pointer, nlohmann::ordered_json value);
    /**
     * @brief Check whether an adaptive-gate scalar can be changed in the current state.
     * @param pointer JSON pointer to the existing gate setting.
     * @return True for a supported scalar while editing is enabled.
     */
    bool CanEditExternal(std::string const& pointer) const noexcept;
    /** @brief Validate a staged copy with the native provider before atomically saving. */
    bool Save();
    void Revert();
    std::string const& Status() const noexcept { return status_; }
    /** @brief Reload the configured source files, retaining read errors for display. */
    void Reload();
    void DrawEnvironment();
    void DrawCurriculum();
    void DrawLearnerSettings() const;
    /** @brief Overview rows obtained from the declared curriculum, including read errors. */
    std::vector<std::string> CurriculumSummary() const;

private:
    /** @brief Snapshot the committed metadata when opening or refreshing the native curriculum. */
    void ReloadPublishedState();
    void DrawPublishedState() const;
    void DrawEditStatus();
    bool SaveScenario();
    bool SaveExternal();
    void RevertExternal();

    schema::Project const& project_;
    nlohmann::ordered_json scenario_;
    nlohmann::ordered_json savedScenario_;
    std::shared_ptr<environment::EnvironmentLibrary const> provider_;
    std::string providerError_;
    bool dirty_ = false;
    bool editingAllowed_ = true;
    std::string status_;
    nlohmann::ordered_json learnerSettings_;
    std::string scenarioError_;
    std::string learnerError_;
    std::string curriculumError_;
    nlohmann::ordered_json externalCurriculum_;
    nlohmann::ordered_json savedExternalCurriculum_;
    bool externalDirty_ = false;
    nlohmann::ordered_json externalProgress_;
    std::string externalCurriculumError_;
    std::string externalProgressError_;
    std::optional<std::uint64_t> stateUpdate_;
    nlohmann::ordered_json publishedState_;
    std::string stateError_;
};
} // namespace ppokit::studio
