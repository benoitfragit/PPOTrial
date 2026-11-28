/**
 * @file PPOKit/studio/help_tooltip.cpp
 * @brief Help tooltips shared by the studio views.
 */
#include "help_tooltip.hpp"

#include <imgui.h>

namespace ppokit::studio {

void HelpTooltip(editor::HelpEntry const& help, std::string const& footer)
{
    if (!ImGui::BeginItemTooltip())
    {
        return;
    }
    // Wrapped at about 32 characters of width per em, so that long advice stays readable.
    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 32.F);
    ImGui::TextUnformatted(help.title);
    ImGui::Separator();
    ImGui::TextUnformatted(help.what);
    ImGui::TextDisabled("Typical: %s", help.typical);
    ImGui::TextUnformatted(help.advice);
    if (!footer.empty())
    {
        ImGui::Separator();
        ImGui::TextDisabled("%s", footer.c_str());
    }
    ImGui::PopTextWrapPos();
    ImGui::EndTooltip();
}

void HelpMarker(std::string const& name)
{
    editor::HelpEntry const* const help = editor::FindHelp(name);
    if (help == nullptr)
    {
        return;
    }
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    HelpTooltip(*help);
}

} // namespace ppokit::studio
