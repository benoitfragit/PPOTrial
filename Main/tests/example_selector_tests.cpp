#include "exampleselector.hpp"

int main()
{
    using ppo::app::ShouldShowExampleSelector;
    static_assert(ShouldShowExampleSelector(false, false),
                  "Interactive launch without an explicit example must show the selector");
    static_assert(!ShouldShowExampleSelector(true, false),
                  "--example must bypass the GUI selector");
    static_assert(!ShouldShowExampleSelector(false, true),
                  "Headless --check-example must not create a window");
    static_assert(!ShouldShowExampleSelector(true, true),
                  "Explicit headless example must not create a window");
    return 0;
}
