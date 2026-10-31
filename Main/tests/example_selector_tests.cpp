/**
 * @file Main/tests/example_selector_tests.cpp
 * @brief Tests et assertions déterministes pour example selector tests.
 * @details Référence de l'unité, de ses types et de ses fonctions C++.
 * Les implémentations sont accessibles via le navigateur de sources Doxygen.
 */
#include "exampleselector.hpp"

/**
 * @brief Analyse les arguments, charge l'exemple demandé et lance la simulation graphique ou headless avec supervision et checkpoints.
 */
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
