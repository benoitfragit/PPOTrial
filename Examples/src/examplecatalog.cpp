#include "examplecatalog.hpp"
#include "airspaceexample.hpp"
#include "bluedefenseexample.hpp"

namespace ppo::app {

ExampleCatalog CreateExampleCatalog()
{
    // Keep Airspace first so the existing no-argument startup is unchanged.
    return ExampleCatalog{ppo::examples::airspace::Definition(),
                          ppo::examples::bluedefense::Definition(),
                          ppo::examples::bluedefense::BluePatrolDefinition()};
}

} // namespace ppo::app
