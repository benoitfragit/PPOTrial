#include "examplecatalog.hpp"
#include "airspaceexample.hpp"

namespace ppo::app {

ExampleCatalog CreateExampleCatalog()
{
    // Keep Airspace first so the existing no-argument startup is unchanged.
    return ExampleCatalog{ppo::examples::airspace::Definition()};
}

} // namespace ppo::app
