#include "airspacemeta.hpp"
#include "metaregistry.hpp"
#include "environment.hpp"
#include "components.hpp"
#include <stdexcept>

int main()
{
    using namespace ppo::reflection;
    ppo::examples::airspace::RegisterAirspaceMeta();
    ppo::examples::airspace::RegisterAirspaceMeta();

    for (auto const* name : {"Aircraft", "SAM", "Objective",
                             "SNATrainingAgent", "SNAProdAgent",
                             "TargetSelectorTrainingAgent", "TargetSelectorProdAgent",
                             "SNAState", "FlightModelState", "PseudoLidarState",
                             "InterceptionStateModel", "MissionStateModel"})
    {
        if (!ResolveType(name))
            throw std::runtime_error(name);
    }

    ppo::model::Environment env;
    auto objective = CreateEntity("Objective", env, "", ppo::model::eEntitySide::Friend);
    if (objective == nullptr || !objective->has<ppo::model::sEntityState>())
        throw std::runtime_error("Airspace objective factory failed");
    if (CreateEntity("NoSuchEntity", env, "", ppo::model::eEntitySide::Friend))
        throw std::runtime_error("Unknown entity should not be created");
    return 0;
}
