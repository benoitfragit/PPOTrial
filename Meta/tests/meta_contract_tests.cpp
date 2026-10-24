#include "metaregistry.hpp"
#include "environment.hpp"
#include <stdexcept>

int main()
{
    using namespace ppo::reflection;
    RegisterAllMeta();
    RegisterAllMeta();

    if (!IsMetaRegistered() || !ResolveType("BaseEntity") ||
        !ResolveType("Entity") || !ResolveType("IAgent") ||
        !ResolveType("EntityState"))
        throw std::runtime_error("Missing shared reflected type");

    if (ResolveType("Aircraft") || ResolveType("SNATrainingAgent"))
        throw std::runtime_error("Airspace types leaked into MetaReflection");

    ppo::model::Environment world;
    if (CreateEntity("Aircraft", world, "", ppo::model::eEntitySide::Unknown))
        throw std::runtime_error("Unknown entity unexpectedly registered");
    return 0;
}
