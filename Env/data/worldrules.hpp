#pragma once

#include <entt/entt.hpp>
#include <memory>
#include <vector>

namespace ppo::model {

class Entity;
using WorldEntities = std::vector<std::shared_ptr<Entity>>;

// Scenario extension point; no policy, learning, reward or physics assumptions.
// Called only on the simulation thread (except initial registration).
class IWorldRules
{
public:
    virtual ~IWorldRules() = default;

    virtual void BeforeStep(WorldEntities& entities) = 0;
    virtual void Resolve(WorldEntities& entities) = 0;
    virtual void AfterStep() = 0;
    virtual void CopySnapshot(entt::registry& source, entt::registry& destination) const = 0;
};

} // namespace ppo::model
