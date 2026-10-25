#pragma once
#include "airspaceworldrules.hpp"

namespace ppo::examples::twovstwo {
class TwoVsTwoWorldRules final : public ppo::model::IWorldRules {
public:
    void BeforeStep(ppo::model::WorldEntities& entities) override;
    void Resolve(ppo::model::WorldEntities& entities) override;
    void AfterStep() override;
    void CopySnapshot(entt::registry& source, entt::registry& target) const override;
private:
    ppo::examples::airspace::AirspaceWorldRules original_{};
};
}
