#pragma once

#include <string>

#include "entity.hpp"

namespace ppo::model {

class Aircraft: public Entity {
public:
    Aircraft(entt::registry* registry, std::string const& name, eEntitySide side);

    void Run    (void) noexcept final;

private:
    void OnInit(void) noexcept final;
};

}