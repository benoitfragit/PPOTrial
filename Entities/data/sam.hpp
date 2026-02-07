#pragma once

#include <string>

#include "entity.hpp"

namespace ppo::model {

class SAM: public Entity {
public:
    SAM(entt::registry* registry, std::string const& name, eEntitySide side);

    void Run    (void) noexcept final;

private:
    void OnInit(void) noexcept final;
};

}