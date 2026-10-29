#pragma once

namespace ppo::examples::airspace {
// Register Airspace's concrete types and factories once, on the application
// thread, before creating agents or starting evaluation workers.
void RegisterAirspaceMeta();
} // namespace ppo::examples::airspace
