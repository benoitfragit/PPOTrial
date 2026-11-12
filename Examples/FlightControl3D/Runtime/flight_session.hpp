#pragma once
#include "flight_environment.hpp"
#include "flight_learner.hpp"

namespace ppo::examples::flight {

/** Value-only render snapshot; the renderer never reads FDM or network objects. */
struct FlightSnapshot final
{
    Runway runway{};
    FlightPlan plan{};
    AircraftState aircraft{};
    MissionState mission{}, lastEpisode{};
    Guidance guidance{};
    Controls controls{};
    StepResult lastStep{};
    LearningStats learning{};
    std::uint64_t episodes{}, touchdowns{}, stops{}, crashes{}, timeouts{};
    double glideDegrees{}, ilsRange{}, stepSeconds{};
    bool training{}, finished{}, failed{};
};

/** Shared orchestration for the current desktop framework and CPU headless runner. */
class FlightSession final
{
public:
    FlightSession(Config config, bool training, std::filesystem::path checkpoint);
    void Tick();
    void ResetEpisode();
    FlightSnapshot Snapshot() const noexcept;
    bool Finished() const noexcept { return finished_; }
    FlightLearner const& Learner() const noexcept { return learner_; }
private:
    Config config_;
    bool training_{};
    std::filesystem::path checkpoint_;
    FlightEnvironment environment_;
    FlightLearner learner_;
    Controls controls_{};
    StepResult lastStep_{};
    MissionState lastEpisode_{};
    std::uint64_t episodes_{}, touchdowns_{}, stops_{}, crashes_{}, timeouts_{};
    bool pendingReset_{}, finished_{};
};
} // namespace ppo::examples::flight
