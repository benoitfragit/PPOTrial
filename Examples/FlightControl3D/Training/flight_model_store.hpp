#pragma once
#include "flight_learner.hpp"
#include <optional>

namespace ppo::examples::flight {

struct ValidationMetrics final
{
    std::uint64_t update{};
    std::size_t episodes{},touchdowns{},stops{},crashes{},timeouts{};
    double touchdownSumMeters{},stopSumMeters{};
    double SuccessRate() const noexcept { return episodes ? static_cast<double>(stops)/static_cast<double>(episodes) : 0; }
    double Score() const noexcept { return 100*SuccessRate(); }
    double MeanStop() const noexcept { return stops ? stopSumMeters/static_cast<double>(stops) : 0; }
    void Record(MissionState const& mission);
};

bool Validated(Config const& config,ValidationMetrics const& metrics) noexcept;
bool Better(ValidationMetrics const& candidate,ValidationMetrics const& incumbent) noexcept;

/** Mutable training state and immutable evaluated BEST generations are separate. */
class FlightModelStore final
{
public:
    FlightModelStore(Config const& config,std::filesystem::path directory);
    std::filesystem::path Latest() const;
    std::optional<ValidationMetrics> BestMetrics(bool allowChangedStarts = false) const;
    void LoadBest(FlightLearner& learner) const;
    bool Promote(FlightLearner const& learner,ValidationMetrics const& metrics);
private:
    Config config_;
    std::filesystem::path directory_;
};
} // namespace ppo::examples::flight
