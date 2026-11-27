/**
 * @file Examples/Airspace/Agents/Policy/mission_policy.hpp
 * @brief The red RLMission production policy (track attention), computed without LibTorch.
 */
#pragma once

#include "legacy_policy_file.hpp"
#include "missionlayout.hpp"

#include "ppo/core/environment.hpp"

#include <array>
#include <filesystem>

namespace rl {

/**
 * @brief Deterministic actor of the RLMission track-attention network.
 * @details Same computation as the LibTorch ActorCriticTrackAttention::forward_mu it replaces,
 * including its edge cases:
 * - every track goes through the shared encoder relu(enc2(relu(enc1(track)))), present or not;
 * - presence is the last track feature clamped to [0, 1]; the mean pools with it as a weight,
 *   the maximum keeps tracks above 0.5 and is -1e9 where none is;
 * - attention keys are attn_k(feature), the query attn_q([mean, max]); absent tracks get
 *   -1e9 added to their score (with no track at all the weights become uniform);
 * - the heads read [mean, max, attention context, ammo, pseudo-lidar].
 * The action is [mu0, mu1, fire logit], as LibTorch returned it.
 */
class MissionPolicy final : public ppo::core::IPolicy
{
public:
    static constexpr std::size_t Tracks = MissionObservationLayout::kMaxTracks;
    static constexpr std::size_t TrackFeatures = MissionObservationLayout::kTrackFeatureCount;
    static constexpr std::size_t GlobalFeatures = MissionObservationLayout::kGlobalFeatureCount;
    static constexpr std::size_t ObservationSize = MissionObservationLayout::kStateDimension;
    static constexpr std::size_t ActionSize = MissionObservationLayout::kActionCount;
    /** @brief Width of a track embedding. */
    static constexpr std::size_t Width = 64;
    /** @brief Hidden width of the track encoder. */
    static constexpr std::size_t EncoderWidth = 128;
    /** @brief Input of the heads: mean, max, context and the global features. */
    static constexpr std::size_t HeadInputs = 3 * Width + GlobalFeatures;

    /**
     * @brief Loads a "mission_track_attention" legacy policy file.
     * @throw std::runtime_error when it cannot be read or belongs to another architecture.
     */
    explicit MissionPolicy(std::filesystem::path const& path);

    ppo::core::Action ChooseAction(ppo::core::Observation const& observation) override;

private:
    void EncodeTracks(float const* observation) noexcept;
    void Pool() noexcept;
    void Attend() noexcept;

    LegacyDense encoder1_;
    LegacyDense encoder2_;
    LegacyDense query_;
    LegacyDense key_;
    LegacyDense mean_;
    LegacyDense fire_;

    std::array<float, Tracks> presence_{};
    std::array<float, Tracks * Width> features_{};
    std::array<float, EncoderWidth> hidden_{};
    /** @brief mean | max | context | global features, the input of the heads. */
    std::array<float, HeadInputs> heads_{};
    std::array<float, Width> queryVector_{};
    std::array<float, Width> keyVector_{};
    std::array<float, Tracks> scores_{};
};

} // namespace rl
