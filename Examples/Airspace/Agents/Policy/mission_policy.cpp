/**
 * @file Examples/Airspace/Agents/Policy/mission_policy.cpp
 * @brief Forward pass of the RLMission track-attention actor, in float like LibTorch.
 */
#include "mission_policy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace rl {
namespace {

constexpr char Architecture[] = "mission_track_attention";
/** @brief The large negative number LibTorch used for absent tracks. */
constexpr float Absent = -1e9F;
/** @brief Smallest presence sum of the mean, as LibTorch's clamp_min(1e-6). */
constexpr float MinimumPresence = 1e-6F;

} // namespace

MissionPolicy::MissionPolicy(std::filesystem::path const& path)
{
    LegacyPolicyFile const file(path);
    if (file.Architecture() != Architecture)
    {
        throw std::runtime_error(path.string() + ": architecture " + file.Architecture() + ", expected "
            + Architecture);
    }
    encoder1_ = LegacyDense(file, "enc1", TrackFeatures, EncoderWidth);
    encoder2_ = LegacyDense(file, "enc2", EncoderWidth, Width);
    query_ = LegacyDense(file, "attn_q", 2 * Width, Width);
    key_ = LegacyDense(file, "attn_k", Width, Width);
    mean_ = LegacyDense(file, "mu_head", HeadInputs, 2);
    fire_ = LegacyDense(file, "fire_head", HeadInputs, 1);
}

ppo::core::Action MissionPolicy::ChooseAction(ppo::core::Observation const& observation)
{
    if (observation.size() != ObservationSize)
    {
        throw std::invalid_argument("MissionPolicy: observation dimension mismatch");
    }
    float const* const values = observation.data();
    EncodeTracks(values);
    Pool();
    Attend();
    // Ammo then pseudo-lidar follow the tracks in the observation and the context in the heads.
    std::copy(values + Tracks * TrackFeatures, values + ObservationSize, heads_.begin() + 3 * Width);
    ppo::core::Action action(ActionSize);
    mean_.Apply(heads_.data(), action.data());
    fire_.Apply(heads_.data(), action.data() + 2);
    return action;
}

/** @brief Presence and embedding of every track slot. */
void MissionPolicy::EncodeTracks(float const* observation) noexcept
{
    for (std::size_t track = 0; track < Tracks; ++track)
    {
        float const* const features = observation + track * TrackFeatures;
        presence_[track] = std::clamp(features[TrackFeatures - 1], 0.F, 1.F);
        encoder1_.Apply(features, hidden_.data());
        ApplyRelu(hidden_.data(), EncoderWidth);
        float* const embedding = features_.data() + track * Width;
        encoder2_.Apply(hidden_.data(), embedding);
        ApplyRelu(embedding, Width);
    }
}

/** @brief Presence-weighted mean into heads_[0, 64), maximum of present tracks into heads_[64, 128). */
void MissionPolicy::Pool() noexcept
{
    float presenceSum = 0.F;
    for (std::size_t track = 0; track < Tracks; ++track)
    {
        presenceSum += presence_[track];
    }
    float const denominator = std::max(presenceSum, MinimumPresence);
    for (std::size_t d = 0; d < Width; ++d)
    {
        float sum = 0.F;
        // Absent tracks count as -1e9: with no present track the maximum is -1e9.
        float largest = Absent;
        for (std::size_t track = 0; track < Tracks; ++track)
        {
            float const value = features_[track * Width + d];
            sum += value * presence_[track];
            if (presence_[track] > 0.5F)
            {
                largest = std::max(largest, value);
            }
        }
        heads_[d] = sum / denominator;
        heads_[Width + d] = largest;
    }
}

/** @brief Attention context of the pooled query over the track keys, into heads_[128, 192). */
void MissionPolicy::Attend() noexcept
{
    query_.Apply(heads_.data(), queryVector_.data());
    float const scale = std::sqrt(static_cast<float>(Width));
    float largest = 0.F;
    for (std::size_t track = 0; track < Tracks; ++track)
    {
        key_.Apply(features_.data() + track * Width, keyVector_.data());
        float dot = 0.F;
        for (std::size_t d = 0; d < Width; ++d)
        {
            dot += keyVector_[d] * queryVector_[d];
        }
        // Absent tracks: + (1 - presence) * -1e9, in float like LibTorch (it swamps the score).
        scores_[track] = dot / scale + (1.F - presence_[track]) * Absent;
        largest = track == 0 ? scores_[track] : std::max(largest, scores_[track]);
    }
    float total = 0.F;
    for (std::size_t track = 0; track < Tracks; ++track)
    {
        scores_[track] = std::exp(scores_[track] - largest);
        total += scores_[track];
    }
    float* const context = heads_.data() + 2 * Width;
    std::fill(context, context + Width, 0.F);
    for (std::size_t track = 0; track < Tracks; ++track)
    {
        float const weight = scores_[track] / total;
        for (std::size_t d = 0; d < Width; ++d)
        {
            context[d] += weight * features_[track * Width + d];
        }
    }
}

} // namespace rl
