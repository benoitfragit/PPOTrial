#pragma once

#include "generalized_observation.hpp"

// Establish CPU and model operations before defining RLtools adapters.
#include "patrol_rltools.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace ppo::examples::bluedefense::defenders {

namespace rlt = rl_tools;

/**
 * @brief Masked order-invariant attention over a fixed-capacity raw track set.
 * @tparam T Scalar type.
 * @tparam TI Index type.
 * @tparam F Features per raw track; the final feature is the mask.
 * @tparam N Raw track capacity.
 * @tparam D Learned embedding width.
 */
template<typename T, typename TI, TI F, TI N, TI D>
struct GeneralizedTrackAttention final
{
    static constexpr TI Features = F;
    static constexpr TI Tracks = N;
    static constexpr TI Width = D;
    static constexpr TI OutputWidth = 3 * D;

    template<TI Rows, TI Cols, typename Kind>
    using Parameter = typename rlt::nn::parameters::Adam::template instance<
        typename rlt::nn::parameters::Plain::template spec<
            rlt::MatrixDynamic<rlt::matrix::Specification<
                T, TI, Rows, Cols>>,
            rlt::nn::parameters::groups::Normal, Kind>>;
    using Weights = rlt::nn::parameters::categories::Weights;
    using Biases = rlt::nn::parameters::categories::Biases;

    Parameter<D, F, Weights> encoderW{};
    Parameter<1, D, Biases> encoderB{};
    Parameter<D, ObservationLayout::Own, Weights> queryW{};
    Parameter<1, D, Biases> queryB{};

    using Vector = std::array<T, D>;
    struct Cache final
    {
        std::array<Vector, N> features{};
        std::array<T, N> probabilities{};
        std::array<bool, N> valid{};
        Vector query{};
        Vector context{};
        Vector mean{};
        Vector max{};
        TI count{};
    };

    /**
     * @brief Pools a masked raw set into attention, mean and max summaries.
     * @param source Raw observation matrix.
     * @param row Matrix row.
     * @param sourceOffset First raw track feature.
     * @param destination Compact actor-input matrix.
     * @param destinationOffset First compact pooling feature.
     * @return Forward cache used by the attention backward pass.
     */
    template<typename Input, typename Output>
    Cache Transform(Input const& source, TI const row,
                    TI const sourceOffset, Output& destination,
                    TI const destinationOffset) const
    {
        Cache cache{};
        T const scale = T{1} / std::sqrt(static_cast<T>(D));
        cache.max.fill(-std::numeric_limits<T>::infinity());
        for (TI dimension = 0; dimension < D; ++dimension)
        {
            T value = rlt::get(queryB.parameters, 0, dimension);
            for (TI feature = 0; feature < ObservationLayout::Own; ++feature)
            {
                value += rlt::get(queryW.parameters, dimension, feature)
                    * rlt::get(source, row, feature);
            }
            cache.query[dimension] = std::tanh(value);
        }

        std::array<T, N> scores{};
        T largest = -std::numeric_limits<T>::infinity();
        for (TI track = 0; track < N; ++track)
        {
            TI const base = sourceOffset + track * F;
            cache.valid[track] =
                rlt::get(source, row, base + F - 1) > T{0.5};
            if (!cache.valid[track])
                continue;
            ++cache.count;
            T score{};
            for (TI dimension = 0; dimension < D; ++dimension)
            {
                T value = rlt::get(encoderB.parameters, 0, dimension);
                for (TI feature = 0; feature < F; ++feature)
                {
                    value += rlt::get(encoderW.parameters, dimension, feature)
                        * rlt::get(source, row, base + feature);
                }
                T const encoded = std::tanh(value);
                cache.features[track][dimension] = encoded;
                cache.mean[dimension] += encoded;
                cache.max[dimension] =
                    std::max(cache.max[dimension], encoded);
                score += cache.query[dimension] * encoded * scale;
            }
            scores[track] = score;
            largest = std::max(largest, score);
        }

        if (cache.count != 0)
        {
            T denominator{};
            for (TI track = 0; track < N; ++track)
            {
                if (!cache.valid[track])
                    continue;
                cache.probabilities[track] =
                    std::exp(scores[track] - largest);
                denominator += cache.probabilities[track];
            }
            for (TI track = 0; track < N; ++track)
            {
                if (!cache.valid[track])
                    continue;
                cache.probabilities[track] /= denominator;
                for (TI dimension = 0; dimension < D; ++dimension)
                {
                    cache.context[dimension] +=
                        cache.probabilities[track]
                        * cache.features[track][dimension];
                }
            }
            for (TI dimension = 0; dimension < D; ++dimension)
                cache.mean[dimension] /= static_cast<T>(cache.count);
        }

        for (TI dimension = 0; dimension < D; ++dimension)
        {
            rlt::set(destination, row, destinationOffset + dimension,
                     cache.context[dimension]);
            rlt::set(destination, row, destinationOffset + D + dimension,
                     cache.mean[dimension]);
            rlt::set(destination, row, destinationOffset + 2 * D + dimension,
                     cache.count != 0 ? cache.max[dimension] : T{});
        }
        return cache;
    }

    /**
     * @brief Accumulates attention parameter gradients from a compact input.
     * @param input Raw observation matrix.
     * @param row Matrix row.
     * @param sourceOffset First raw track feature.
     * @param derivative Derivative with respect to compact actor input.
     * @param destinationOffset First compact pooling feature.
     * @param cache Matching forward cache.
     */
    template<typename Input, typename Derivative>
    void Differentiate(Input const& input, TI const row,
                       TI const sourceOffset, Derivative const& derivative,
                       TI const destinationOffset, Cache const& cache)
    {
        if (cache.count == 0)
            return;
        Vector queryDerivative{};
        T const scale = T{1} / std::sqrt(static_cast<T>(D));
        for (TI track = 0; track < N; ++track)
        {
            if (!cache.valid[track])
                continue;
            T dot{};
            for (TI dimension = 0; dimension < D; ++dimension)
            {
                dot += rlt::get(derivative, row,
                                destinationOffset + dimension)
                    * (cache.features[track][dimension]
                       - cache.context[dimension]);
            }
            T const scoreDerivative =
                cache.probabilities[track] * dot * scale;
            for (TI dimension = 0; dimension < D; ++dimension)
            {
                TI ties{};
                for (TI candidate = 0; candidate < N; ++candidate)
                {
                    if (cache.valid[candidate]
                        && cache.features[candidate][dimension]
                               == cache.max[dimension])
                    {
                        ++ties;
                    }
                }
                T const gradient =
                    cache.probabilities[track]
                        * rlt::get(derivative, row,
                                   destinationOffset + dimension)
                    + scoreDerivative * cache.query[dimension]
                    + rlt::get(derivative, row,
                               destinationOffset + D + dimension)
                        / static_cast<T>(cache.count)
                    + (cache.features[track][dimension]
                               == cache.max[dimension]
                           ? rlt::get(derivative, row,
                                      destinationOffset + 2 * D + dimension)
                                 / static_cast<T>(ties)
                           : T{});
                T const encoded = cache.features[track][dimension];
                T const activationDerivative =
                    gradient * (T{1} - encoded * encoded);
                rlt::increment(encoderB.gradient, 0, dimension,
                               activationDerivative);
                TI const base = sourceOffset + track * F;
                for (TI feature = 0; feature < F; ++feature)
                {
                    rlt::increment(
                        encoderW.gradient, dimension, feature,
                        activationDerivative
                            * rlt::get(input, row, base + feature));
                }
                queryDerivative[dimension] +=
                    scoreDerivative * encoded;
            }
        }

        for (TI dimension = 0; dimension < D; ++dimension)
        {
            T const query = cache.query[dimension];
            T const activationDerivative =
                queryDerivative[dimension] * (T{1} - query * query);
            rlt::increment(queryB.gradient, 0, dimension,
                           activationDerivative);
            for (TI feature = 0; feature < ObservationLayout::Own; ++feature)
            {
                rlt::increment(queryW.gradient, dimension, feature,
                               activationDerivative
                                   * rlt::get(input, row, feature));
            }
        }
    }
};

/**
 * @brief Preallocated scratch storage for generalized actor evaluation.
 * @tparam Specification Compact MLP specification.
 * @tparam Batch Maximum row count handled by this buffer.
 * @tparam ContainerTag RLTools dynamic or static matrix container tag.
 */
template<typename Specification, typename Specification::TI Batch,
         typename ContainerTag = rlt::MatrixDynamicTag>
struct GeneralizedAttentionActorBuffer
    : rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<
          Specification>::template Buffer<Batch, ContainerTag>
{
    using Base = typename rlt::nn_models::mlp_unconditional_stddev::
        NeuralNetworkAdam<Specification>::template Buffer<Batch, ContainerTag>;
    using CompactSpecification = rlt::matrix::Specification<
        typename Specification::T, typename Specification::TI, Batch,
        ObservationLayout::Size>;
    using CompactMatrix =
        typename ContainerTag::template type<CompactSpecification>;

    /** @brief Reusable raw-to-MLP compact representation. */
    CompactMatrix compact{};
    /** @brief Reusable derivative with respect to the compact representation. */
    CompactMatrix compactDerivative{};
};

/**
 * @brief P/N-capable shared actor with raw-set pooling and a compact v7 trunk.
 *
 * The inherited MLP consumes exactly the historical 156 compact inputs. Raw
 * track capacities therefore do not enlarge its dense layers. The actor adds
 * P/N and formation-slot signals in compact columns that were always zero in
 * v7, enabling an output-preserving migration.
 */
template<typename Specification>
struct GeneralizedAttentionActor
    : rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<
          Specification>
{
    using Base =
        rlt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<
            Specification>;
    using T = typename Specification::T;
    using TI = typename Specification::TI;
    static_assert(Specification::INPUT_DIM == ObservationLayout::Size,
                  "The generalized actor trunk must retain the v7 width");

    template<TI Batch, typename Tag = rlt::MatrixDynamicTag>
    using Buffer = GeneralizedAttentionActorBuffer<Specification, Batch, Tag>;

    using ForwardCompactSpecification = rlt::matrix::Specification<
        T, TI, Specification::BATCH_SIZE, ObservationLayout::Size>;
    using ForwardCompactMatrix = typename Specification::CONTAINER_TYPE_TAG::
        template type<ForwardCompactSpecification>;

    /**
     * @brief Preallocated PPO-forward scratch matrix.
     *
     * PPO's RLTools forward API has no external buffer argument. Keeping this
     * matrix with the trainable actor avoids heap activity in every update.
     */
    ForwardCompactMatrix forwardCompact{};

    GeneralizedTrackAttention<T, TI,
                              GeneralizedObservationLayout::RedFeatures,
                              GeneralizedObservationLayout::RedTracks, 8> red;
    GeneralizedTrackAttention<T, TI,
                              GeneralizedObservationLayout::FriendlyFeatures,
                              GeneralizedObservationLayout::BlueTracks
                                  + GeneralizedObservationLayout::ObjectiveTracks,
                              6> friendlies;
};

/** @brief Compact columns introduced by the generalized raw-to-MLP adapter. */
struct GeneralizedCompactFeatures final
{
    static constexpr std::size_t FormationSlot =
        ObservationLayout::GlobalOffset + 1U;
    static constexpr std::size_t ActiveBlueCount =
        ObservationLayout::RedOffset + 3U * 8U;
    static constexpr std::size_t ActiveRedCount =
        ActiveBlueCount + 1U;
};

} // namespace ppo::examples::bluedefense::defenders

namespace rl_tools {
namespace blue_generalized_attention_detail {

template<typename Device, typename Block>
void malloc(Device& device, Block& block)
{
    rl_tools::malloc(device, block.encoderW);
    rl_tools::malloc(device, block.encoderB);
    rl_tools::malloc(device, block.queryW);
    rl_tools::malloc(device, block.queryB);
}

template<typename Device, typename Block>
void free(Device& device, Block& block)
{
    rl_tools::free(device, block.encoderW);
    rl_tools::free(device, block.encoderB);
    rl_tools::free(device, block.queryW);
    rl_tools::free(device, block.queryB);
}

template<typename Device, typename Block>
void zero_gradient(Device& device, Block& block)
{
    rl_tools::set_all(device, block.encoderW.gradient, 0);
    rl_tools::set_all(device, block.encoderB.gradient, 0);
    rl_tools::set_all(device, block.queryW.gradient, 0);
    rl_tools::set_all(device, block.queryB.gradient, 0);
}

template<typename Device, typename Block, typename Optimizer>
void reset_optimizer(Device& device, Block& block, Optimizer& optimizer)
{
    rl_tools::_reset_optimizer_state(device, block.encoderW, optimizer);
    rl_tools::_reset_optimizer_state(device, block.encoderB, optimizer);
    rl_tools::_reset_optimizer_state(device, block.queryW, optimizer);
    rl_tools::_reset_optimizer_state(device, block.queryB, optimizer);
}

template<typename Device, typename Block, typename Optimizer>
void update(Device& device, Block& block, Optimizer& optimizer)
{
    rl_tools::update(device, block.encoderW, optimizer);
    rl_tools::update(device, block.encoderB, optimizer);
    rl_tools::update(device, block.queryW, optimizer);
    rl_tools::update(device, block.queryB, optimizer);
}

template<typename Device, typename Block, typename RandomEngine>
void initialize(Device&, Block& block, RandomEngine& randomEngine)
{
    auto const fill = [&](auto& matrix, bool const weight) {
        using Matrix = std::remove_reference_t<decltype(matrix)>;
        for (typename Device::index_t row = 0; row < Matrix::SPEC::ROWS; ++row)
        {
            for (typename Device::index_t column = 0;
                 column < Matrix::SPEC::COLS; ++column)
            {
                rl_tools::set(
                    matrix, row, column,
                    weight
                        ? rl_tools::random::uniform_real_distribution(
                              typename Device::SPEC::RANDOM{}, -0.05F, 0.05F,
                              randomEngine)
                        : 0.F);
            }
        }
    };
    fill(block.encoderW.parameters, true);
    fill(block.encoderB.parameters, false);
    fill(block.queryW.parameters, true);
    fill(block.queryB.parameters, false);
}

template<typename Device, typename Actor, typename Input, typename Output>
void transform(Device& device, Actor const& actor, Input const& input,
               Output& compact)
{
    namespace blue = ppo::examples::bluedefense::defenders;
    using Raw = blue::GeneralizedObservationLayout;
    using Legacy = blue::ObservationLayout;
    rl_tools::set_all(device, compact, 0.F);
    for (typename Actor::TI row = 0; row < Input::SPEC::ROWS; ++row)
    {
        for (typename Actor::TI feature = 0;
             feature < Legacy::Own; ++feature)
        {
            rl_tools::set(compact, row, feature,
                          rl_tools::get(input, row, Raw::OwnOffset + feature));
        }
        for (typename Actor::TI ray = 0; ray < Legacy::Lidar; ++ray)
        {
            rl_tools::set(
                compact, row, Legacy::Own + ray,
                rl_tools::get(input, row, Raw::LidarOffset + ray));
        }
        actor.red.Transform(input, row, Raw::RedOffset, compact,
                            Legacy::RedOffset);
        actor.friendlies.Transform(input, row, Raw::BlueOffset, compact,
                                   Legacy::FriendOffset);
        for (typename Actor::TI feature = 0;
             feature < Legacy::Patrol; ++feature)
        {
            rl_tools::set(
                compact, row, Legacy::PatrolOffset + feature,
                rl_tools::get(input, row, Raw::PatrolOffset + feature));
        }
        rl_tools::set(compact, row, Legacy::PatrolOffset + 8U,
                      rl_tools::get(input, row, Raw::RoleOffset));
        rl_tools::set(compact, row, Legacy::GlobalOffset,
                      rl_tools::get(input, row, Raw::GlobalOffset));
        rl_tools::set(
            compact, row, blue::GeneralizedCompactFeatures::FormationSlot,
            rl_tools::get(input, row, Raw::RoleOffset + 1U));
        rl_tools::set(
            compact, row, blue::GeneralizedCompactFeatures::ActiveBlueCount,
            rl_tools::get(input, row, Raw::TeamCountsOffset));
        rl_tools::set(
            compact, row, blue::GeneralizedCompactFeatures::ActiveRedCount,
            rl_tools::get(input, row, Raw::TeamCountsOffset + 1U));
    }
}

} // namespace blue_generalized_attention_detail

template<typename Device, typename Specification>
void malloc(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor)
{
    using Actor = ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>;
    rl_tools::malloc(device, static_cast<typename Actor::Base&>(actor));
    rl_tools::malloc(device, actor.forwardCompact);
    blue_generalized_attention_detail::malloc(device, actor.red);
    blue_generalized_attention_detail::malloc(device, actor.friendlies);
}

template<typename Device, typename Specification>
void free(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor)
{
    using Actor = ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>;
    blue_generalized_attention_detail::free(device, actor.red);
    blue_generalized_attention_detail::free(device, actor.friendlies);
    rl_tools::free(device, actor.forwardCompact);
    rl_tools::free(device, static_cast<typename Actor::Base&>(actor));
}

template<typename Device, typename Specification,
         typename Specification::TI Batch, typename ContainerTag>
void malloc(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActorBuffer<
        Specification, Batch, ContainerTag>& buffer)
{
    using Buffer = ppo::examples::bluedefense::defenders::
        GeneralizedAttentionActorBuffer<Specification, Batch, ContainerTag>;
    rl_tools::malloc(device, static_cast<typename Buffer::Base&>(buffer));
    rl_tools::malloc(device, buffer.compact);
    rl_tools::malloc(device, buffer.compactDerivative);
}

template<typename Device, typename Specification,
         typename Specification::TI Batch, typename ContainerTag>
void free(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActorBuffer<
        Specification, Batch, ContainerTag>& buffer)
{
    using Buffer = ppo::examples::bluedefense::defenders::
        GeneralizedAttentionActorBuffer<Specification, Batch, ContainerTag>;
    rl_tools::free(device, buffer.compactDerivative);
    rl_tools::free(device, buffer.compact);
    rl_tools::free(device, static_cast<typename Buffer::Base&>(buffer));
}

template<typename Device, typename Specification, typename RandomEngine>
void init_weights(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor,
    RandomEngine& randomEngine)
{
    using Actor = ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>;
    rl_tools::init_weights(device, static_cast<typename Actor::Base&>(actor),
                           randomEngine);
    blue_generalized_attention_detail::initialize(device, actor.red,
                                                   randomEngine);
    blue_generalized_attention_detail::initialize(device, actor.friendlies,
                                                   randomEngine);
}

template<typename Device, typename Specification>
void zero_gradient(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor)
{
    using Actor = ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>;
    rl_tools::zero_gradient(device, static_cast<typename Actor::Base&>(actor));
    blue_generalized_attention_detail::zero_gradient(device, actor.red);
    blue_generalized_attention_detail::zero_gradient(device,
                                                      actor.friendlies);
}

template<typename Device, typename Specification, typename InputSpecification,
         typename OutputSpecification, typename Buffer>
void evaluate(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification> const& actor,
    Matrix<InputSpecification> const& input,
    Matrix<OutputSpecification>& output, Buffer& buffer)
{
    static_assert(InputSpecification::COLS
                  == ppo::examples::bluedefense::defenders::
                      GeneralizedObservationLayout::Size);
    using BufferType = std::remove_reference_t<Buffer>;
    static_assert(BufferType::CompactSpecification::ROWS
                  >= InputSpecification::ROWS);
    auto compact = rl_tools::view(
        device, buffer.compact,
        matrix::ViewSpec<
            InputSpecification::ROWS,
            ppo::examples::bluedefense::defenders::ObservationLayout::Size>{});
    blue_generalized_attention_detail::transform(device, actor, input,
                                                  compact);
    using Mlp = nn_models::mlp::NeuralNetwork<Specification>;
    auto& baseBuffer = static_cast<typename BufferType::Base&>(buffer);
    rl_tools::evaluate(device, static_cast<Mlp const&>(actor), compact, output,
                       baseBuffer);
}

template<typename Device, typename Specification, typename InputSpecification,
         typename OutputSpecification>
void forward(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor,
    Matrix<InputSpecification> const& input,
    Matrix<OutputSpecification>& output)
{
    static_assert(InputSpecification::COLS
                  == ppo::examples::bluedefense::defenders::
                      GeneralizedObservationLayout::Size);
    static_assert(Specification::BATCH_SIZE >= InputSpecification::ROWS);
    auto compact = rl_tools::view(
        device, actor.forwardCompact,
        matrix::ViewSpec<
            InputSpecification::ROWS,
            ppo::examples::bluedefense::defenders::ObservationLayout::Size>{});
    blue_generalized_attention_detail::transform(device, actor, input,
                                                  compact);
    using Mlp = nn_models::mlp::NeuralNetworkBackwardGradient<Specification>;
    rl_tools::forward(device, static_cast<Mlp&>(actor), compact, output);
}

template<typename Device, typename Specification, typename InputSpecification,
         typename OutputSpecification, typename Buffer>
void backward(
    Device& device,
    ppo::examples::bluedefense::defenders::GeneralizedAttentionActor<
        Specification>& actor,
    Matrix<InputSpecification> const& input,
    Matrix<OutputSpecification>& outputDerivative, Buffer& buffer)
{
    namespace blue = ppo::examples::bluedefense::defenders;
    using Index = typename Specification::TI;
    using BufferType = std::remove_reference_t<Buffer>;
    static_assert(BufferType::CompactSpecification::ROWS
                  >= InputSpecification::ROWS);
    auto compact = rl_tools::view(
        device, buffer.compact,
        matrix::ViewSpec<InputSpecification::ROWS,
                         blue::ObservationLayout::Size>{});
    auto compactDerivative = rl_tools::view(
        device, buffer.compactDerivative,
        matrix::ViewSpec<InputSpecification::ROWS,
                         blue::ObservationLayout::Size>{});
    blue_generalized_attention_detail::transform(device, actor, input,
                                                  compact);
    using Mlp = nn_models::mlp::NeuralNetworkBackwardGradient<Specification>;
    auto& baseBuffer = static_cast<typename BufferType::Base&>(buffer);
    rl_tools::backward_full(device, static_cast<Mlp&>(actor), compact,
                            outputDerivative, compactDerivative, baseBuffer);
    for (Index row = 0; row < InputSpecification::ROWS; ++row)
    {
        auto const redCache = actor.red.Transform(
            input, row, blue::GeneralizedObservationLayout::RedOffset, compact,
            blue::ObservationLayout::RedOffset);
        actor.red.Differentiate(
            input, row, blue::GeneralizedObservationLayout::RedOffset,
            compactDerivative, blue::ObservationLayout::RedOffset, redCache);
        auto const friendlyCache = actor.friendlies.Transform(
            input, row, blue::GeneralizedObservationLayout::BlueOffset,
            compact, blue::ObservationLayout::FriendOffset);
        actor.friendlies.Differentiate(
            input, row, blue::GeneralizedObservationLayout::BlueOffset,
            compactDerivative, blue::ObservationLayout::FriendOffset,
            friendlyCache);
    }
}

} // namespace rl_tools

namespace ppo::examples::bluedefense::defenders {

template<typename Device, typename Specification, typename Optimizer>
void update(Device& device, GeneralizedAttentionActor<Specification>& actor,
            Optimizer& optimizer)
{
    rl_tools::update(device,
                     static_cast<typename GeneralizedAttentionActor<
                         Specification>::Base&>(actor),
                     optimizer);
    rl_tools::blue_generalized_attention_detail::update(device, actor.red,
                                                        optimizer);
    rl_tools::blue_generalized_attention_detail::update(device,
                                                        actor.friendlies,
                                                        optimizer);
}

template<typename Device, typename Specification, typename Optimizer>
void _reset_optimizer_state(
    Device& device, GeneralizedAttentionActor<Specification>& actor,
    Optimizer& optimizer)
{
    rl_tools::_reset_optimizer_state(
        device,
        static_cast<typename GeneralizedAttentionActor<Specification>::Base&>(
            actor),
        optimizer);
    rl_tools::blue_generalized_attention_detail::reset_optimizer(
        device, actor.red, optimizer);
    rl_tools::blue_generalized_attention_detail::reset_optimizer(
        device, actor.friendlies, optimizer);
}

} // namespace ppo::examples::bluedefense::defenders
