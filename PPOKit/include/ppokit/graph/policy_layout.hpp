/**
 * @file PPOKit/include/ppokit/graph/policy_layout.hpp
 * @brief Where graph inputs and outputs live in the flat observation and action vectors.
 * @details Tasks exchange flat observation and action arrays (the layout of task.json).
 * A PolicyLayout tells a graph learner which slice of the observation feeds each graph
 * input, and which action components each head produces. ppokit::schema::MakePolicyLayout
 * derives it from task.json and graph.json.
 */
#pragma once

#include "ppokit/graph/graph.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ppokit::graph {

/** @brief Slice of the observation bound to one graph input. */
struct InputSlice final
{
    /** @brief First value of the slice in the observation vector. */
    std::size_t offset = 0;
    graph::Shape shape;
    /** @brief For a set: feature that holds the presence flag of each entity (> 0.5 means present). */
    std::optional<std::size_t> maskFeature;
};

/**
 * @brief An action component that is active only in some states (task.json "active"):
 *        observation[feature] > threshold, or <= threshold when `above` is false.
 */
struct ComponentActivity final
{
    /** @brief Index of the component in the flat action vector. */
    std::size_t axis = 0;
    /** @brief Index of the tested feature in the flat observation vector. */
    std::size_t feature = 0;
    double threshold = 0.0;
    bool above = true;

    /** @brief Whether the component is active for this observation. */
    bool Active(float const* observation) const noexcept
    {
        double const value = static_cast<double>(observation[feature]);
        if (above)
        {
            return value > threshold;
        }
        return value <= threshold;
    }
};

/** @brief A critic block that uses the parameters of an actor block (network.json "share"). */
struct SharedBlock final
{
    std::string critic;
    std::string actor;
};

/** @brief Inputs of both networks and the action components of the actor heads. */
struct PolicyLayout final
{
    /** @brief Actor inputs, in graph input order. */
    std::vector<InputSlice> actorInputs;
    /** @brief Critic inputs, in graph input order, read from the same observation. */
    std::vector<InputSlice> criticInputs;
    /** @brief Action index of each output of the Gaussian head, in head order. */
    std::vector<std::size_t> gaussianAxes;
    /** @brief Action index of each output of the Bernoulli head, in head order. */
    std::vector<std::size_t> bernoulliAxes;
    /** @brief Bernoulli axes whose deterministic tie is one. All others choose zero. */
    std::vector<std::size_t> oneOnTie;
    /** @brief Components active only in some states; the others are always active. */
    std::vector<ComponentActivity> activities;
    /** @brief Critic blocks trained with the actor blocks whose parameters they share. */
    std::vector<SharedBlock> sharedBlocks;
};

/**
 * @brief Turns an observation vector into the input signals of a graph.
 * @details Vector inputs point straight into the observation; set inputs also get a
 * presence mask read from their mask feature. Buffers are allocated once: Bind never
 * allocates. The observation must outlive the Forward and Backward calls that use it.
 */
class InputBinder final
{
public:
    /**
     * @param slices Graph inputs, in graph order.
     * @param observationSize Size of the observation the slices refer to.
     * @throws std::invalid_argument for a slice outside the observation or a set without mask.
     */
    InputBinder(std::vector<InputSlice> slices, std::size_t observationSize) : slices_(std::move(slices))
    {
        for (std::size_t input = 0; input < slices_.size(); ++input)
        {
            InputSlice const& slice = slices_[input];
            std::string const name = "PPOKit input binder: input " + std::to_string(input + 1);
            if (slice.offset > observationSize || slice.shape.Size() > observationSize - slice.offset)
            {
                throw std::invalid_argument(name + " ends past the observation");
            }
            bool const isSet = slice.shape.kind == SignalKind::Set;
            if (isSet && (!slice.maskFeature || *slice.maskFeature >= slice.shape.width))
            {
                throw std::invalid_argument(name + " is a set without a valid mask feature");
            }
            masks_.emplace_back(isSet ? slice.shape.capacity : 0, std::uint8_t{0});
        }
        signals_.resize(slices_.size());
    }

    std::vector<InputSlice> const& Slices() const noexcept { return slices_; }

    /** @brief Signals of one observation, valid until the next Bind. */
    std::vector<Signal> const& Bind(float const* observation) noexcept
    {
        for (std::size_t input = 0; input < slices_.size(); ++input)
        {
            InputSlice const& slice = slices_[input];
            float const* values = observation + slice.offset;
            signals_[input].values = values;
            signals_[input].mask = nullptr;
            if (slice.shape.kind == SignalKind::Set)
            {
                std::vector<std::uint8_t>& mask = masks_[input];
                for (std::size_t entity = 0; entity < slice.shape.capacity; ++entity)
                {
                    float const flag = values[entity * slice.shape.width + *slice.maskFeature];
                    mask[entity] = flag > 0.5F ? 1 : 0;
                }
                signals_[input].mask = mask.data();
            }
        }
        return signals_;
    }

private:
    std::vector<InputSlice> slices_;
    std::vector<std::vector<std::uint8_t>> masks_;
    std::vector<Signal> signals_;
};

} // namespace ppokit::graph
