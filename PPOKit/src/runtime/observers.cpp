/**
 * @file PPOKit/src/runtime/observers.cpp
 * @brief Calling the observers of a graph after each environment step.
 */
#include "ppokit/runtime/observers.hpp"

#include <algorithm>
#include <exception>

namespace ppokit::runtime {

/** @brief One input port: where it reads in a step, and its own buffer of rows. */
struct ObserverSet::Input final
{
    schema::ObservedSignal signal = schema::ObservedSignal::Observation;
    std::size_t offset = 0;
    std::size_t size = 0;
    /** @brief maximumAgents rows of `size` values. */
    std::vector<float> rows;
    /** @brief Per-agent shape handed to the DLL. */
    std::vector<std::uint32_t> shape;
};

/** @brief One observer: its instance, its inputs and the C views of their buffers. */
struct ObserverSet::Observer final
{
    std::string name;
    std::unique_ptr<component::Instance> instance;
    std::vector<Input> inputs;
    std::vector<PpokitBuffer> buffers;
    bool active = true;
    std::string error;
};

namespace {

/** @brief Seed of one observer: the run's seed mixed with the observer's name (FNV-1a). */
std::uint64_t ObserverSeed(std::uint64_t seed, std::string const& name) noexcept
{
    std::uint64_t hash = 14695981039346656037ULL ^ seed;
    for (char const character : name)
    {
        hash = (hash ^ static_cast<unsigned char>(character)) * 1099511628211ULL;
    }
    return hash;
}

} // namespace

ObserverSet::ObserverSet(schema::TaskSchema const& task, std::vector<schema::ObserverSpec> const& observers,
    std::size_t maximumAgents, std::uint64_t seed)
    : observationStride_(task.observation.size),
      worldStride_(task.world ? task.world->size : 0),
      actionStride_(task.actionSize),
      rewardStride_(task.rewards.size()),
      maximumAgents_(maximumAgents)
{
    for (schema::ObserverSpec const& spec : observers)
    {
        auto observer = std::make_unique<Observer>();
        observer->name = spec.name;
        observer->instance = std::make_unique<component::Instance>(spec.component, spec.config, spec.shapes,
            ObserverSeed(seed, spec.name));
        observer->inputs.reserve(spec.inputs.size());
        for (schema::ObservedInput const& source : spec.inputs)
        {
            Input input;
            input.signal = source.signal;
            input.offset = source.offset;
            input.size = source.size;
            input.rows.assign(maximumAgents * source.size, 0.0F);
            for (std::size_t const dimension : source.shape)
            {
                input.shape.push_back(static_cast<std::uint32_t>(dimension));
            }
            observer->inputs.push_back(std::move(input));
        }
        // The C views point into the inputs, which no longer move.
        for (Input& input : observer->inputs)
        {
            PpokitBuffer buffer{};
            buffer.data = input.rows.data();
            buffer.dtype = PPOKIT_DTYPE_F32;
            buffer.rank = static_cast<std::uint32_t>(input.shape.size());
            buffer.shape = input.shape.data();
            observer->buffers.push_back(buffer);
        }
        observers_.push_back(std::move(observer));
    }
}

ObserverSet::~ObserverSet() = default;

void ObserverSet::Gather(Input& input, ObservedStep const& step, std::size_t agents) const noexcept
{
    float const* source = nullptr;
    std::size_t stride = 0;
    switch (input.signal)
    {
    case schema::ObservedSignal::Observation:
        source = step.observation;
        stride = observationStride_;
        break;
    case schema::ObservedSignal::World:
        source = step.world;
        stride = worldStride_;
        break;
    case schema::ObservedSignal::Action:
        source = step.actions;
        stride = actionStride_;
        break;
    case schema::ObservedSignal::Reward:
        source = step.rewards;
        stride = rewardStride_;
        break;
    }
    float* const destination = input.rows.data();
    if (source == nullptr)
    {
        // A signal the step does not carry reads as zeros.
        std::fill(destination, destination + agents * input.size, 0.0F);
        return;
    }
    for (std::size_t agent = 0; agent < agents; ++agent)
    {
        float const* const row = source + agent * stride + input.offset;
        std::copy(row, row + input.size, destination + agent * input.size);
    }
}

void ObserverSet::Notify(ObservedStep const& step) noexcept
{
    std::size_t const agents = std::min(step.agents, maximumAgents_);
    if (agents == 0)
    {
        return;
    }
    for (auto const& observer : observers_)
    {
        if (!observer->active)
        {
            continue;
        }
        for (Input& input : observer->inputs)
        {
            Gather(input, step, agents);
        }
        try
        {
            observer->instance->Forward(static_cast<std::uint32_t>(agents), observer->buffers.data(), nullptr,
                nullptr);
        }
        catch (std::exception const& failure)
        {
            // A broken observer must not stop the run: it is switched off with its reason.
            observer->active = false;
            observer->error = failure.what();
        }
        catch (...)
        {
            observer->active = false;
            observer->error = "unknown exception";
        }
    }
}

std::string const& ObserverSet::Name(std::size_t index) const noexcept
{
    return observers_[index]->name;
}

bool ObserverSet::Active(std::size_t index) const noexcept
{
    return observers_[index]->active;
}

std::string const& ObserverSet::Error(std::size_t index) const noexcept
{
    return observers_[index]->error;
}

} // namespace ppokit::runtime
