/**
 * @file PPOKit/include/ppokit/runtime/probe_record.hpp
 * @brief What the probes measured on the blocks of a network during one update, as the
 *        run journal stores it (`probes` events).
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ppokit::runtime {

/** @brief Measures of one block during one update. */
struct BlockProbe final
{
    /** @brief Block name in graph.json, and its type (dense, masked_attention...). */
    std::string block;
    std::string type;
    /** @brief Output values observed (entities x width for sets, present entities only). */
    std::size_t samples = 0;
    double mean = 0.0;
    double deviation = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    /** @brief Share of outputs exactly zero: dead ReLU units. */
    double deadShare = 0.0;
    /** @brief Share of outputs at |x| >= 0.99: saturated tanh or sigmoid units. */
    double saturatedShare = 0.0;
    /** @brief The block has parameters, so a gradient norm is measured. */
    bool hasGradient = false;
    /** @brief Mean L2 norm of the gradient of its parameters per optimizer step. */
    double gradientNorm = 0.0;
};

/** @brief One `probes` event: a network (actor, critic) during update `update`. */
struct ProbeRecord final
{
    std::uint64_t update = 0;
    std::string network;
    std::vector<BlockProbe> blocks;
};

} // namespace ppokit::runtime
