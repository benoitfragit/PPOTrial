/**
 * @file PPOKit/include/ppokit/runtime/publication.hpp
 * @brief Recoverable publication of a learner, its policy and their environment state.
 */
#pragma once

#include "ppokit/schema/project.hpp"

#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace ppokit::runtime {

struct PublishedGeneration final
{
    std::uint64_t update;
    std::string trainingDigest;
    std::string policyDigest;
    std::string state;
    /** @brief Empty only for generation/1, whose weights were recorded in the journal. */
    std::optional<std::vector<double>> rewardWeights;
};

/** @brief Requires ownership of the training outputs. A prepared generation is replayable after process exit. */
void PublishGeneration(schema::Project const& project, std::uint64_t update, std::string const& state,
    std::vector<double> const& weights, std::function<void(std::filesystem::path const&, std::filesystem::path const&)> const& save);

/** @brief Finish a prepared generation before fresh training or resume, holding the writer lock. */
void RecoverPublication(schema::Project const& project);

/**
 * @brief Read the committed manifest for display, without opening the learner or actor.
 * @return Empty when the project has no published manifest.
 * @throws std::runtime_error if publication is in progress or the output paths differ.
 */
std::optional<PublishedGeneration> ReadPublicationManifest(schema::Project const& project);

/** @brief Verify the manifest against both artifacts and the environment sidecar before using a generation. */
std::optional<PublishedGeneration> VerifyPublication(schema::Project const& project);

} // namespace ppokit::runtime
