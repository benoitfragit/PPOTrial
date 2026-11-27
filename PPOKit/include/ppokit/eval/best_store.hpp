/**
 * @file PPOKit/include/ppokit/eval/best_store.hpp
 * @brief The BEST policy of each evaluation suite: checkpoints/best/<suite>.json and the
 *        policy files it points to.
 * @details A campaign that passes every gate of its suite and ranks above the current BEST
 * (eval::RanksAbove on the suite's "best" criteria) becomes the BEST. Policy files are
 * immutable generations named by their content, `<suite>_<update>_<16 first hexadecimal
 * digits of their SHA-256>` with the policy file's extension (.ppokit, or the own format of
 * a team's network DLL): two trainings that reach the same update keep their own weights.
 * A generation is copied, checked against the digest of the policy evaluated, then renamed;
 * the manifest names it, keeps its full SHA-256 and its report, and is replaced atomically
 * only after that, so a reader never sees half a promotion and a failure keeps the previous
 * BEST. ReadBest checks the policy against the digest.
 *
 * @code{.json}
 * { "schema": "ppokit.best/2", "suite": "validation", "update": 250,
 *   "policy": "validation_250_3f2a9c0d1e4b5a67.ppokit", "sha256": "3f2a9c0d...",
 *   "report": { ... EvaluationReport::ToJson ... } }
 * @endcode
 *
 * Manifests "ppokit.best/1" (policy `<suite>_<update>`, no digest) are still read.
 */
#pragma once

#include "ppokit/eval/evaluation.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ppokit::eval {

/** @brief The current BEST of a suite. */
struct BestRecord final
{
    std::uint64_t update = 0;
    /** @brief Absolute path of its policy file. */
    std::filesystem::path policy;
    /** @brief SHA-256 of the policy file (empty for a former "ppokit.best/1" manifest). */
    std::string sha256;
    EvaluationReport report;
    /** @brief The environment state used for this campaign (absent in legacy records). */
    std::optional<std::string> environmentState;
    std::vector<double> rewardWeights;
};

/** @brief checkpoints/best/<suite>.json. */
std::filesystem::path BestManifest(std::filesystem::path const& checkpoints, std::string const& suite);

/**
 * @brief Enumerate the run's BEST manifests and immutable generations for archiving on a fresh start.
 * @param checkpoints Directory containing best/.
 * @param configured Current suites. Manifests also identify suites removed from eval/.
 * @param policyFile Published actor path. Its extension identifies generations of configured suites.
 * @return Flat files owned by PPOKit, once each. Native gate directories have their own lifecycle.
 */
std::vector<std::filesystem::path> BestRunFiles(std::filesystem::path const& checkpoints,
    std::vector<EvaluationSuite> const& configured, std::filesystem::path const& policyFile);

/**
 * @brief The BEST of `suite`, if it has one.
 * @throws std::runtime_error for a damaged manifest, a missing policy file, or a policy
 *         file that does not match the manifest's SHA-256.
 */
std::optional<BestRecord> ReadBest(std::filesystem::path const& checkpoints, std::string const& suite);

/**
 * @brief Makes the policy of `update` the BEST of `suite` when `report` passed every gate
 *        and ranks above the current BEST (always when there is none).
 * @param policyFile The policy to copy after verifying its bytes against the evaluated actor.
 * @param report The campaign result, including the SHA-256 captured when its actor was opened.
 * @return True when the manifest now names it.
 * @throws std::runtime_error when the copy cannot be made or checked, or when the current
 *         BEST cannot be read; the previous manifest is then unchanged.
 */
bool OfferBest(std::filesystem::path const& checkpoints, EvaluationSuite const& suite, std::uint64_t update,
    std::filesystem::path const& policyFile, EvaluationReport const& report,
    std::optional<std::string> const& environmentState = std::nullopt,
    std::vector<double> const& rewardWeights = {});

} // namespace ppokit::eval
