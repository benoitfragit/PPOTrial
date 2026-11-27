/**
 * @file PPOKit/tests/best_store_tests.cpp
 * @brief The BEST store keeps the weights that were evaluated: generations named and checked
 *        by their SHA-256, the manifest written last, the previous BEST kept on any failure.
 * @details Also the SHA-256 itself, on the FIPS 180-4 test vectors.
 */
#include "ppokit/artifacts/sha256.hpp"
#include "ppokit/eval/best_store.hpp"
#include "ppokit/schema/task_schema.hpp"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>

namespace {

namespace fs = std::filesystem;
using ppokit::eval::EvaluationReport;
using ppokit::eval::EvaluationSuite;

int failures = 0;

void Check(bool condition, std::string const& description)
{
    if (!condition)
    {
        std::cerr << "FAILED: " << description << '\n';
        ++failures;
    }
}

void Write(fs::path const& path, std::string const& text)
{
    std::ofstream(path, std::ios::binary) << text;
}

std::string Read(fs::path const& path)
{
    std::ifstream stream(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

/** @brief True when `call` throws a std::exception whose message contains `expected`. */
template<typename Call>
bool Throws(Call&& call, std::string const& expected)
{
    try
    {
        call();
    }
    catch (std::exception const& error)
    {
        bool const matches = std::string(error.what()).find(expected) != std::string::npos;
        if (!matches)
        {
            std::cerr << "  message was: " << error.what() << '\n';
        }
        return matches;
    }
    return false;
}

/** @brief FIPS 180-4 examples, a message across the padding boundary, a file read in chunks. */
void Sha256Vectors(fs::path const& root)
{
    using ppokit::artifacts::Sha256Hex;
    Check(Sha256Hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "SHA-256 of nothing");
    Check(Sha256Hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "SHA-256 of abc");
    Check(Sha256Hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")
            == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        "SHA-256 of the two-block example (56 bytes: the length goes to a second block)");
    std::string const million(1000000, 'a');
    Check(Sha256Hex(million) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        "SHA-256 of a million 'a'");
    Write(root / "million.txt", million);
    Check(ppokit::artifacts::Sha256OfFile(root / "million.txt") == Sha256Hex(million),
        "a file is hashed like its bytes, across read chunks");
    Check(Throws([&] { ppokit::artifacts::Sha256OfFile(root / "absent.txt"); }, "cannot open"),
        "hashing a missing file is an error");
}

EvaluationSuite Suite()
{
    std::string const task = R"({ "schema": "ppokit.task/1", "name": "landing",
      "observation": { "state": { "fields": ["x"] } },
      "actions": { "move": { "head": "gaussian", "names": ["dx"] } },
      "rewards": { "R": {} },
      "metrics": { "stopped": { "kind": "flag" } } })";
    auto const metrics = ppokit::schema::ParseTaskSchema(task, "task.json").metrics;
    return ppokit::eval::ParseEvaluationSuite(R"({ "schema": "ppokit.eval/1", "suite": "validation",
      "seeds": { "first": 1, "count": 4 }, "episodes": 1,
      "gates": { "stopped.rate": ">= 0.25" },
      "best": [ { "metric": "stopped.rate", "prefer": "higher" } ] })",
        "validation.json", metrics);
}

/** @brief A passed campaign that stopped `stops` of its 4 episodes. */
EvaluationReport Campaign(EvaluationSuite const& suite, std::size_t stops, fs::path const& policy)
{
    EvaluationReport report;
    report.suite = suite.name;
    report.policySha256 = ppokit::artifacts::Sha256OfFile(policy);
    for (std::size_t index = 0; index < suite.Episodes(); ++index)
    {
        ppokit::eval::EpisodeResult episode;
        episode.seed = suite.seeds[index];
        episode.length = 10;
        episode.terminated = true;
        episode.metrics = {index < stops ? 1.0 : 0.0};
        report.episodes.push_back(episode);
    }
    ppokit::eval::Summarize(suite, report);
    return report;
}

/** @brief The BEST of the suite, or a description of why it cannot be read. */
std::string BestWeights(fs::path const& checkpoints)
{
    auto const best = ppokit::eval::ReadBest(checkpoints, "validation");
    return best ? Read(best->policy) : std::string("<none>");
}

/** @brief Two trainings reach the same update: each BEST keeps its own weights. */
void SameUpdateOfTwoTrainings(fs::path const& root, EvaluationSuite const& suite)
{
    fs::path const checkpoints = root / "same_update";
    fs::create_directories(checkpoints);
    fs::path const policy = checkpoints / "policy.ppokit";
    Write(policy, "weights of training A");
    Check(ppokit::eval::OfferBest(checkpoints, suite, 10, policy, Campaign(suite, 2, policy)), "a first BEST is promoted");
    auto const first = ppokit::eval::ReadBest(checkpoints, "validation");
    Check(first && first->update == 10 && Read(first->policy) == "weights of training A"
            && first->sha256 == ppokit::artifacts::Sha256Hex("weights of training A")
            && first->policy.filename().string() == "validation_10_" + first->sha256.substr(0, 16) + ".ppokit",
        "the generation is named by suite, update and digest, and the manifest keeps the digest");

    // A fresh training reaches update 10 again with other weights and a better campaign.
    Write(policy, "weights of training B");
    Check(ppokit::eval::OfferBest(checkpoints, suite, 10, policy, Campaign(suite, 3, policy)),
        "the better campaign of another training is promoted");
    auto const second = ppokit::eval::ReadBest(checkpoints, "validation");
    Check(second && Read(second->policy) == "weights of training B" && second->policy != first->policy,
        "the BEST of the same update holds the weights that were evaluated, in its own file");
    Check(Read(first->policy) == "weights of training A", "a generation is never rewritten");

    // The same weights offered again share their generation.
    Check(ppokit::eval::OfferBest(checkpoints, suite, 10, policy, Campaign(suite, 4, policy)), "a better campaign again");
    Check(ppokit::eval::ReadBest(checkpoints, "validation")->policy == second->policy,
        "the same weights at the same update reuse their generation");
    Check(!ppokit::eval::OfferBest(checkpoints, suite, 20, policy, Campaign(suite, 1, policy)),
        "a campaign that does not rank above the BEST is not promoted");
}

/** @brief A damaged or missing BEST is reported, and nothing replaces it silently. */
void DamagedBest(fs::path const& root, EvaluationSuite const& suite)
{
    fs::path const checkpoints = root / "damaged";
    fs::create_directories(checkpoints);
    fs::path const policy = checkpoints / "policy.ppokit";
    Write(policy, "evaluated weights");
    ppokit::eval::OfferBest(checkpoints, suite, 10, policy, Campaign(suite, 2, policy));
    fs::path const generation = ppokit::eval::ReadBest(checkpoints, "validation")->policy;
    std::string const manifest = Read(ppokit::eval::BestManifest(checkpoints, "validation"));

    Write(generation, "evaluated weightz");
    Check(Throws([&] { ppokit::eval::ReadBest(checkpoints, "validation"); }, "SHA-256 differs"),
        "a corrupted BEST policy is refused");
    Write(policy, "next weights");
    Check(Throws([&] { ppokit::eval::OfferBest(checkpoints, suite, 20, policy, Campaign(suite, 4, policy)); },
              "SHA-256 differs"),
        "a promotion over a corrupted BEST fails");
    Check(Read(ppokit::eval::BestManifest(checkpoints, "validation")) == manifest,
        "and leaves the manifest as it was");

    fs::remove(generation);
    Check(Throws([&] { ppokit::eval::ReadBest(checkpoints, "validation"); }, "is missing"),
        "a missing BEST policy is refused");
}

/** @brief A failed promotion keeps the previous BEST; an interrupted copy leaves no generation. */
void FailedPromotion(fs::path const& root, EvaluationSuite const& suite)
{
    fs::path const checkpoints = root / "failed";
    fs::create_directories(checkpoints);
    fs::path const policy = checkpoints / "policy.ppokit";
    Write(policy, "first weights");
    ppokit::eval::OfferBest(checkpoints, suite, 10, policy, Campaign(suite, 2, policy));
    std::string const manifest = Read(ppokit::eval::BestManifest(checkpoints, "validation"));

    Check(Throws([&] { ppokit::eval::OfferBest(checkpoints, suite, 20, checkpoints / "absent.ppokit",
                           Campaign(suite, 4, policy)); },
              "cannot open"),
        "offering a missing policy fails");
    Check(Read(ppokit::eval::BestManifest(checkpoints, "validation")) == manifest
            && BestWeights(checkpoints) == "first weights",
        "and keeps the previous BEST");

    // A delayed evaluation finishes after another actor replaced the writable alias at the same update.
    auto report = Campaign(suite, 4, policy);
    Write(policy, "another actor at the same update");
    Check(Throws([&] { ppokit::eval::OfferBest(checkpoints, suite, 10, policy, report); }, "policy evaluated"),
        "a successful evaluation cannot promote an actor it never used");
    report = Campaign(suite, 4, policy);
    report.suite = "another_suite";
    Check(Throws([&] { ppokit::eval::OfferBest(checkpoints, suite, 10, policy, report); }, "another evaluation suite"),
        "the score must belong to the target suite");
    Check(Read(ppokit::eval::BestManifest(checkpoints, suite.name)) == manifest
            && BestWeights(checkpoints) == "first weights", "stale reports leave the incumbent intact");

    // A generation of the same name with other bytes (tampered with) is not trusted.
    Write(policy, "second weights");
    std::string const digest = ppokit::artifacts::Sha256Hex("second weights");
    fs::path const best = checkpoints / "best";
    fs::path const clash = best / ("validation_20_" + digest.substr(0, 16) + ".ppokit");
    Write(clash, "other bytes");
    Check(Throws([&] { ppokit::eval::OfferBest(checkpoints, suite, 20, policy, Campaign(suite, 4, policy)); },
              "other contents"),
        "an existing generation with other contents is refused");
    Check(BestWeights(checkpoints) == "first weights", "and the previous BEST stays");
    fs::remove(clash);

    // A copy interrupted by a crash left its temporary file: the next promotion replaces it.
    fs::path const partial = best / ("validation_20_" + digest.substr(0, 16) + ".ppokit.partial");
    Write(partial, "second wei");
    Check(BestWeights(checkpoints) == "first weights", "an interrupted copy is not a BEST");
    Check(ppokit::eval::OfferBest(checkpoints, suite, 20, policy, Campaign(suite, 4, policy)),
        "the next promotion succeeds after an interrupted copy");
    Check(BestWeights(checkpoints) == "second weights" && !fs::exists(partial),
        "with the evaluated weights, and no temporary left");
}

/** @brief Manifests written before the digest still name their policy. */
void FormerManifest(fs::path const& root, EvaluationSuite const& suite)
{
    fs::path const checkpoints = root / "former";
    fs::create_directories(checkpoints / "best");
    Write(checkpoints / "best" / "validation_10.ppokit", "former weights");
    auto oldReport = Campaign(suite, 2, checkpoints / "best" / "validation_10.ppokit");
    oldReport.policySha256.clear();
    std::ostringstream manifest;
    manifest << R"({ "schema": "ppokit.best/1", "suite": "validation", "update": 10,
        "policy": "validation_10.ppokit", "report": )"
             << oldReport.ToJson() << " }";
    Write(ppokit::eval::BestManifest(checkpoints, "validation"), manifest.str());
    auto const best = ppokit::eval::ReadBest(checkpoints, "validation");
    Check(best && best->sha256.empty() && Read(best->policy) == "former weights",
        "a ppokit.best/1 manifest is read, without a digest");
    fs::path const policy = checkpoints / "policy.ppokit";
    Write(policy, "newer weights");
    Check(ppokit::eval::OfferBest(checkpoints, suite, 20, policy, Campaign(suite, 3, policy))
            && !ppokit::eval::ReadBest(checkpoints, "validation")->sha256.empty(),
        "and a better campaign replaces it by a ppokit.best/2 manifest");
}

/** @brief Concurrent rank decisions cannot let a slower, worse candidate replace the winner. */
void ConcurrentPromotion(fs::path const& root, EvaluationSuite const& suite)
{
    fs::path const checkpoints = root / "concurrent";
    fs::create_directories(checkpoints);
    fs::path const betterPolicy = checkpoints / "better.ppokit";
    fs::path const worsePolicy = checkpoints / "worse.ppokit";
    Write(betterPolicy, "better weights");
    Write(worsePolicy, "worse weights\n" + std::string(8 * 1024 * 1024, 'w'));
    EvaluationReport const better = Campaign(suite, 4, betterPolicy);
    EvaluationReport const worse = Campaign(suite, 2, worsePolicy);
    std::atomic<bool> start{false};
    std::atomic<int> errors{0};
    auto offer = [&](fs::path const& policy, EvaluationReport const& report) {
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        try
        {
            (void)ppokit::eval::OfferBest(checkpoints, suite, 10, policy, report);
        }
        catch (std::exception const&)
        {
            errors.fetch_add(1, std::memory_order_relaxed);
        }
    };
    std::thread worseThread(offer, std::cref(worsePolicy), std::cref(worse));
    std::thread betterThread(offer, std::cref(betterPolicy), std::cref(better));
    start.store(true, std::memory_order_release);
    worseThread.join();
    betterThread.join();
    Check(errors.load(std::memory_order_relaxed) == 0, "concurrent BEST offers complete without errors");
    Check(BestWeights(checkpoints) == "better weights", "concurrent promotion keeps the highest-ranked campaign");
}

} // namespace

int main()
{
    fs::path const root = fs::temp_directory_path()
        / ("ppokit_best_store_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    try
    {
        fs::create_directories(root);
        Sha256Vectors(root);
        EvaluationSuite const suite = Suite();
        SameUpdateOfTwoTrainings(root, suite);
        DamagedBest(root, suite);
        FailedPromotion(root, suite);
        FormerManifest(root, suite);
        ConcurrentPromotion(root, suite);
    }
    catch (std::exception const& error)
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        ++failures;
    }
    std::error_code ignored;
    fs::remove_all(root, ignored);
    if (failures != 0)
    {
        std::cerr << failures << " BEST store check(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "PPOKit BEST store tests passed\n";
    return EXIT_SUCCESS;
}
