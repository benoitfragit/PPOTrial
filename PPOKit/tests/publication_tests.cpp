/** @file PPOKit/tests/publication_tests.cpp
 * @brief Replay each observable stop point of a two-artifact checkpoint publication.
 */
#include "ppokit/runtime/publication.hpp"
#include "ppokit/runtime/run_state.hpp"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace ppokit;

void Require(bool result, char const* message)
{
    if (!result) throw std::runtime_error(message);
}

template<typename Operation>
void RequireRefusal(Operation const& operation)
{
    bool refused = false;
    try { operation(); } catch (std::exception const&) { refused = true; }
    Require(refused, "incoherent generation was accepted");
}

void Save(schema::Project const& project, int number)
{
    runtime::PublishGeneration(project, number, std::to_string(number), {static_cast<double>(-number), 0.0}, [&](auto const& training, auto const& policy) {
        std::ofstream(training) << "training " << number;
        std::ofstream(policy) << "policy " << number;
    });
}

std::string ReadText(fs::path const& path)
{
    std::ifstream input(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

/** @brief Publishers racing on one project still commit one coherent generation. */
void ConcurrentPublishers(schema::Project project)
{
    project.root /= "concurrent";
    fs::create_directories(project.root);
    constexpr int Publishers = 4;
    std::atomic<bool> start{false};
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int number = 1; number <= Publishers; ++number)
    {
        threads.emplace_back([&, number] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            try
            {
                Save(project, 100 + number);
            }
            catch (std::exception const&)
            {
                failures.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& thread : threads) thread.join();
    Require(failures.load(std::memory_order_relaxed) == 0, "a concurrent publisher failed");
    auto const generation = runtime::VerifyPublication(project);
    Require(generation && generation->update >= 101 && generation->update <= 104,
        "concurrent publishers did not leave a known generation");
    std::string const number = std::to_string(generation->update);
    Require(ReadText(project.LatestCheckpoint()) == "training " + number,
        "training checkpoint differs from the concurrent manifest");
    Require(ReadText(project.TrainingPolicy()) == "policy " + number,
        "policy differs from the concurrent manifest");
}

int main()
{
    try
    {
        schema::Project project;
        project.root = fs::temp_directory_path() / "ppokit_publication_tests";
        fs::remove_all(project.root);
        auto const root = project.Checkpoints();
        Save(project, 1);
        RequireRefusal([&] {
            runtime::PublishGeneration(project, 2, "2", {1.0}, [](auto const& train, auto const&) {
                std::ofstream(train) << "part";
                throw std::runtime_error("interrupted before policy");
            });
        });
        Require(runtime::VerifyPublication(project)->update == 1, "unprepared save modified generation one");
        auto const matching = project.root / "prepared";
        Save(project, 2);
        fs::create_directory(matching);
        fs::copy_file(project.LatestCheckpoint(), matching / "training");
        fs::copy_file(project.TrainingPolicy(), matching / "policy");
        fs::copy_file(root / "generation.json", matching / "generation.json");
        for (int installed = 0; installed <= 4; ++installed)
        {
            Save(project, 1);
            fs::copy(matching, root / ".publication-ready", fs::copy_options::recursive);
            if (installed >= 1) runtime::SaveTrainingState(root, 2, "2");
            if (installed >= 2) fs::copy_file(matching / "training", project.LatestCheckpoint(), fs::copy_options::overwrite_existing);
            if (installed >= 3) fs::copy_file(matching / "policy", project.TrainingPolicy(), fs::copy_options::overwrite_existing);
            if (installed >= 4) fs::copy_file(matching / "generation.json", root / "generation.json", fs::copy_options::overwrite_existing);
            RequireRefusal([&] { runtime::ReadPublicationManifest(project); });
            RequireRefusal([&] { runtime::VerifyPublication(project); });
            runtime::RecoverPublication(project);
            runtime::RecoverPublication(project);
            auto const state = runtime::VerifyPublication(project);
            Require(state && state->update == 2 && state->state == "2" && state->rewardWeights
                    && *state->rewardWeights == std::vector<double>({-2.0, 0.0}), "replay lost generation context");
        }

        Save(project, 1);
        fs::copy(matching, root / ".publication-ready", fs::copy_options::recursive);
        std::ofstream(root / ".publication-ready" / "policy", std::ios::binary | std::ios::trunc) << "damaged";
        RequireRefusal([&] { runtime::RecoverPublication(project); });
        Require(ReadText(project.LatestCheckpoint()) == "training 1"
                && ReadText(project.TrainingPolicy()) == "policy 1",
            "a damaged prepared generation modified the valid generation");
        fs::remove_all(root / ".publication-ready");
        Require(runtime::VerifyPublication(project)->update == 1,
            "the valid generation was lost after refusing damaged staging");

        ConcurrentPublishers(project);
        std::ofstream(project.TrainingPolicy()) << "unrelated";
        RequireRefusal([&] { runtime::VerifyPublication(project); });
        fs::remove_all(project.root);
    }
    catch (std::exception const& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
