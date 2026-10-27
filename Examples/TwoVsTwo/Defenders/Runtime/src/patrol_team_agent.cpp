#include "patrol_team_agent.hpp"

#include "patrol_agent.hpp"
#include "patrol_checkpoint.hpp"
#include "patrol_learner.hpp"
#include "patrol_policy.hpp"
#include "initializer.hpp"
#include "components.hpp"
#include "sharedworldcomponents.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace ppo::examples::twovstwo::defenders {

std::shared_ptr<PatrolPolicy>
LoadPatrolPolicy(std::filesystem::path const& checkpoint)
{
    return std::make_shared<PatrolPolicy>(checkpoint);
}

std::shared_ptr<PatrolPolicy> CreateUntrainedPatrolPolicy(std::uint32_t seed)
{
    return std::make_shared<PatrolPolicy>(seed);
}

namespace {

Mobile FromEcs(ppo::model::BaseEntity& entity)
{
    auto const [state] = entity.get<ppo::model::sEntityState>();
    float turnRate = 0.F;
    if (entity.has<ppo::model::sFlightModelState>())
    {
        auto const [flight] = entity.get<ppo::model::sFlightModelState>();
        turnRate = flight.m_accpsi;
    }
    return {state.m_x, state.m_y, state.m_rotation, state.m_speed,
            turnRate, !state.m_destroyed};
}

DefenseTrack ToDefenseTrack(Mobile const& mobile) noexcept
{
    return {mobile.north, mobile.east, mobile.heading,
            mobile.speed, mobile.available};
}

} // namespace

PatrolMappoAgent::PatrolMappoAgent(
    std::array<ppo::model::BaseEntity*, 2> const& guards,
    std::vector<ppo::model::BaseEntity*> const& attackers,
    ppo::model::BaseEntity& objective,
    std::shared_ptr<PatrolPolicy> policy,
    Parameters parameters,
    PatrolTrainingSettings training,
    std::filesystem::path latestCheckpoint,
    std::filesystem::path trainingCheckpoint,
    std::filesystem::path productionCheckpoint)
    : registry_(objective.registry()), objectiveId_(objective.uniqid()),
      policy_(std::move(policy)), parameters_(parameters), training_(training),
      latestCheckpoint_(std::move(latestCheckpoint)),
      trainingCheckpoint_(std::move(trainingCheckpoint)),
      productionCheckpoint_(std::move(productionCheckpoint)),
      evaluation_(training.evaluation)
{
    if (registry_ == nullptr || guards[0] == nullptr || guards[1] == nullptr
        || guards[0]->registry() != registry_ || guards[1]->registry() != registry_
        || !policy_ || !Valid(parameters_) || !Valid(training_)
        || latestCheckpoint_.empty() || trainingCheckpoint_.empty()
        || productionCheckpoint_.empty()
        || attackers.size() > attackerIds_.size())
        throw std::invalid_argument("Invalid blue MAPPO team configuration");

    for (std::size_t role = 0; role < guardIds_.size(); ++role)
        guardIds_[role] = guards[role]->uniqid();
    for (std::size_t i = 0; i < attackers.size(); ++i)
    {
        if (attackers[i] == nullptr || attackers[i]->registry() != registry_)
            throw std::invalid_argument("Invalid red track for blue MAPPO");
        attackerIds_[i] = attackers[i]->uniqid();
    }
    LoadPromotedScore();
}

PatrolMappoAgent::~PatrolMappoAgent()
{
    SaveLatest();
}

void PatrolMappoAgent::Init()
{
    // Preserve in-memory optimizer state and a partially collected rollout
    // across ECS episode resets. Only transient physical state is cleared.
    episodeStep_ = 0U;
    currentReward_ = 0.F;
    rewardValues_.fill(0.F);
    defenseCoordinator_.Reset();
    defenseDecision_ = DefenseDecision{};
    hasDefenseDecision_ = false;
    hasBeforeStep_ = false;
    episodeDone_ = false;
    if (evaluation_.InProgress())
    {
        try
        {
            Parameters const parameters = ReadParameters();
            if (!Valid(parameters))
                throw std::runtime_error("Invalid evaluation ECS parameters");
            evaluation_.BeginEpisode(ReadState(parameters));
        }
        catch (std::exception const& error)
        {
            std::cerr << "[BLUE MAPPO] Evaluation initialization failed: "
                      << error.what() << '\n';
            evaluation_.Cancel();
            RestoreTrainingInitializer();
            finished_ = true;
            episodeDone_ = true;
        }
    }
}

Parameters PatrolMappoAgent::ReadParameters() const
{
    Parameters result = parameters_;
    if (registry_ != nullptr
        && registry_->ctx().contains<ppo::model::sEnvironmentState>())
    {
        auto const& world = registry_->ctx().get<ppo::model::sEnvironmentState>();
        result.halfNorth = 0.5F * world.m_maxNorth;
        result.halfEast = 0.5F * world.m_maxEast;
        result.dt = world.m_dt;
    }
    if (registry_ != nullptr
        && registry_->ctx().contains<
            ppo::examples::twovstwo::InitializerContext>())
    {
        auto const& context = registry_->ctx().get<
            ppo::examples::twovstwo::InitializerContext>();
        if (context.patrolGeometryReady)
            result.geometry = context.patrolGeometry;
    }
    ppo::model::BaseEntity first{registry_, GuardIdForRole(0U)};
    if (first.has<ppo::model::sFlightModelState>())
    {
        auto const [flight] = first.get<ppo::model::sFlightModelState>();
        result.maxSpeed = flight.m_maxSpeed;
        result.maxAcceleration = flight.m_maxAcc;
        result.maxRotationRate = flight.m_maxAccRot;
    }
    return result;
}

entt::entity PatrolMappoAgent::GuardIdForRole(std::size_t role) const noexcept
{
    std::size_t guardIndex = role % guardIds_.size();
    if (registry_ != nullptr
        && registry_->ctx().contains<
            ppo::examples::twovstwo::InitializerContext>())
    {
        auto const& context = registry_->ctx().get<
            ppo::examples::twovstwo::InitializerContext>();
        guardIndex = context.GuardForRole(role);
    }
    return guardIds_[guardIndex];
}

State PatrolMappoAgent::ReadState(Parameters const& parameters) const
{
    State result{};
    result.controlledRole = 0U;
    result.step = episodeStep_;
    if (registry_ != nullptr
        && registry_->ctx().contains<ppo::examples::twovstwo::InitializerContext>())
    {
        auto const& context = registry_->ctx().get<
            ppo::examples::twovstwo::InitializerContext>();
        result.side = context.side;
        result.arclength = context.arclength;
    }
    else
    {
        result.side = 1;
    }

    for (std::size_t role = 0; role < guardIds_.size(); ++role)
    {
        ppo::model::BaseEntity guard{registry_, GuardIdForRole(role)};
        result.guards[role] = FromEcs(guard);
        Mobile const& mobile = result.guards[role];
        result.failed |= !mobile.available || !std::isfinite(mobile.north)
            || !std::isfinite(mobile.east) || !std::isfinite(mobile.heading)
            || !std::isfinite(mobile.speed)
            || std::abs(mobile.north)
                >= parameters.hardBoundaryScale * parameters.halfNorth
            || std::abs(mobile.east)
                >= parameters.hardBoundaryScale * parameters.halfEast;
    }
    for (std::size_t i = 0; i < attackerIds_.size(); ++i)
    {
        if (attackerIds_[i] == entt::null || !registry_->valid(attackerIds_[i]))
        {
            result.attackers[i] = Mobile{0.F, 0.F, 0.F, 0.F, 0.F, false};
            continue;
        }
        ppo::model::BaseEntity attacker{registry_, attackerIds_[i]};
        result.attackers[i] = FromEcs(attacker);
    }

    if (objectiveId_ == entt::null || !registry_->valid(objectiveId_))
        result.objectiveAvailable = false;
    else
    {
        ppo::model::BaseEntity objective{registry_, objectiveId_};
        auto const [state] = objective.get<ppo::model::sEntityState>();
        result.objectiveAvailable = !state.m_destroyed;
    }
    UpdatePatrolProgress(parameters, result);
    return result;
}

void PatrolMappoAgent::UpdateDefenseState(Parameters const& parameters,
                                          State& state)
{
    std::array<DefenseTrack, 2> guards{
        ToDefenseTrack(state.guards[0]), ToDefenseTrack(state.guards[1])};
    std::array<DefenseTrack, 2> attackers{
        ToDefenseTrack(state.attackers[0]), ToDefenseTrack(state.attackers[1])};
    DefenseTrack const objective{
        0.F,
        static_cast<float>(state.side >= 0 ? 1 : -1)
            * 0.92F * parameters.halfEast,
        0.F, 0.F, state.objectiveAvailable};
    defenseDecision_ = defenseCoordinator_.Update(
        parameters.defense, parameters.dt, guards, attackers, objective);
    hasDefenseDecision_ = true;
    ApplyDefenseDecision(state, defenseDecision_);
}

void PatrolMappoAgent::ApplyCurrentDefenseState(State& state) const noexcept
{
    if (hasDefenseDecision_)
        ApplyDefenseDecision(state, defenseDecision_);
}

std::array<Observation, 2> PatrolMappoAgent::BuildObservations(
    Parameters const& parameters, State const& state) const
{
    std::array<Observation, 2> observations{};
    for (std::size_t role = 0; role < observations.size(); ++role)
    {
        State local = state;
        local.controlledRole = role;
        observations[role] = Observe(parameters, local);
    }
    return observations;
}

void PatrolMappoAgent::ChooseAction()
{
    if (episodeDone_ || registry_ == nullptr)
        return;
    for (entt::entity id : guardIds_)
    {
        ppo::model::BaseEntity guard{registry_, id};
        if (!guard.has<ppo::model::sEntityState, ppo::model::sSNAControlState>())
            return;
    }

    Parameters const parameters = ReadParameters();
    if (!Valid(parameters))
        return;

    try
    {
        beforeStep_ = ReadState(parameters);
        if (hasDefenseDecision_)
            ApplyCurrentDefenseState(beforeStep_);
        else
            UpdateDefenseState(parameters, beforeStep_);
        hasBeforeStep_ = true;
        auto const observations = BuildObservations(parameters, beforeStep_);
        if (!finished_ && !evaluation_.InProgress())
        {
            if (!learner_)
            {
                learner_ = std::make_unique<PatrolLearner>(
                    parameters_, training_.seed, training_.rolloutSteps);
                if (std::filesystem::exists(trainingCheckpoint_))
                {
                    learner_->LoadTrainingState(trainingCheckpoint_);
                    updates_ = learner_->StepCount();
                    lastSavedUpdate_ = updates_;
                    PublishActor();
                    finished_ = updates_ >= training_.maxUpdates;
                    std::cout << "[BLUE MAPPO] Resumed full training state at update "
                              << updates_ << " from "
                              << trainingCheckpoint_.string() << '\n';
                }
                else if (policy_->IsTrained())
                {
                    std::filesystem::path const source =
                        std::filesystem::exists(latestCheckpoint_)
                        ? latestCheckpoint_ : productionCheckpoint_;
                    if (!std::filesystem::exists(source))
                        throw std::runtime_error(
                            "Trained blue policy has no Actor warm-start checkpoint");
                    learner_->LoadActor(source);
                    std::cout << "[BLUE MAPPO] Loaded legacy Actor-only warm-start from "
                              << source.string() << '\n';
                }
            }
            if (!finished_)
                lastActions_ = learner_->ChooseActions(observations);
            else
                for (std::size_t role = 0; role < lastActions_.size(); ++role)
                    lastActions_[role] = policy_->Predict(observations[role]);
        }
        else
        {
            for (std::size_t role = 0; role < lastActions_.size(); ++role)
                lastActions_[role] = policy_->Predict(observations[role]);
        }

        for (std::size_t role = 0; role < guardIds_.size(); ++role)
        {
            ppo::model::BaseEntity guard{registry_, GuardIdForRole(role)};
            auto [control] = guard.get<ppo::model::sSNAControlState>();
            control.m_rotSpeedCmd = Clamp(lastActions_[role][0], -1.F, 1.F);
            control.m_accCmd = Clamp(lastActions_[role][1], -1.F, 1.F);
        }
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO] Action failed: " << error.what() << '\n';
        if (evaluation_.InProgress())
        {
            evaluation_.Cancel();
            RestoreTrainingInitializer();
        }
        finished_ = true;
        episodeDone_ = true;
    }
}

bool PatrolMappoAgent::SetNewState()
{
    if (!hasBeforeStep_)
        return episodeDone_;

    ++episodeStep_;
    try
    {
        Parameters const parameters = ReadParameters();
        if (!Valid(parameters))
            throw std::runtime_error("Invalid ECS parameters for MAPPO feedback");

        State after = ReadState(parameters);
        UpdateDefenseState(parameters, after);
        after.collided = GuardsCollided(parameters, beforeStep_, after);
        after.failed |= after.collided;
        bool pendingThreats = false;
        if (registry_->ctx().contains<
                ppo::examples::twovstwo::InitializerContext>())
        {
            auto const& context = registry_->ctx().get<
                ppo::examples::twovstwo::InitializerContext>();
            pendingThreats = context.episodeLayout
                && ppo::examples::twovstwo::HasPendingThreats(
                    *context.episodeLayout, context.elapsedSteps);
        }
        bool const defeatedThreats = HasActiveThreat(beforeStep_)
                                  && !HasActiveThreat(after)
                                  && !pendingThreats;
        bool const terminated = after.failed || !after.objectiveAvailable
                             || defeatedThreats;
        bool const truncated = !terminated && episodeStep_ >=
            (evaluation_.InProgress() ? training_.evaluation.maxSteps
                                      : parameters.episodeSteps);

        RewardBreakdown const breakdown = TeamRewardBreakdown(
            parameters, beforeStep_, lastActions_, after, pendingThreats);
        for (std::size_t i = 0; i < rewardValues_.size(); ++i)
            rewardValues_[i] = breakdown.values[i] * training_.rewardWeights[i];
        currentReward_ = breakdown.Total(training_.rewardWeights,
                                         training_.rewardClamp);

        if (evaluation_.InProgress())
        {
            evaluation_.ObserveStep(parameters, after);
            episodeDone_ = terminated || truncated;
            if (episodeDone_)
            {
                bool const hasMore = evaluation_.FinishEpisode(
                    parameters, after, truncated);
                if (hasMore)
                    PrepareEvaluationCase(parameters);
                else
                {
                    RestoreTrainingInitializer();
                    if (evaluation_.Improved())
                        PromoteEvaluatedActor();
                    std::cout << "[BLUE MAPPO] Evaluation #"
                              << evaluation_.CompletedCampaigns()
                              << " score=" << evaluation_.LatestScore()
                              << " success=" << evaluation_.SuccessRate()
                              << " survival=" << evaluation_.SurvivalRate()
                              << "\n";
                    if (finished_)
                        std::cout << "[BLUE MAPPO] Training completed after "
                                  << updates_ << " updates\n";
                }
            }
        }
        else if (!finished_ && learner_ && learner_->HasPendingAction())
        {
            bool const updated = learner_->Observe(
                BuildObservations(parameters, after), currentReward_,
                terminated, truncated);
            if (updated)
            {
                updates_ = learner_->StepCount();
                PublishActor();
                finished_ = updates_ >= training_.maxUpdates;
                // Establish an early deterministic baseline, then keep the
                // configured cadence. The UI bar remains evaluation-only.
                if (updates_ == 1U
                    || updates_ % training_.checkpointEvery == 0U || finished_)
                {
                    SaveLatest();
                    StartEvaluation(parameters);
                    episodeDone_ = true;
                }
            }
            if (!evaluation_.InProgress())
                episodeDone_ = terminated || truncated;
        }
        else
            episodeDone_ = terminated || truncated;
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO] Feedback failed: " << error.what() << '\n';
        if (evaluation_.InProgress())
        {
            evaluation_.Cancel();
            RestoreTrainingInitializer();
        }
        finished_ = true;
        episodeDone_ = true;
    }
    hasBeforeStep_ = false;
    return episodeDone_;
}

void PatrolMappoAgent::PublishActor()
{
    if (!learner_)
        return;
    std::ostringstream serialized;
    learner_->WithActor([&](auto const& actor) {
        WritePatrolPolicy(serialized, actor);
    });
    policy_->ReloadFromText(serialized.str());
}

void PatrolMappoAgent::SaveTo(std::filesystem::path const& checkpoint)
{
    if (!learner_ || updates_ == 0U)
        return;
    learner_->WithActor([&](auto const& actor) {
        SavePatrolPolicy(checkpoint, actor);
    });
}

void PatrolMappoAgent::SaveLatest() noexcept
{
    if (!learner_ || updates_ == 0U || updates_ == lastSavedUpdate_)
        return;
    try
    {
        learner_->SaveTrainingState(trainingCheckpoint_);
        SaveTo(latestCheckpoint_);
        lastSavedUpdate_ = updates_;
        std::cout << "[BLUE MAPPO] Saved full training state at update "
                  << updates_ << " to " << trainingCheckpoint_.string()
                  << '\n';
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO] Checkpoint save failed: "
                  << error.what() << '\n';
    }
}

void PatrolMappoAgent::StartEvaluation(Parameters const& parameters)
{
    if (registry_ == nullptr
        || !registry_->ctx().contains<ppo::examples::twovstwo::InitializerContext>())
        throw std::runtime_error("Missing initializer context for evaluation");
    auto& context = registry_->ctx().get<
        ppo::examples::twovstwo::InitializerContext>();
    trainingRng_ = context.rng;
    trainingRngSaved_ = true;
    evaluation_.Start(!context.patrolOnly);
    try
    {
        PrepareEvaluationCase(parameters);
    }
    catch (...)
    {
        evaluation_.Cancel();
        RestoreTrainingInitializer();
        throw;
    }
    std::cout << "[BLUE MAPPO] Starting deterministic evaluation after update "
              << updates_ << " (" << evaluation_.TestCount() << " cases)\n";
}

void PatrolMappoAgent::PrepareEvaluationCase(Parameters const& parameters)
{
    if (registry_ == nullptr
        || !registry_->ctx().contains<ppo::examples::twovstwo::InitializerContext>())
        throw std::runtime_error("Missing initializer context for evaluation case");
    PatrolEvaluationCase const test = evaluation_.CurrentCase(parameters);
    auto& context = registry_->ctx().get<
        ppo::examples::twovstwo::InitializerContext>();
    context.evaluationOverride = true;
    context.evaluationSide = test.side;
    context.evaluationArclength = test.arclength;
    context.evaluationSecondAttackerDestroyed = test.secondAttackerDestroyed;
    context.rng.seed(test.seed);
    context.next = 0U;
}

void PatrolMappoAgent::RestoreTrainingInitializer() noexcept
{
    if (registry_ == nullptr
        || !registry_->ctx().contains<ppo::examples::twovstwo::InitializerContext>())
        return;
    auto& context = registry_->ctx().get<
        ppo::examples::twovstwo::InitializerContext>();
    context.evaluationOverride = false;
    context.next = 0U;
    if (trainingRngSaved_)
    {
        context.rng = trainingRng_;
        trainingRngSaved_ = false;
    }
}

void PatrolMappoAgent::PromoteEvaluatedActor() noexcept
{
    try
    {
        SaveTo(productionCheckpoint_);
        SavePromotedScore();
        std::cout << "[BLUE MAPPO] Promoted evaluated Actor to "
                  << productionCheckpoint_.string() << '\n';
    }
    catch (std::exception const& error)
    {
        std::cerr << "[BLUE MAPPO] Actor promotion failed: "
                  << error.what() << '\n';
    }
}

void PatrolMappoAgent::LoadPromotedScore() noexcept
{
    try
    {
        if (!std::filesystem::exists(productionCheckpoint_))
            return;
        auto scorePath = productionCheckpoint_;
        scorePath += ".score";
        std::ifstream stream{scorePath};
        std::string magic;
        float score = 0.F;
        if (stream && stream >> magic >> score
            && magic == "BLUE_PATROL_EVALUATION_V1")
            evaluation_.RestoreBestScore(score);
    }
    catch (...)
    {
        // Missing or malformed optional metadata only forces one re-evaluation.
    }
}

void PatrolMappoAgent::SavePromotedScore()
{
    auto scorePath = productionCheckpoint_;
    scorePath += ".score";
    if (scorePath.has_parent_path())
        std::filesystem::create_directories(scorePath.parent_path());
    auto temporary = scorePath;
    temporary += ".tmp";
    auto backup = scorePath;
    backup += ".bak";
    {
        std::ofstream stream{temporary, std::ios::trunc};
        if (!stream)
            throw std::runtime_error("Cannot create blue evaluation score");
        stream << "BLUE_PATROL_EVALUATION_V1 "
               << std::setprecision(9) << evaluation_.BestScore() << '\n';
        if (!stream)
            throw std::runtime_error("Cannot write blue evaluation score");
    }
    std::error_code ignored;
    std::filesystem::remove(backup, ignored);
    bool const existed = std::filesystem::exists(scorePath);
    if (existed)
        std::filesystem::rename(scorePath, backup);
    try
    {
        std::filesystem::rename(temporary, scorePath);
    }
    catch (...)
    {
        if (existed)
            std::filesystem::rename(backup, scorePath);
        throw;
    }
    if (existed)
        std::filesystem::remove(backup, ignored);
}

void PatrolMappoAgent::ProcessCommand(
    ppo::model::sAgentRealTimeCommand const& command) noexcept
{
    if (command.m_id == 0U || command.m_id != id_)
        return;
    switch (command.m_type)
    {
    case ppo::model::sAgentRealTimeCommand::eTypeCommand::RewardClamp:
        if (std::isfinite(command.m_rewardClamp) && command.m_rewardClamp > 0.F)
            training_.rewardClamp = command.m_rewardClamp;
        break;
    case ppo::model::sAgentRealTimeCommand::eTypeCommand::RewardScaling:
        if (command.m_rewardScalingType < training_.rewardWeights.size()
            && std::isfinite(command.m_rewardScaling)
            && command.m_rewardScaling >= 0.F)
            training_.rewardWeights[command.m_rewardScalingType] =
                command.m_rewardScaling;
        break;
    case ppo::model::sAgentRealTimeCommand::eTypeCommand::RolloutBuffer:
    {
        std::size_t const requested = command.m_rolloutBuffer > 0
            ? static_cast<std::size_t>(command.m_rolloutBuffer) : 0U;
        // Changing tensor dimensions discards optimizer moments. Permit it
        // only before the first collected transition; later changes belong in
        // Training/BluePatrol.json and take effect after restarting the run.
        if (IsSupportedRolloutSize(requested) && updates_ == 0U
            && (!learner_ || (!learner_->HasPendingAction()
                && learner_->CollectedSteps() == 0U)))
        {
            training_.rolloutSteps = requested;
            learner_.reset();
        }
        break;
    }
    default:
        break;
    }
}

ppo::model::sAgentStateSnaphot PatrolMappoAgent::GetSnapshot() const noexcept
{
    ppo::model::sAgentStateSnaphot snapshot{};
    snapshot.m_name = "Blue MAPPO (RLTools)";
    snapshot.m_id = id_;
    snapshot.m_common.m_step = learner_
        ? static_cast<long long>(learner_->CollectedSteps()) : 0LL;
    snapshot.m_common.m_maxStep = static_cast<long long>(training_.rolloutSteps);
    snapshot.m_common.m_reward = currentReward_;
    snapshot.m_common.m_rewardClamp = training_.rewardClamp;
    snapshot.m_common.m_checkpoint = static_cast<unsigned int>(
        training_.checkpointEvery);
    snapshot.m_common.m_version = static_cast<unsigned int>(updates_);
    if (learner_)
    {
        PatrolTrainingDiagnostics const diagnostics = learner_->Diagnostics();
        snapshot.m_common.m_loss = diagnostics.totalLoss;
        snapshot.m_common.m_actorloss = diagnostics.actorLoss;
        snapshot.m_common.m_valueloss = diagnostics.valueLoss;
        snapshot.m_common.m_entropyloss = diagnostics.entropyLoss;
    }
    snapshot.m_common.m_done = episodeDone_;
    snapshot.m_common.m_training = !finished_ && !evaluation_.InProgress();
    snapshot.m_common.m_notation = evaluation_.BestScore();
    snapshot.m_common.m_latestEvaluationNotation = evaluation_.LatestScore();
    snapshot.m_common.m_completedEvaluations = evaluation_.CompletedCampaigns();
    snapshot.m_common.m_hasLatestEvaluation = evaluation_.HasLatest();
    snapshot.m_common.m_latestEvaluationSuccessRate = evaluation_.SuccessRate();
    snapshot.m_common.m_latestEvaluationSurvivalRate = evaluation_.SurvivalRate();
    snapshot.m_common.m_latestEvaluationTestCount = static_cast<unsigned int>(
        evaluation_.TestCount());
    snapshot.m_common.m_evaluationTestsFinished = static_cast<unsigned int>(
        evaluation_.TestsFinished());
    snapshot.m_common.m_evaluationTimeoutTests = static_cast<unsigned int>(
        evaluation_.TimeoutTests());
    snapshot.m_common.m_latestEvaluationTimeoutRate = evaluation_.TimeoutRate();
    snapshot.m_common.m_evaluationStepsExecuted = evaluation_.StepsExecuted();
    snapshot.m_common.m_evaluationStepsPerSecond = evaluation_.StepsPerSecond();
    snapshot.m_common.m_evaluationElapsedSeconds = evaluation_.ElapsedSeconds();
    snapshot.m_common.m_evaluationprogress = evaluation_.Progress();
    snapshot.m_common.m_evaluationinprogress = evaluation_.InProgress();
    snapshot.m_rewards.assign(rewardValues_.begin(), rewardValues_.end());
    snapshot.m_scalings.assign(training_.rewardWeights.begin(),
                               training_.rewardWeights.end());
    snapshot.m_rewardnames.reserve(kRewardChannelNames.size());
    for (char const* name : kRewardChannelNames)
        snapshot.m_rewardnames.emplace_back(name);
    return snapshot;
}

} // namespace ppo::examples::twovstwo::defenders
