#include "targetselector.hpp"
#include "utils.hpp"
#include "snasystem.hpp"

namespace rl {

void
StatesAndActionsForMission::Init(void)
{
    // Ensure the entity has the required components
    if (!m_entity.has<ppo::model::sEntityState>())
        return;

    auto [myState] =
        m_entity.get<ppo::model::sEntityState>();

    m_threats.clear();

    auto registry = m_entity.registry();

    // View containing entity state and SAM capability data
    auto view = registry->view<ppo::model::sEntityState, ppo::model::sSNAState>();

    // Additional safety margin used when deciding if two SAM zones belong
    // to the same cluster.
    constexpr float gatheringSecurityRange = 10000.f;

    // Structure used to cache SAM data locally for faster processing
    struct SamData
    {
        entt::entity e;
        float x;
        float y;
        float nez;
    };

    std::vector<SamData> sams;
    sams.reserve(64);

    // ---------------------------------------------------------------------
    // Collect all enemy SAMs
    // ---------------------------------------------------------------------

    for (auto entity : view)
    {
        auto [state, sna] = view.get(entity);

        // Only consider alive enemy SAM sites
        if (state.m_type != ppo::model::eEntityType::SAM ||
            state.m_side == myState.m_side ||
            state.m_destroyed)
            continue;

        sams.push_back({ entity, state.m_x, state.m_y, sna.m_noescapezone });
    }

    const size_t n = sams.size();

    if (n == 0)
        return;

    // ---------------------------------------------------------------------
    // Union-Find structure
    //
    // This structure groups SAMs into connected components.
    // If SAM A is close to B and B is close to C, then A,B,C will belong
    // to the same cluster even if A is not directly close to C.
    // ---------------------------------------------------------------------

    std::vector<size_t> parent(n);

    for (size_t i = 0; i < n; ++i)
        parent[i] = i;

    // Find with path compression
    auto find = [&](size_t a)
        {
            while (parent[a] != a)
                a = parent[a] = parent[parent[a]];
            return a;
        };

    // Union operation
    auto unite = [&](size_t a, size_t b)
        {
            a = find(a);
            b = find(b);

            if (a != b)
                parent[b] = a;
        };

    // ---------------------------------------------------------------------
    // Build connectivity graph between SAMs
    //
    // Two SAMs are considered connected if their NEZ plus the security
    // gathering margin overlap.
    // ---------------------------------------------------------------------

    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = i + 1; j < n; ++j)
        {
            float dx = sams[i].x - sams[j].x;
            float dy = sams[i].y - sams[j].y;

            float d = std::sqrt(dx * dx + dy * dy);

            if (d - sams[i].nez - sams[j].nez < gatheringSecurityRange)
                unite(i, j);
        }
    }

    // ---------------------------------------------------------------------
    // Rebuild clusters from the union-find structure
    // ---------------------------------------------------------------------

    std::unordered_map<size_t, std::vector<size_t>> clusters;

    for (size_t i = 0; i < n; ++i)
        clusters[find(i)].push_back(i);

    m_threats.reserve(clusters.size());

    // ---------------------------------------------------------------------
    // For each cluster compute:
    //  - weighted barycenter (using NEZ as weight)
    //  - global no-escape-zone radius covering all SAMs
    // ---------------------------------------------------------------------

    for (auto& [root, ids] : clusters)
    {
        sRealThreat threat{};

        float weightSum = 0.f;

        // Compute weighted center
        for (auto id : ids)
        {
            const auto& sam = sams[id];

            threat.m_sam.push_back(sam.e);

            threat.m_x += sam.x * sam.nez;
            threat.m_y += sam.y * sam.nez;

            weightSum += sam.nez;
        }

        if (weightSum > 0.f)
        {
            threat.m_x /= weightSum;
            threat.m_y /= weightSum;
        }

        // Compute the global NEZ radius needed to cover all SAM NEZ
        for (auto id : ids)
        {
            const auto& sam = sams[id];

            float dx = sam.x - threat.m_x;
            float dy = sam.y - threat.m_y;

            float d = std::sqrt(dx * dx + dy * dy);

            threat.m_noescapeZone =
                std::max(threat.m_noescapeZone, d + sam.nez);
        }

        m_threats.push_back(std::move(threat));
    }
}

void
StatesAndActionsForMission::SetNextAction(ACTIONS const& action) noexcept
{
    if (!m_entity.has<ppo::model::sSNAControlState>())
        return;

    auto [snaControlState] = m_entity.get<ppo::model::sSNAControlState>();

    snaControlState.m_rotSpeedCmd   = std::clamp(action[0], -1.F, 1.F);
    snaControlState.m_accCmd        = std::clamp(action[1], -1.F, 1.F);

    // action 2 for further use is missile departure
}

void 
StatesAndActionsForMission::SetNewState(STATES& state) noexcept
{
    memset(state.data(), 0, sizeof(float) * state.size());

    auto registry{ m_entity.registry() };
    if (registry == nullptr)
        return;

    auto [aircraftState, snaState, snaControlState] = m_entity.get< ppo::model::sEntityState, 
                                                                    ppo::model::sSNAState, 
                                                                    ppo::model::sSNAControlState>();

    unsigned int j{ 0 };
    auto v = registry->view<ppo::model::sEntityState>();

    std::array<float, 61> nearestCollisionRanges{};
    float const maxAngleArea{ 180.F * static_cast<float>(ppo::model::DEG_TO_RAD) };
    float const halfAngleArea{ maxAngleArea / 2.F };
    float const stepAngle{ maxAngleArea / static_cast<float>(nearestCollisionRanges.size()-1) };
    // maximumal free range area
    for (auto& r : nearestCollisionRanges)
        r = 1.F;

    for (auto const& threat : m_threats)
    {
        float angle{ -halfAngleArea };

        for (auto iang = 0; iang < nearestCollisionRanges.size(); ++iang)
        {
            // direction to be checked
            float const angleRad{ aircraftState.m_rotation + angle };

            float mx{ aircraftState.m_x - threat.m_x };
            float my{ aircraftState.m_y - threat.m_y };

            float b = mx * cos(angleRad) + my * sin(angleRad);
            float c = mx * mx + my * my - (1.2F*1.2F*threat.m_noescapeZone * threat.m_noescapeZone);

            float const delta = b * b - c;

            // there is an intersection
            if (delta >= 0.F)
            {
                float const sqrtDelta = sqrt(delta);
                float const d1{ -b - sqrtDelta };
                float const d2{ -b + sqrtDelta };

                // first possible intersection
                if (d1 >= 0.F)
                    nearestCollisionRanges[iang] = std::min(nearestCollisionRanges[iang], d1 / (d1 + snaState.m_noescapezone));

                // second possible intersection
                if (d2 >= 0.F)
                    nearestCollisionRanges[iang] = std::min(nearestCollisionRanges[iang], d2 / (d2 + snaState.m_noescapezone));
            }

            angle += stepAngle;
        }
    }

    for (auto entity : v)
    {
        ppo::model::BaseEntity otherEntity{ registry, entity };

        auto [entityState] = otherEntity.get<ppo::model::sEntityState>();

        if (entityState.m_destroyed
        ||  entityState.m_entityId == m_entity.uniqid()
        ||  entityState.m_side == aircraftState.m_side)
            continue;

        float const dx{ entityState.m_x - aircraftState.m_x };
        float const dy{ entityState.m_y - aircraftState.m_y };
        float const d{ sqrt(dx * dx + dy * dy) };
        float const ux{ dx / (d + 1e-6F) };
        float const uy{ dy / (d + 1e-6F) };
        float az = atan2(uy, ux);
        az = ppo::model::wrapMinusPi2Pi(az);
        az = ppo::model::wrapMinusPi2Pi(az - aircraftState.m_rotation);

        // SAM are not considered as targets rather as collision risks, so we ignore them for the state representation of targets, but we keep track of them for collision risk estimation
        if (j < MAX_TRACKS
        && entityState.m_type != ppo::model::eEntityType::SAM)
        {
            float importancy = 0.F;
            if (snaState.m_target == entity)
                importancy = 1.F;

            float const vrx{ entityState.m_dx - aircraftState.m_dx };
            float const vry{ entityState.m_dy - aircraftState.m_dy };

            float const closingSpeed = -(vrx * ux + vry * uy);

            float side{ 1.F };
            if (entityState.m_side == ppo::model::eEntitySide::Unknown
             || entityState.m_side == aircraftState.m_side)
                side = -0.F;

            state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::NorthRange)]   = dx / std::max(d, 1e-6F);
            state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::EastRange)]    = dy / std::max(d, 1e-6F);
            state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::Range)]        = d/ (d + snaState.m_noescapezone);
            state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::CosDir)]       = cos(az);
            state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::SinSin)]       = sin(az);
            state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::ClosingSpeed)] = std::tanh(closingSpeed / 800.F);
            state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::Hostile)]      = side;
            state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::Importancy)]   = importancy;
            state[j * static_cast<unsigned int>(eFeatures::_Size) + static_cast<unsigned int>(eFeatures::Available)]    = 1.F;

            ++j;
        }
    }

    state[MAX_TRACKS * static_cast<unsigned int>(eFeatures::_Size)] = (static_cast<float>(snaState.m_nbmissiles) / std::max(static_cast<float>(snaState.m_maxnbmissiles), 1e-6F));

    for (auto iang = 0; iang < nearestCollisionRanges.size();++iang)
        state[MAX_TRACKS * static_cast<unsigned int>(eFeatures::_Size) + iang + 1] = nearestCollisionRanges[iang];
}

void
TargetSelectorTrainingAgent::Init(void)
{
    Base::Init();

    if (!m_entity.has<StatesAndActionsForMission>())
        return;

    auto [stateActions] =
        m_entity.get<StatesAndActionsForMission>();

    stateActions.Init();
}

bool
TargetSelectorTrainingAgent::RewardFunction(void) noexcept
{
    entt::registry* registry{ m_entity.registry() };

    if (registry == nullptr
    || !registry->ctx().contains<ppo::model::sEnvironmentState>()
    || m_entity.uniqid() == entt::null
    || !m_entity.has<ppo::model::sEntityState,
        ppo::model::sFlightModelState,
        ppo::model::sSNAState>())
        return false;

    auto const [entityState, flightModelState, snaState] =
        m_entity.get<ppo::model::sEntityState, 
                     ppo::model::sFlightModelState, 
                     ppo::model::sSNAState>();

    auto const& envState{ registry->ctx().get<ppo::model::sEnvironmentState>() };

    float                   dt                  { envState.m_dt };
    float                   x                   { entityState.m_x };
    float                   y                   { entityState.m_y };
    float                   d                   { 0.F };

    float                   maxSpeed            { flightModelState.m_maxSpeed };
    float                   closingSpeed        { snaState.m_closingspeed };
    ppo::model::EntityId    targetId            { snaState.m_target };
    float const             vref                { 800.F };
    float                   fuelconsumption     { flightModelState.m_fuelconsumption };
    float const             fuelref             { ppo::model::C0 + ppo::model::CV3 * vref * vref * vref };
    float                   R_fuel              { std::clamp(-fuelconsumption / (fuelref * dt), -1.F, 0.F) };

    bool                    done                { false };
    bool                    targetDestroyed     { false };
    bool                    targetDestroyedByOwnship{ false };

    float                   R_close             { tanh(closingSpeed / 900.F) };
    float                   R_Intercept         { 0.F };
    float                   R_Range             { 0.F };
    float                   R_Danger            { -5.F };
    float                   R_NearDanger        { 0.F };
    float                   R_Guidance          { 0.F };
    float                   R_MissionFailure    { -300.F };
    float                   R_WrondDestruction  { -80.F };
    float                   R_GoodDestruction   { 80.F };
    float                   R_TargetDestruction { 400.F };
    float                   R_OwnshipDestruction{ -400.F }; 
    float                   R_OutOfArea         { 0.F };  

    std::array<float, static_cast<unsigned char>(StatesAndActionsForMission::RewardType::_Size)> rewards{};
    for (auto& r: rewards)
        r = 0.F;

    // Out of the Area
    float const outNorth{ envState.m_maxNorth/2.F - std::fabs(entityState.m_x) };
    float const outEast{ envState.m_maxEast/2.F - std::fabs(entityState.m_y) };
    if (envState.m_maxNorth < 2.F * entityState.m_x
    ||  2.F * entityState.m_x < -envState.m_maxNorth
    ||  envState.m_maxEast < 2.F * entityState.m_y
    ||  2.F * entityState.m_y < -envState.m_maxEast)
    {
        float nextOutRange{ std::max(envState.m_maxNorth - 2.F * std::fabs(entityState.m_x),
                                     envState.m_maxEast - 2.F * std::fabs(entityState.m_y)) };

        if (m_timeOutOfArea <= std::numeric_limits<float>::epsilon())
            m_prevOutRange = nextOutRange;
        
        m_timeOutOfArea += dt;

        if (m_timeOutOfArea >= 0.5F)
        {
            R_OutOfArea = std::min(0.F, 0.01F * (m_prevOutRange - nextOutRange)) ;

            rewards[static_cast<int>(StatesAndActionsForMission::RewardType::OutOfArea)] = R_OutOfArea;
        }

        m_prevOutRange = nextOutRange;
    }
    else
    {
        m_prevOutRange = 0.F;
        m_timeOutOfArea = 0.F;
    }

    ppo::model::BaseEntity targetEntity{ registry, targetId };
    if (targetEntity.has<ppo::model::sEntityState>())
    {
        auto const [targetState] = targetEntity.get<ppo::model::sEntityState>();

        float const tx{ targetState.m_x - x };
        float const ty{ targetState.m_y - y };
        d = static_cast<float>(sqrt(pow(tx, 2.F) + pow(ty, 2.F))) ;

        if (!m_memoryInitialized)
        {
            m_prevRange = d;
        }
        else if (!m_interceptReward && d < snaState.m_escapezone)
        {
            R_Intercept = 50.F;
            m_interceptReward = true;
        }

        R_Range = 0.01F*(m_prevRange - d);     

        m_prevRange = d;

        targetDestroyed = targetState.m_destroyed;
        targetDestroyedByOwnship = targetDestroyed && (targetState.m_destroyedById == entityState.m_entityId);
    }

    unsigned char nbEntitiesToBeDestroyed{ 0 };
    if (targetDestroyedByOwnship)
    {
        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::MainTargetDestruction)] =  R_TargetDestruction;
    }
    else if (entityState.m_destroyed)
    {
        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::OwnshipDestruction)] = R_OwnshipDestruction;
    }
    else
    {
        auto v = registry->view<ppo::model::sEntityState>();

        bool  inDanger = false;
        unsigned int nbDanger{ 0 };
        float minRangeFromDanger{ 1e9F };
        bool anyOpposingSAM{ false };

        for (auto entity : v)
        {
            ppo::model::BaseEntity otherEntity{ registry, entity };

            if (!otherEntity.has<ppo::model::sEntityState>()
            ||  entity  == m_entity.uniqid())
                continue;

            auto [otherEntityState] = otherEntity.get<ppo::model::sEntityState>();

            if (!otherEntityState.m_destroyed
            && otherEntityState.m_type != ppo::model::eEntityType::SAM
            && otherEntityState.m_side != ppo::model::eEntitySide::Unknown
            && otherEntityState.m_side != entityState.m_side)
                ++nbEntitiesToBeDestroyed;

            if (entity == snaState.m_target)
                continue;

            if (otherEntityState.m_destroyed
            && !otherEntityState.m_prevDestroyed
            &&  otherEntityState.m_destroyedById == m_entity.uniqid())
            {
                if (otherEntityState.m_side == entityState.m_side)
                    rewards[static_cast<int>(StatesAndActionsForMission::RewardType::WingmanDestruction)] = R_WrondDestruction;
                else if (otherEntityState.m_side != ppo::model::eEntitySide::Unknown)
                    rewards[static_cast<int>(StatesAndActionsForMission::RewardType::SecondaryTargetDestruction)] =  R_GoodDestruction;
                else
                    rewards[static_cast<int>(StatesAndActionsForMission::RewardType::CivilianDestruction)] = R_WrondDestruction;
            }

            if (otherEntityState.m_type == ppo::model::eEntityType::SAM
                && otherEntityState.m_side != entityState.m_side
                && !otherEntityState.m_destroyed)
            {
                auto [otherSNState] = otherEntity.get<ppo::model::sSNAState>();

                double const Rsec{ 1.2F * otherSNState.m_noescapezone };
                float dx{ otherEntityState.m_x - x };
                float dy{ otherEntityState.m_y - y };
                float dsam{ sqrt(dx * dx + dy * dy) };
                
                if (dsam <= Rsec)
                {
                    inDanger = true;
                }
                else if (dsam <= 1.5F * otherSNState.m_noescapezone)
                {
                    minRangeFromDanger = std::min(minRangeFromDanger, dsam);

                    anyOpposingSAM = true;
                }
            }
        }
        
        // Penalty reward for being in the danger zone of a SAM
        if (inDanger)
        {
            rewards[static_cast<int>(StatesAndActionsForMission::RewardType::Danger)] = R_Danger;
        }
        else if (anyOpposingSAM)
        {
            if (!m_firstDanger)
            {
                m_prevDangerRange = minRangeFromDanger;
                m_firstDanger = true;
            }

            R_NearDanger = std::min(0.F, 0.01F * (minRangeFromDanger - m_prevDangerRange));

            m_prevDangerRange = minRangeFromDanger;

            rewards[static_cast<int>(StatesAndActionsForMission::RewardType::NearDanger)] = R_NearDanger;
        }
        else
        {
            m_firstDanger = false;
            m_prevDangerRange = 0.F;
        }

        if (!m_memoryInitialized)
            m_prevDangerRange = minRangeFromDanger;

        // Bonus reward for intercepting the target (entering its escape zone)
        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::Intercept)] = R_Intercept;

        // potential based reward for range
        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::Range)] = R_Range;

        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::Fuel)] = R_fuel;
    }

    // Mission failure penalty
    if (IsLastStep() && !targetDestroyed)
        rewards[static_cast<int>(StatesAndActionsForMission::RewardType::MissionFailure)] = R_MissionFailure;

    for (auto i = 0; i < rewards.size(); ++i)
        AddReward(static_cast<StatesAndActionsForMission::RewardType>(i), rewards[i]);

    done = entityState.m_destroyed || targetDestroyed || (nbEntitiesToBeDestroyed == 0);

    m_memoryInitialized = true;

    if (done)
    {
        m_interceptReward = false;
        m_memoryInitialized = false;
        m_firstDanger = false;
        m_prevRange = 0.F;
        m_timeOutOfArea = 0.F;
    }

    SetDone(done);

    return done;
}
                           
void 
TargetSelectorProdAgent::Init()
{
    Base::Init();

    if (!m_entity.has<StatesAndActionsForMission>())
        return;

    auto [stateActions] =
        m_entity.get<StatesAndActionsForMission>();

    stateActions.Init();
}

}