#include "navigation.hpp"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace navigation
{
namespace
{
double distance(Point a, Point b)
{
    return std::hypot(a.x - b.x, a.y - b.y);
}
bool finite_point(Point p)
{
    return std::isfinite(p.x) && std::isfinite(p.y);
}
void positive(double value, double maximum)
{
    if (!std::isfinite(value) || value <= 0 || value > maximum)
        throw std::invalid_argument("scenario positive parameter out of range");
}
Scenario validate(Scenario s)
{
    positive(s.half_extent, 1e6);
    positive(s.dt, 10);
    positive(s.max_speed, 1e4);
    positive(s.max_acceleration, 1e4);
    positive(s.max_turn_rate, 100);
    if (s.max_steps == 0 || s.max_steps > 10000000 || s.mobiles.empty() || s.mobiles.size() > 4096 ||
        s.zones.size() > 1024)
        throw std::invalid_argument("invalid world size or episode limit");
    std::set<EntityId> ids, mobile_ids;
    std::set<std::string> names;
    for (const auto& mobile : s.mobiles)
    {
        if (mobile.id == no_entity || !ids.insert(mobile.id).second || !names.insert(mobile.name).second ||
            mobile.name.empty() || mobile.name.size() > 128 || mobile.name.find_first_of("\r\n") != std::string::npos ||
            !finite_point(mobile.initial.position) || std::abs(mobile.initial.position.x) > s.half_extent ||
            std::abs(mobile.initial.position.y) > s.half_extent || !std::isfinite(mobile.initial.heading) ||
            std::abs(mobile.initial.heading) > 1e6 || !std::isfinite(mobile.initial.speed) ||
            mobile.initial.speed < 0 || mobile.initial.speed > s.max_speed)
            throw std::invalid_argument("invalid or duplicate mobile definition");
        positive(mobile.arrival_radius, s.half_extent);
        const auto& e = mobile.energy;
        if (!std::isfinite(e.min_speed) || e.min_speed < 0 || e.min_speed > s.max_speed
            || !std::isfinite(e.capacity) || e.capacity < 0 || e.capacity > 1e12
            || !std::isfinite(e.idle) || e.idle < 0 || e.idle > 1e12
            || !std::isfinite(e.speed_cubic) || e.speed_cubic < 0 || e.speed_cubic > 1e12
            || !std::isfinite(e.turning) || e.turning < 0 || e.turning > 1e12
            || (e.capacity == 0 && (e.idle != 0 || e.speed_cubic != 0 || e.turning != 0)))
            throw std::invalid_argument("invalid mobile energy configuration");
        const auto& w = mobile.rewards;
        for (double value : {w.progress, w.step_cost, w.arrival, w.collision,
                             w.safety, w.boundary, w.fuel, w.failure})
            if (!std::isfinite(value) || value < 0 || value > 1e6)
                throw std::invalid_argument("invalid Navigation2D reward weight");
        mobile_ids.insert(mobile.id);
    }
    if (mobile_ids.count(s.controlled) == 0)
        throw std::invalid_argument("controlled mobile is missing");
    for (const auto& mobile : s.mobiles)
        if ((mobile.destination != no_entity &&
             (mobile.destination == mobile.id || mobile_ids.count(mobile.destination) == 0)) ||
            (mobile.id == s.controlled && mobile.destination == no_entity))
            throw std::invalid_argument("invalid mobile destination reference");
    for (const auto& zone : s.zones)
    {
        positive(zone.radius, s.half_extent);
        if (zone.id == no_entity || !ids.insert(zone.id).second || !names.insert(zone.name).second ||
            zone.name.empty() || zone.name.size() > 128 || zone.name.find_first_of("\r\n") != std::string::npos ||
            !finite_point(zone.center) || std::abs(zone.center.x) > s.half_extent ||
            std::abs(zone.center.y) > s.half_extent)
            throw std::invalid_argument("invalid or duplicate zone definition");
        for (const auto& mobile : s.mobiles)
            if (distance(mobile.initial.position, zone.center) <= zone.radius)
                throw std::invalid_argument("mobile starts in forbidden zone");
    }
    // Stable observation ordering independent of declaration or EnTT storage order.
    std::sort(s.mobiles.begin(), s.mobiles.end(),
              [](const MobileDefinition& a, const MobileDefinition& b) { return a.id < b.id; });
    std::sort(s.zones.begin(), s.zones.end(),
              [](const ZoneDefinition& a, const ZoneDefinition& b) { return a.id < b.id; });
    return s;
}
bool same_topology(const Scenario& a, const Scenario& b)
{
    if (a.controlled != b.controlled || a.mobiles.size() != b.mobiles.size()
        || a.zones.size() != b.zones.size()
        || a.half_extent != b.half_extent || a.dt != b.dt
        || a.max_steps != b.max_steps || a.max_speed != b.max_speed
        || a.max_acceleration != b.max_acceleration || a.max_turn_rate != b.max_turn_rate)
        return false;
    for (std::size_t i = 0; i < a.mobiles.size(); ++i)
        if (a.mobiles[i].id != b.mobiles[i].id
            || a.mobiles[i].name != b.mobiles[i].name
            || a.mobiles[i].destination != b.mobiles[i].destination
            || a.mobiles[i].arrival_radius != b.mobiles[i].arrival_radius
            || a.mobiles[i].energy.min_speed != b.mobiles[i].energy.min_speed
            || a.mobiles[i].energy.capacity != b.mobiles[i].energy.capacity
            || a.mobiles[i].energy.idle != b.mobiles[i].energy.idle
            || a.mobiles[i].energy.speed_cubic != b.mobiles[i].energy.speed_cubic
            || a.mobiles[i].energy.turning != b.mobiles[i].energy.turning
            || a.mobiles[i].rewards.progress != b.mobiles[i].rewards.progress
            || a.mobiles[i].rewards.step_cost != b.mobiles[i].rewards.step_cost
            || a.mobiles[i].rewards.arrival != b.mobiles[i].rewards.arrival
            || a.mobiles[i].rewards.collision != b.mobiles[i].rewards.collision
            || a.mobiles[i].rewards.safety != b.mobiles[i].rewards.safety
            || a.mobiles[i].rewards.boundary != b.mobiles[i].rewards.boundary
            || a.mobiles[i].rewards.fuel != b.mobiles[i].rewards.fuel
            || a.mobiles[i].rewards.failure != b.mobiles[i].rewards.failure)
            return false;
    for (std::size_t i = 0; i < a.zones.size(); ++i)
        if (a.zones[i].id != b.zones[i].id
            || a.zones[i].name != b.zones[i].name
            || a.zones[i].radius != b.zones[i].radius)
            return false;
    return true;
}
Scenario random_episode(const Scenario& baseline, std::mt19937_64& random)
{
    // Randomize the complete mobile population without altering world topology.
    // Rejection sampling keeps every mobile outside forbidden circles.
    std::uniform_real_distribution<double> location(-0.85 * baseline.half_extent,
                                                     0.85 * baseline.half_extent);
    std::uniform_real_distribution<double> heading(-3.14159265358979323846,
                                                    3.14159265358979323846);
    std::uniform_real_distribution<double> speed(0.0, baseline.max_speed);
    for (int attempt = 0; attempt < 128; ++attempt)
    {
        auto candidate = baseline;
        for (auto& mobile : candidate.mobiles)
        {
            mobile.initial.position = {location(random), location(random)};
            mobile.initial.heading = heading(random);
            mobile.initial.speed = speed(random);
        }
        bool collision = false;
        for (const auto& mobile : candidate.mobiles)
        {
            for (const auto& zone : candidate.zones)
                if (distance(mobile.initial.position, zone.center) <= zone.radius)
                    collision = true;
            for (const auto& destination : candidate.mobiles)
                if (mobile.destination == destination.id
                    && distance(mobile.initial.position, destination.initial.position)
                        <= mobile.arrival_radius)
                    collision = true;
        }
        if (!collision) return candidate;
    }
    throw std::runtime_error("failed to sample valid Navigation2D initial states");
}
bool energy_enabled(const Scenario& s)
{
    return std::any_of(s.mobiles.begin(), s.mobiles.end(), [](const MobileDefinition& mobile)
    {
        return mobile.energy.capacity > 0 || mobile.energy.min_speed > 0;
    });
}
ppo::rl::Space observation_space_for(const Scenario& s)
{
    const std::size_t size = 10 + (energy_enabled(s) ? 1 : 0)
        + 3 * s.zones.size() + kLidarBinCount;
    const float infinity = std::numeric_limits<float>::infinity();
    return ppo::rl::Box({size}, std::vector<float>(size, -infinity), std::vector<float>(size, infinity));
}
// First entry on a swept segment. Infinity denotes no intersection.
double entry(Point a, Point b, Point center, double radius)
{
    const double x = a.x - center.x, y = a.y - center.y, dx = b.x - a.x, dy = b.y - a.y;
    const double c = x * x + y * y - radius * radius;
    if (c <= 0)
        return 0;
    const double length = dx * dx + dy * dy;
    if (length == 0)
        return std::numeric_limits<double>::infinity();
    const double projection = x * dx + y * dy, discriminant = projection * projection - length * c;
    if (discriminant < 0)
        return std::numeric_limits<double>::infinity();
    const double time = (-projection - std::sqrt(discriminant)) / length;
    return time >= 0 && time <= 1 ? time : std::numeric_limits<double>::infinity();
}
double collision_time(Point from, Point to, const entt::registry& world, double limit)
{
    double time = std::numeric_limits<double>::infinity();
    for (const auto entity : world.view<const State, const ForbiddenZone>())
        time =
            std::min(time, entry(from, to, world.get<State>(entity).position, world.get<ForbiddenZone>(entity).radius));
    if (to.x > limit)
        time = std::min(time, (limit - from.x) / (to.x - from.x));
    if (to.x < -limit)
        time = std::min(time, (-limit - from.x) / (to.x - from.x));
    if (to.y > limit)
        time = std::min(time, (limit - from.y) / (to.y - from.y));
    if (to.y < -limit)
        time = std::min(time, (-limit - from.y) / (to.y - from.y));
    return time;
}
State advance_motion(State state, Commands commands, const Scenario& scenario)
{
    constexpr double two_pi = 6.28318530717958647692;
    state.heading = std::remainder(state.heading + commands.turn * scenario.max_turn_rate * scenario.dt, two_pi);
    state.speed = std::clamp(state.speed + commands.acceleration * scenario.max_acceleration * scenario.dt, 0.0,
                             scenario.max_speed);
    state.position.x += std::cos(state.heading) * state.speed * scenario.dt;
    state.position.y += std::sin(state.heading) * state.speed * scenario.dt;
    return state;
}
Commands commands_from(const ppo::rl::Space& space, const ppo::rl::Value& value)
{
    if (!ppo::rl::contains(space, value))
        throw std::invalid_argument("invalid navigation action");
    const auto& commands = std::get<std::vector<float>>(value);
    return {commands[0], commands[1]};
}
EntityId read_id(std::istream& input)
{
    std::string token;
    input >> token;
    EntityId id = no_entity;
    const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
    if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size())
        input.setstate(std::ios::failbit);
    return id;
}
struct PendingMotion
{
    State before;
    State after;
    double collision;
    double arrival;
    double energy_event = std::numeric_limits<double>::infinity();
    double fuel_burn = 0;
    bool speed_stall = false;
};
} // namespace
Scenario load_scenario(const std::filesystem::path& path)
{
    std::ifstream input(path);
    std::string line;
    if (!std::getline(input, line) || line != "navigation2d-v3")
        throw std::invalid_argument("invalid scenario header (expected navigation2d-v3)");
    Scenario s;
    s.mobiles.clear();
    std::set<std::string> singleton_keys;
    std::map<EntityId, EnergyModel> energy_config;
    while (std::getline(input, line))
    {
        if (line.empty() || line.front() == '#')
            continue;
        std::istringstream row(line);
        row.imbue(std::locale::classic());
        std::string key;
        row >> key;
        if (key == "mobile")
        {
            MobileDefinition mobile{};
            mobile.id = read_id(row);
            row >> std::quoted(mobile.name) >> mobile.initial.position.x >> mobile.initial.position.y >>
                mobile.initial.heading >> mobile.initial.speed;
            mobile.destination = read_id(row);
            row >> mobile.arrival_radius;
            s.mobiles.push_back(std::move(mobile));
        }
        else if (key == "energy")
        {
            const auto id = read_id(row);
            EnergyModel model;
            row >> model.min_speed >> model.capacity >> model.idle
                >> model.speed_cubic >> model.turning;
            if (id == no_entity || !energy_config.emplace(id, model).second)
                throw std::invalid_argument("duplicate or invalid energy model ID");
        }
        else if (key == "zone")
        {
            ZoneDefinition zone{};
            zone.id = read_id(row);
            row >> std::quoted(zone.name) >> zone.center.x >> zone.center.y >> zone.radius;
            s.zones.push_back(std::move(zone));
        }
        else
        {
            if (!singleton_keys.insert(key).second)
                throw std::invalid_argument("duplicate scenario field: " + key);
            if (key == "primary")
                s.controlled = read_id(row);
            else if (key == "world")
                row >> s.half_extent >> s.dt >> s.max_steps;
            else if (key == "motion")
                row >> s.max_speed >> s.max_acceleration >> s.max_turn_rate;
            else
                throw std::invalid_argument("unknown scenario field: " + key);
        }
        if (!row || s.mobiles.size() > 4096 || s.zones.size() > 1024)
            throw std::invalid_argument("invalid scenario value: " + key);
        std::string extra;
        if (row >> extra)
            throw std::invalid_argument("trailing scenario field: " + key);
    }
    if (input.bad())
        throw std::runtime_error("scenario read failed");
    for (auto& mobile : s.mobiles)
    {
        const auto it = energy_config.find(mobile.id);
        if (it != energy_config.end())
            mobile.energy = it->second;
    }
    for (const auto& configured : energy_config)
        if (std::none_of(s.mobiles.begin(), s.mobiles.end(), [&](const MobileDefinition& mobile)
                         { return mobile.id == configured.first; }))
            throw std::invalid_argument("energy model refers to unknown mobile");
    return validate(std::move(s));
}
std::string scenario_text(const Scenario& scenario)
{
    const auto s = validate(scenario);
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10) << "navigation2d-v3\nprimary "
           << s.controlled << "\nworld " << s.half_extent << ' ' << s.dt << ' ' << s.max_steps << "\nmotion "
           << s.max_speed << ' ' << s.max_acceleration << ' ' << s.max_turn_rate << '\n';
    for (const auto& m : s.mobiles)
        output << "mobile " << m.id << ' ' << std::quoted(m.name) << ' ' << m.initial.position.x << ' '
               << m.initial.position.y << ' ' << m.initial.heading << ' ' << m.initial.speed << ' ' << m.destination
               << ' ' << m.arrival_radius << '\n';
    for (const auto& mobile : s.mobiles)
    {
        const auto& e = mobile.energy;
        if (e.min_speed != 0 || e.capacity != 0 || e.idle != 0
            || e.speed_cubic != 0 || e.turning != 0)
            output << "energy " << mobile.id << ' ' << e.min_speed << ' ' << e.capacity
                   << ' ' << e.idle << ' ' << e.speed_cubic << ' ' << e.turning << '\n';
    }
    for (const auto& z : s.zones)
        output << "zone " << z.id << ' ' << std::quoted(z.name) << ' ' << z.center.x << ' ' << z.center.y << ' '
               << z.radius << '\n';
    return output.str();
}
void save_scenario(const Scenario& scenario, const std::filesystem::path& path)
{
    const auto text = scenario_text(scenario);
    std::ofstream output(path);
    output << text;
    output.close();
    if (!output)
        throw std::runtime_error("cannot save scenario");
}
std::vector<Scenario> load_episode_cases(const std::filesystem::path& manifest)
{
    std::ifstream input(manifest);
    if (!input) throw std::runtime_error("cannot open episode-case manifest");
    std::vector<Scenario> cases;
    std::string line;
    while (std::getline(input, line))
    {
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos || line[first] == '#') continue;
        const auto last = line.find_last_not_of(" \t\r");
        const auto name = line.substr(first, last - first + 1);
        if (cases.size() >= 10000)
            throw std::invalid_argument("too many episode cases");
        cases.push_back(load_scenario(manifest.parent_path() / name));
    }
    if (input.bad()) throw std::runtime_error("episode-case manifest read failed");
    if (cases.empty()) throw std::invalid_argument("episode-case manifest is empty");
    return cases;
}
Environment::Environment(Scenario scenario)
    : scenario_(validate(std::move(scenario))), observations_(observation_space_for(scenario_))
{
    rebuild(scenario_);
}
void Environment::rebuild(Scenario scenario)
{
    auto next = validate(std::move(scenario));
    entt::registry world;
    std::map<EntityId, entt::entity> handles;
    std::vector<EntityId> entities;
    for (const auto& m : next.mobiles)
    {
        const auto entity = world.create();
        handles.emplace(m.id, entity);
        entities.push_back(m.id);
        world.emplace<Identity>(entity, m.id, m.name);
        world.emplace<Mobile>(entity);
        world.emplace<State>(entity, m.initial);
        world.emplace<InitialState>(entity, m.initial);
        world.emplace<Destination>(entity, m.destination, m.arrival_radius);
        world.emplace<Commands>(entity);
        world.emplace<EnergyModel>(entity, m.energy);
        world.emplace<RewardWeights>(entity, m.rewards);
        world.emplace<EnergyState>(entity, m.energy.capacity, 0.0, false, false);
    }
    for (const auto& z : next.zones)
    {
        const auto entity = world.create();
        handles.emplace(z.id, entity);
        entities.push_back(z.id);
        world.emplace<Identity>(entity, z.id, z.name);
        world.emplace<State>(entity, State{z.center, 0, 0});
        world.emplace<ForbiddenZone>(entity, z.radius);
    }
    auto space = observation_space_for(next);
    world_ = std::move(world);
    handles_ = std::move(handles);
    entities_ = std::move(entities);
    scenario_ = std::move(next);
    observations_ = std::move(space);
    baseline_scenario_ = scenario_;
    episode_cases_.clear();
    initialization_mode_ = InitializationMode::Fixed;
    episode_index_ = 0;
    steps_ = 0;
    active_ = false;
    transitions_.clear();
}
void Environment::require_editable() const
{
    if (active_ && steps_ != 0)
        throw std::logic_error("finish or stop the world episode before editing");
}
void Environment::create_mobile(MobileDefinition mobile)
{
    require_editable();
    if (!controllers_.empty())
        throw std::logic_error("detach policies before editing topology");
    auto next = scenario_;
    next.mobiles.push_back(std::move(mobile));
    rebuild(std::move(next));
}
void Environment::create_zone(ZoneDefinition zone)
{
    require_editable();
    if (!controllers_.empty())
        throw std::logic_error("detach policies before editing topology");
    auto next = scenario_;
    next.zones.push_back(std::move(zone));
    rebuild(std::move(next));
}
void Environment::remove_entity(EntityId id)
{
    require_editable();
    if (!controllers_.empty())
        throw std::logic_error("detach policies before editing topology");
    handle(id);
    auto next = scenario_;
    next.mobiles.erase(std::remove_if(next.mobiles.begin(), next.mobiles.end(),
                                      [id](const MobileDefinition& m) { return m.id == id; }),
                       next.mobiles.end());
    next.zones.erase(
        std::remove_if(next.zones.begin(), next.zones.end(), [id](const ZoneDefinition& z) { return z.id == id; }),
        next.zones.end());
    rebuild(std::move(next)); // validation rejects deleting a referenced destination or primary
}
RewardWeights Environment::reward_weights(EntityId mobile) const
{
    const auto entity = handle(mobile);
    if (!world_.all_of<Mobile>(entity))
        throw std::invalid_argument("reward configuration requires mobile");
    return world_.get<RewardWeights>(entity);
}
void Environment::set_reward_weights(EntityId mobile, RewardWeights weights)
{
    const auto entity = handle(mobile);
    if (!world_.all_of<Mobile>(entity))
        throw std::invalid_argument("reward configuration requires mobile");
    for (double value : {weights.progress, weights.step_cost, weights.arrival,
                         weights.collision, weights.safety, weights.boundary,
                         weights.fuel, weights.failure})
        if (!std::isfinite(value) || value < 0 || value > 1e6)
            throw std::invalid_argument("invalid Navigation2D reward weight");
    world_.get<RewardWeights>(entity) = weights;
    const auto assign = [mobile, weights](Scenario& scenario)
    {
        for (auto& definition : scenario.mobiles)
            if (definition.id == mobile) definition.rewards = weights;
    };
    assign(scenario_);
    assign(baseline_scenario_);
    for (auto& episode : episode_cases_) assign(episode);
}
void Environment::configure_initialization(InitializationMode mode, std::vector<Scenario> cases)
{
    require_editable();
    if (mode != InitializationMode::Fixed
        && mode != InitializationMode::Randomized
        && mode != InitializationMode::PresetCases)
        throw std::invalid_argument("invalid Navigation2D initialization mode");
    if (mode == InitializationMode::PresetCases)
    {
        if (cases.empty()) throw std::invalid_argument("preset mode needs at least one case");
        for (auto& episode : cases)
        {
            episode = validate(std::move(episode));
            if (!same_topology(baseline_scenario_, episode))
                throw std::invalid_argument("episode case has different topology or motion rules");
        }
    }
    else if (!cases.empty())
        throw std::invalid_argument("cases are only valid in preset initialization mode");
    episode_cases_ = std::move(cases);
    initialization_mode_ = mode;
    episode_index_ = 0;
}
InitializationMode Environment::initialization_mode() const noexcept
{
    return initialization_mode_;
}
const std::vector<Scenario>& Environment::episode_cases() const noexcept
{
    return episode_cases_;
}
ppo::rl::ResetResult Environment::reset(const ppo::rl::ResetOptions& options)
{
    if (!options.options.empty())
        throw std::invalid_argument("no custom reset options supported");
    if (options.seed && initialization_mode_ == InitializationMode::Randomized)
        random_engine_.seed(*options.seed);
    Scenario chosen = baseline_scenario_;
    if (initialization_mode_ == InitializationMode::Randomized)
        chosen = random_episode(baseline_scenario_, random_engine_);
    else if (initialization_mode_ == InitializationMode::PresetCases)
    {
        const auto index = (options.seed ? *options.seed : episode_index_) % episode_cases_.size();
        chosen = episode_cases_.at(static_cast<std::size_t>(index));
    }
    scenario_ = std::move(chosen);
    for (const auto& mobile : scenario_.mobiles)
    {
        const auto entity = handle(mobile.id);
        world_.get<State>(entity) = mobile.initial;
        world_.get<Commands>(entity) = {};
        world_.get<EnergyState>(entity) = {mobile.energy.capacity, 0.0, false, false};
        world_.get<RewardWeights>(entity) = mobile.rewards;
    }
    for (const auto& zone : scenario_.zones)
        world_.get<State>(handle(zone.id)).position = zone.center;
    ++episode_index_;
    steps_ = 0;
    active_ = true;
    stop_requested_ = false;
    transitions_.clear();
    return {observation(scenario_.controlled), {{"environment", std::string("Navigation2D-v3")}}};
}
void Environment::set_policy(EntityId mobile, std::unique_ptr<ppo::rl::IPolicy> policy, bool learning)
{
    require_editable();
    const auto entity = handle(mobile);
    if (!world_.all_of<Mobile>(entity) || mobile == scenario_.controlled)
        throw std::invalid_argument("internal controller requires a non-primary mobile");
    auto* learner = learning ? dynamic_cast<ppo::rl::ILearner*>(policy.get()) : nullptr;
    if ((learning && !learner) || (policy && world_.get<Destination>(entity).entity == no_entity))
        throw std::invalid_argument("controller requires a destination and a matching learner");
    if (policy)
        controllers_.insert_or_assign(mobile, Controller{std::move(policy), learner});
    else
        controllers_.erase(mobile);
}
void Environment::set_goal_policy(std::unique_ptr<ppo::rl::IPolicy> policy)
{
    set_policy(world_.get<Destination>(handle(scenario_.controlled)).entity, std::move(policy));
}
void Environment::set_manual_action(EntityId mobile, Commands commands)
{
    const auto entity = handle(mobile);
    if (!world_.all_of<Mobile>(entity) || mobile == scenario_.controlled || controllers_.count(mobile))
        throw std::invalid_argument("manual action requires an uncontrolled non-primary mobile");
    commands_from(actions_, std::vector<float>{commands.turn, commands.acceleration});
    world_.get<Commands>(entity) = commands;
}
ppo::rl::Value Environment::observation(EntityId mobile) const
{
    const auto entity = handle(mobile);
    if (!world_.all_of<Mobile>(entity))
        throw std::invalid_argument("observation requires mobile");
    const auto own = state(mobile);
    const auto destination = world_.get<Destination>(entity).entity;
    const auto other = destination == no_entity ? own : state(destination);
    std::vector<float> values{static_cast<float>(own.position.x / scenario_.half_extent),
                              static_cast<float>(own.position.y / scenario_.half_extent),
                              static_cast<float>(other.position.x / scenario_.half_extent),
                              static_cast<float>(other.position.y / scenario_.half_extent),
                              static_cast<float>(std::cos(own.heading)),
                              static_cast<float>(std::sin(own.heading)),
                              static_cast<float>(own.speed / scenario_.max_speed),
                              static_cast<float>(steps_) / static_cast<float>(scenario_.max_steps),
                              static_cast<float>(std::cos(other.heading) * other.speed / scenario_.max_speed),
                              static_cast<float>(std::sin(other.heading) * other.speed / scenario_.max_speed)};
    values.reserve(10 + (energy_enabled(scenario_) ? 1 : 0)
                   + 3 * scenario_.zones.size() + kLidarBinCount);
    if (energy_enabled(scenario_))
    {
        const auto& fuel = world_.get<EnergyState>(entity);
        const auto& model = world_.get<EnergyModel>(entity);
        values.push_back(model.capacity > 0
            ? static_cast<float>(fuel.remaining / model.capacity) : 1.F);
    }
    for (const auto& zone : scenario_.zones)
    {
        const auto handle_ = handle(zone.id);
        const auto center = world_.get<State>(handle_).position;
        values.push_back(static_cast<float>(center.x / scenario_.half_extent));
        values.push_back(static_cast<float>(center.y / scenario_.half_extent));
        values.push_back(static_cast<float>(world_.get<ForbiddenZone>(handle_).radius / scenario_.half_extent));
    }
    const auto scan = lidar(mobile);
    values.insert(values.end(), scan.normalized.begin(), scan.normalized.end());
    return values;
}
LidarScan Environment::lidar(EntityId mobile) const
{
    if (!world_.all_of<Mobile>(handle(mobile)))
        throw std::invalid_argument("LiDAR requires mobile");
    return scan_lidar(state(mobile), scenario_.half_extent, scenario_.zones);
}
ppo::rl::StepResult Environment::step(const ppo::rl::Value& action)
{
    if (!active_)
        throw std::logic_error("Navigation2D reset required");
    std::map<EntityId, Commands> commands;
    commands.emplace(scenario_.controlled, commands_from(actions_, action));
    // Collect all decisions first. Invalid actions leave every physical state unchanged.
    for (const auto& mobile : scenario_.mobiles)
    {
        if (mobile.id == scenario_.controlled)
            continue;
        const auto found = controllers_.find(mobile.id);
        commands.emplace(mobile.id, found == controllers_.end()
                                        ? world_.get<Commands>(handle(mobile.id))
                                        : commands_from(actions_, found->second.policy->act(observation(mobile.id))));
    }
    std::map<EntityId, PendingMotion> motions;
    double fraction = 1;
    bool world_event = false;
    for (const auto& mobile : scenario_.mobiles)
    {
        const auto before = state(mobile.id);
        const auto after = advance_motion(before, commands.at(mobile.id), scenario_);
        const double collision = collision_time(before.position, after.position, world_, scenario_.half_extent);
        const auto& model = world_.get<EnergyModel>(handle(mobile.id));
        const auto& fuel = world_.get<EnergyState>(handle(mobile.id));
        const double turn_rate = commands.at(mobile.id).turn * scenario_.max_turn_rate;
        const double fuel_burn = model.capacity > 0
            ? (model.idle + model.speed_cubic * after.speed * after.speed * after.speed
               + model.turning * turn_rate * turn_rate) * scenario_.dt : 0.0;
        const double exhaustion_time = fuel_burn > 0 && fuel.remaining <= fuel_burn
            ? fuel.remaining / fuel_burn : std::numeric_limits<double>::infinity();
        const bool speed_stall = model.min_speed > 0 && 2.0 * after.speed < model.min_speed;
        motions.emplace(mobile.id, PendingMotion{before, after, collision,
            std::numeric_limits<double>::infinity(), exhaustion_time, fuel_burn, speed_stall});
        if (collision <= 1 || exhaustion_time <= 1 || speed_stall)
        {
            fraction = std::min(fraction, std::min(collision, exhaustion_time));
            world_event = true;
        }
    }
    for (const auto& mobile : scenario_.mobiles)
    {
        if (mobile.destination == no_entity)
            continue;
        auto& motion = motions.at(mobile.id);
        const auto& other = motions.at(mobile.destination);
        const Point relative_next{motion.after.position.x - (other.after.position.x - other.before.position.x),
                                  motion.after.position.y - (other.after.position.y - other.before.position.y)};
        motion.arrival = entry(motion.before.position, relative_next, other.before.position, mobile.arrival_radius);
        if (motion.arrival <= 1)
        {
            fraction = std::min(fraction, motion.arrival);
            world_event = true;
        }
    }
    for (const auto& mobile : scenario_.mobiles)
    {
        const auto& motion = motions.at(mobile.id);
        auto after = motion.after;
        after.position = {motion.before.position.x + fraction * (after.position.x - motion.before.position.x),
                          motion.before.position.y + fraction * (after.position.y - motion.before.position.y)};
        world_.get<State>(handle(mobile.id)) = after;
        auto& fuel = world_.get<EnergyState>(handle(mobile.id));
        fuel.last_burn = std::min(fuel.remaining, motion.fuel_burn * fraction);
        fuel.remaining = std::max(0.0, fuel.remaining - fuel.last_burn);
        fuel.exhausted = motion.energy_event <= fraction;
        fuel.stalled = motion.speed_stall && fraction >= 1.0;
    }
    ++steps_;
    active_ = !(world_event || steps_ == scenario_.max_steps || stop_requested_);
    transitions_.clear();
    for (const auto& mobile : scenario_.mobiles)
    {
        const auto& motion = motions.at(mobile.id);
        const bool blocked = motion.collision <= fraction;
        const auto& own_fuel = world_.get<EnergyState>(handle(mobile.id));
        const bool energy_failure = own_fuel.exhausted || own_fuel.stalled;
        const bool goal_blocked = mobile.destination != no_entity
            && (motions.at(mobile.destination).collision <= fraction
                || world_.get<EnergyState>(handle(mobile.destination)).exhausted
                || world_.get<EnergyState>(handle(mobile.destination)).stalled);
        const bool success = motion.arrival <= fraction && !blocked && !goal_blocked && !energy_failure;
        const bool terminated = success || blocked || goal_blocked || energy_failure;
        const bool truncated = steps_ == scenario_.max_steps || stop_requested_ || (world_event && !terminated);
        const double previous = mobile.destination == no_entity
                                    ? 0
                                    : distance(motion.before.position, motions.at(mobile.destination).before.position);
        const double remaining = mobile.destination == no_entity
                                     ? 0
                                     : distance(state(mobile.id).position, state(mobile.destination).position);
        // The LiDAR describes geometry; this dense safety term teaches the policy
        // to turn away *before* entering an obstacle, including the map boundary.
        const auto& after_state = state(mobile.id);
        const double heading = after_state.heading;
        double nearest_margin = std::numeric_limits<double>::infinity();
        double nearest_scale = 1.0;
        double nearest_alignment = 0.0;
        for (const auto& zone : scenario_.zones)
        {
            const double dx = zone.center.x - after_state.position.x;
            const double dy = zone.center.y - after_state.position.y;
            const double distance_to_center = std::hypot(dx, dy);
            const double margin = distance_to_center - zone.radius;
            if (margin < nearest_margin)
            {
                nearest_margin = margin;
                nearest_scale = std::max(zone.radius, 1.0);
                nearest_alignment = distance_to_center > 1e-9
                    ? (dx * std::cos(heading) + dy * std::sin(heading)) / distance_to_center : 1.0;
            }
        }
        const double boundary_margin = scenario_.half_extent
            - std::max(std::abs(after_state.position.x), std::abs(after_state.position.y));
        const double caution_distance = std::max(2.0 * scenario_.max_speed * scenario_.dt, nearest_scale * 0.25);
        const double proximity = std::clamp(1.0 - nearest_margin / caution_distance, 0.0, 1.0);
        const double approaching = std::max(0.0, nearest_alignment);
        const double safety_penalty = -0.05 * proximity * (1.0 + approaching);
        const double boundary_proximity = std::clamp(1.0 - boundary_margin / caution_distance, 0.0, 1.0);
        const double boundary_penalty = -0.05 * boundary_proximity;
        const double reward =
            (previous - remaining) / scenario_.half_extent - 0.001
            + (success ? 1.0 : 0.0) - (blocked ? 1.0 : 0.0)
            + safety_penalty + boundary_penalty
            - (energy_failure ? 1.0 : 0.0)
            - (scenario_.mobiles.empty() || world_.get<EnergyModel>(handle(mobile.id)).capacity == 0
                ? 0.0 : 0.01 * own_fuel.last_burn
                    / world_.get<EnergyModel>(handle(mobile.id)).capacity);
        transitions_.emplace(mobile.id, ppo::rl::StepResult{observation(mobile.id),
                                                            reward,
                                                            terminated,
                                                            truncated,
                                                            {{"success", success},
                                                             {"collision", blocked},
                                                             {"goal_collision", goal_blocked},
                                                             {"fuel_exhausted", own_fuel.exhausted},
                                                             {"speed_stall", own_fuel.stalled},
                                                             {"fuel_remaining", own_fuel.remaining},
                                                             {"distance", remaining}}});
    }
    // Complete all transitions before any optimiser can change a controller's weights.
    try
    {
        for (auto& item : controllers_)
            if (item.second.learner)
                item.second.learner->observe(transitions_.at(item.first));
    }
    catch (...)
    {
        active_ = false;
        throw;
    } // no retry of an already advanced world
    return transitions_.at(scenario_.controlled);
}
void Environment::request_stop() noexcept
{
    stop_requested_ = true;
}
const ppo::rl::StepResult& Environment::transition(EntityId mobile) const
{
    return transitions_.at(mobile);
}
const entt::registry& Environment::registry() const noexcept
{
    return world_;
}
const std::vector<EntityId>& Environment::entities() const noexcept
{
    return entities_;
}
entt::entity Environment::handle(EntityId id) const
{
    return handles_.at(id);
}
State Environment::state(EntityId id) const
{
    return world_.get<State>(handle(id));
}
State Environment::state() const
{
    return state(scenario_.controlled);
}
State Environment::goal_state() const
{
    return state(world_.get<Destination>(handle(scenario_.controlled)).entity);
}
Point Environment::position() const
{
    return state().position;
}
const ppo::rl::Space& Environment::observation_space() const noexcept
{
    return observations_;
}
const ppo::rl::Space& Environment::action_space() const noexcept
{
    return actions_;
}
const Scenario& Environment::baseline_scenario() const noexcept
{
    return baseline_scenario_;
}
const Scenario& Environment::scenario() const noexcept
{
    return scenario_;
}
} // namespace navigation
