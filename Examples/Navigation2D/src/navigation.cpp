#include "navigation.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace navigation {
namespace {
double distance(Point a, Point b) { return std::hypot(a.x-b.x,a.y-b.y); }
bool finite_point(Point point) { return std::isfinite(point.x) && std::isfinite(point.y); }
void positive(double value, double maximum)
{
    if (!std::isfinite(value) || value <= 0 || value > maximum)
        throw std::invalid_argument("scenario positive parameter out of range");
}
Scenario validate(Scenario s)
{
    positive(s.half_extent,1e6); positive(s.arrival_radius,s.half_extent);
    positive(s.dt,10); positive(s.max_speed,1e4); positive(s.max_acceleration,1e4);
    positive(s.max_turn_rate,100);
    if (!finite_point(s.start) || !finite_point(s.goal) || !finite_point(s.goal_velocity)
        || std::abs(s.start.x)>s.half_extent || std::abs(s.start.y)>s.half_extent
        || std::abs(s.goal.x)+s.arrival_radius>s.half_extent
        || std::abs(s.goal.y)+s.arrival_radius>s.half_extent
        || !std::isfinite(s.heading) || std::abs(s.heading)>1e6
        || !std::isfinite(s.speed) || s.speed<0 || s.speed>s.max_speed
        || s.max_steps==0 || s.max_steps>10000000 || s.obstacles.size()>1024)
        throw std::invalid_argument("invalid navigation scenario");
    if (std::abs(s.goal_velocity.x / s.max_speed) > std::numeric_limits<float>::max()
        || std::abs(s.goal_velocity.y / s.max_speed) > std::numeric_limits<float>::max())
        throw std::invalid_argument("goal velocity cannot be represented in the observation");
    const double duration = s.dt * static_cast<double>(s.max_steps);
    const Point final_goal{s.goal.x + duration * s.goal_velocity.x,
                           s.goal.y + duration * s.goal_velocity.y};
    if (!finite_point(final_goal)
        || std::abs(final_goal.x) + s.arrival_radius > s.half_extent
        || std::abs(final_goal.y) + s.arrival_radius > s.half_extent)
        throw std::invalid_argument("goal mobile trajectory leaves the world during the episode");
    for (const auto& obstacle:s.obstacles)
    {
        positive(obstacle.radius,s.half_extent);
        if (!finite_point(obstacle.center)
            || std::abs(obstacle.center.x)>s.half_extent || std::abs(obstacle.center.y)>s.half_extent
            || distance(s.start,obstacle.center)<=obstacle.radius
            || distance(s.goal,obstacle.center)<=obstacle.radius+s.arrival_radius)
            throw std::invalid_argument("invalid or overlapping obstacle");
    }
    return s;
}
ppo::rl::Space observation_space_for(const Scenario& scenario)
{
    const std::size_t size=10+3*scenario.obstacles.size();
    const float infinity=std::numeric_limits<float>::infinity();
    return ppo::rl::Box({size},std::vector<float>(size,-infinity),std::vector<float>(size,infinity));
}
// First entry on segment [a,b]. Infinity denotes no intersection.
double entry(Point a,Point b,Point center,double radius)
{
    const double x=a.x-center.x, y=a.y-center.y;
    const double dx=b.x-a.x,dy=b.y-a.y;
    const double c=x*x+y*y-radius*radius;
    if(c<=0) return 0;
    const double length=dx*dx+dy*dy;
    if(length==0) return std::numeric_limits<double>::infinity();
    const double projection=x*dx+y*dy;
    const double discriminant=projection*projection-length*c;
    if(discriminant<0) return std::numeric_limits<double>::infinity();
    const double time=(-projection-std::sqrt(discriminant))/length;
    return time>=0 && time<=1 ? time : std::numeric_limits<double>::infinity();
}
}
Scenario load_scenario(const std::filesystem::path& path)
{
    std::ifstream input(path);
    std::string line;
    if(!std::getline(input,line) || line!="navigation2d-v2") throw std::invalid_argument("invalid scenario header");
    Scenario s;
    while(std::getline(input,line))
    {
        if(line.empty() || line.front()=='#') continue;
        std::istringstream row(line);
        row.imbue(std::locale::classic());
        std::string key; row>>key;
        if(key=="start") row>>s.start.x>>s.start.y>>s.heading>>s.speed;
        else if(key=="goal") row>>s.goal.x>>s.goal.y>>s.arrival_radius>>s.goal_velocity.x>>s.goal_velocity.y;
        else if(key=="world") row>>s.half_extent>>s.dt>>s.max_steps;
        else if(key=="motion") row>>s.max_speed>>s.max_acceleration>>s.max_turn_rate;
        else if(key=="obstacle")
        {
            Obstacle obstacle;
            row>>obstacle.center.x>>obstacle.center.y>>obstacle.radius;
            s.obstacles.push_back(obstacle);
            if(s.obstacles.size()>1024) throw std::invalid_argument("too many obstacles");
        }
        else throw std::invalid_argument("unknown scenario field: "+key);
        if(!row) throw std::invalid_argument("invalid scenario value: "+key);
        std::string extra;
        if(row>>extra) throw std::invalid_argument("trailing scenario field: "+key);
    }
    if(input.bad()) throw std::runtime_error("scenario read failed");
    return validate(std::move(s));
}
std::string scenario_text(const Scenario& scenario)
{
    const auto s = validate(scenario);
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << "navigation2d-v2\nstart " << s.start.x << ' ' << s.start.y << ' '
           << s.heading << ' ' << s.speed << "\ngoal " << s.goal.x << ' '
           << s.goal.y << ' ' << s.arrival_radius << ' ' << s.goal_velocity.x << ' ' << s.goal_velocity.y << "\nworld " << s.half_extent
           << ' ' << s.dt << ' ' << s.max_steps << "\nmotion " << s.max_speed
           << ' ' << s.max_acceleration << ' ' << s.max_turn_rate << '\n';
    for (const auto& obstacle : s.obstacles)
        output << "obstacle " << obstacle.center.x << ' ' << obstacle.center.y << ' ' << obstacle.radius << '\n';
    return output.str();
}
void save_scenario(const Scenario& scenario, const std::filesystem::path& path)
{
    const auto text = scenario_text(scenario);
    std::ofstream output(path);
    output << text;
    output.close();
    if (!output) throw std::runtime_error("cannot save scenario");
}
Environment::Environment(Scenario scenario)
    :scenario_(validate(std::move(scenario))),observations_(observation_space_for(scenario_)) {}
ppo::rl::ResetResult Environment::reset(const ppo::rl::ResetOptions& options)
{
    if(!options.options.empty()) throw std::invalid_argument("no reset options supported by Navigation2D v2");
    // Deterministic scenario: all seeds give the same initial state; no global RNG touched.
    goal_position_ = scenario_.goal;
    position_=scenario_.start; heading_=scenario_.heading; speed_=scenario_.speed; steps_=0; active_=true;
    return {observe(),{{"environment",std::string("Navigation2D-v2")}}};
}
ppo::rl::Value Environment::observe() const
{
    std::vector<float> values{static_cast<float>(position_.x/scenario_.half_extent),
        static_cast<float>(position_.y/scenario_.half_extent),
        static_cast<float>(goal_position_.x/scenario_.half_extent),
        static_cast<float>(goal_position_.y/scenario_.half_extent),
        static_cast<float>(std::cos(heading_)),static_cast<float>(std::sin(heading_)),
        static_cast<float>(speed_/scenario_.max_speed),
        static_cast<float>(steps_)/static_cast<float>(scenario_.max_steps),
        static_cast<float>(scenario_.goal_velocity.x/scenario_.max_speed),
        static_cast<float>(scenario_.goal_velocity.y/scenario_.max_speed)};
    values.reserve(10+3*scenario_.obstacles.size());
    for(const auto& obstacle:scenario_.obstacles)
    {
        values.push_back(static_cast<float>(obstacle.center.x/scenario_.half_extent));
        values.push_back(static_cast<float>(obstacle.center.y/scenario_.half_extent));
        values.push_back(static_cast<float>(obstacle.radius/scenario_.half_extent));
    }
    return values;
}
ppo::rl::StepResult Environment::step(const ppo::rl::Value& action)
{
    if(!active_) throw std::logic_error("Navigation2D reset required");
    if(!ppo::rl::contains(actions_,action)) throw std::invalid_argument("invalid navigation action");
    const auto& commands=std::get<std::vector<float>>(action);
    const double previous_distance=distance(position_,goal_position_);
    constexpr double two_pi=6.28318530717958647692;
    heading_=std::remainder(heading_+commands[0]*scenario_.max_turn_rate*scenario_.dt,two_pi);
    speed_=std::clamp(speed_+commands[1]*scenario_.max_acceleration*scenario_.dt,0.0,scenario_.max_speed);
    const Point next{position_.x+std::cos(heading_)*speed_*scenario_.dt,
                     position_.y+std::sin(heading_)*speed_*scenario_.dt};
    const Point goal_delta{scenario_.goal_velocity.x * scenario_.dt,
                           scenario_.goal_velocity.y * scenario_.dt};
    // Relative swept motion detects a meeting even when both endpoints miss.
    const Point relative_next{next.x - goal_delta.x, next.y - goal_delta.y};
    const double arrival = entry(position_, relative_next, goal_position_, scenario_.arrival_radius);
    double collision=std::numeric_limits<double>::infinity();
    for(const auto& obstacle:scenario_.obstacles)
        collision=std::min(collision,entry(position_,next,obstacle.center,obstacle.radius));
    const double limit=scenario_.half_extent;
    if(next.x>limit) collision=std::min(collision,(limit-position_.x)/(next.x-position_.x));
    if(next.x<-limit) collision=std::min(collision,(-limit-position_.x)/(next.x-position_.x));
    if(next.y>limit) collision=std::min(collision,(limit-position_.y)/(next.y-position_.y));
    if(next.y<-limit) collision=std::min(collision,(-limit-position_.y)/(next.y-position_.y));
    const bool success=arrival<=1 && arrival<collision;
    const bool blocked=collision<=1 && collision<=arrival;
    const double fraction=std::min(1.0,std::min(arrival,collision));
    position_={position_.x+fraction*(next.x-position_.x),position_.y+fraction*(next.y-position_.y)};
    goal_position_ = {goal_position_.x + fraction * goal_delta.x, goal_position_.y + fraction * goal_delta.y};
    ++steps_;
    const bool truncated=steps_==scenario_.max_steps;
    active_=!(success || blocked || truncated);
    const double remaining=distance(position_,goal_position_);
    const double reward=(previous_distance-remaining)/scenario_.half_extent-0.001+(success?1.0:0.0)-(blocked?1.0:0.0);
    return {observe(),reward,success || blocked,truncated,
        {{"success",success},{"collision",blocked},{"distance",remaining}}};
}
const ppo::rl::Space& Environment::observation_space() const noexcept {return observations_;}
const ppo::rl::Space& Environment::action_space() const noexcept {return actions_;}
State Environment::goal_state() const noexcept
{
    const auto velocity = scenario_.goal_velocity;
    return {goal_position_, std::atan2(velocity.y, velocity.x), std::hypot(velocity.x, velocity.y)};
}
State Environment::state() const noexcept { return {position_, heading_, speed_}; }
Point Environment::position() const noexcept {return position_;}
const Scenario& Environment::scenario() const noexcept {return scenario_;}
} // namespace navigation
