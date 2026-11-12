#include "flight_model_store.hpp"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace ppo::examples::flight {
namespace {
using Json=nlohmann::json;
std::string Checksum(std::filesystem::path const& path)
{
    // FNV-1a is an integrity checksum, not a security/authentication mechanism.
    std::uint64_t hash=14695981039346656037ULL;
    std::ifstream stream(path,std::ios::binary);
    if(!stream) throw std::runtime_error("FlightControl3D: missing BEST checkpoint");
    char byte{};
    while(stream.get(byte)) { hash^=static_cast<unsigned char>(byte);hash*=1099511628211ULL; }
    if(!stream.eof()) throw std::runtime_error("FlightControl3D: BEST checksum read failure");
    std::ostringstream text;text<<std::hex<<std::setw(16)<<std::setfill('0')<<hash;return text.str();
}
struct BestRecord final { ValidationMetrics metrics{};std::filesystem::path checkpoint; };
std::optional<BestRecord> ReadBest(Config const& config,std::filesystem::path const& directory,bool allowTrainingChanges = false)
{
    auto const pointer=directory/"best/current.json";
    if(!std::filesystem::exists(pointer)) return std::nullopt;
    std::ifstream stream(pointer);Json json;stream>>json;
    auto const signature=json.at("signature").get<std::string>();
    bool const changedContract=signature!=config.checkpointSignature;
    if(json.at("formatVersion")!=1 || (changedContract && (!allowTrainingChanges || !CompatibleTrainingSignature(signature,config.checkpointSignature)))
        || json.at("seed").get<std::uint32_t>()!=config.evaluation.seed
        || json.at("requiredEpisodes").get<std::size_t>()!=config.evaluation.episodes)
        throw std::runtime_error("FlightControl3D: incompatible BEST validation contract");
    BestRecord record{};
    auto& m=record.metrics;
    m.update=json.at("update").get<std::uint64_t>();m.episodes=json.at("episodes").get<std::size_t>();
    m.touchdowns=json.at("touchdowns").get<std::size_t>();m.stops=json.at("stops").get<std::size_t>();
    m.crashes=json.at("crashes").get<std::size_t>();m.timeouts=json.at("timeouts").get<std::size_t>();
    m.touchdownSumMeters=json.at("touchdownSumMeters").get<double>();m.stopSumMeters=json.at("stopSumMeters").get<double>();
    auto const filename=std::filesystem::path(json.at("checkpoint").get<std::string>());
    if(filename.empty() || filename.is_absolute() || filename!=filename.filename() || !Validated(config,m))
        throw std::runtime_error("FlightControl3D: unvalidated BEST manifest");
    record.checkpoint=directory/"best"/filename;
    if(Checksum(record.checkpoint)!=json.at("fnv1a64").get<std::string>())
        throw std::runtime_error("FlightControl3D: BEST checkpoint/manifest mismatch");
    // Preserve the old generation; its score requires the exact current contract.
    if(changedContract) return std::nullopt;
    return record;
}
void CommitPointer(std::filesystem::path const& path,Json const& json)
{
    auto temporary=path;temporary+=".tmp";
    { std::ofstream stream(temporary,std::ios::trunc);stream<<json.dump(2)<<'\n';stream.flush();
        if(!stream) throw std::runtime_error("FlightControl3D: BEST manifest write failed"); }
    auto backup=path;backup+=".previous";
    if(std::filesystem::exists(backup)) std::filesystem::remove(backup);
    bool const hadPrevious=std::filesystem::exists(path);
    if(hadPrevious) std::filesystem::rename(path,backup);
    try { std::filesystem::rename(temporary,path); }
    catch(...) { if(hadPrevious) std::filesystem::rename(backup,path);throw; }
}
}
void ValidationMetrics::Record(MissionState const& mission)
{
    if(mission.outcome==Outcome::Flying) throw std::logic_error("FlightControl3D: incomplete validation episode");
    ++episodes;
    if(mission.touchedDown) { ++touchdowns;touchdownSumMeters+=mission.touchdownFromThreshold; }
    if(mission.outcome==Outcome::Stopped) { ++stops;stopSumMeters+=mission.stopFromThreshold; }
    if(mission.outcome==Outcome::Crash) ++crashes;
    if(mission.outcome==Outcome::Timeout) ++timeouts;
}
bool Validated(Config const& config,ValidationMetrics const& m) noexcept
{
    return m.episodes==config.evaluation.episodes && m.episodes>0 && m.touchdowns<=m.episodes && m.stops<=m.touchdowns
        && m.stops<=m.episodes && m.crashes<=m.episodes && m.timeouts<=m.episodes
        && m.stops+m.crashes+m.timeouts==m.episodes && m.SuccessRate()>=config.evaluation.minimumSuccessRate
        && std::isfinite(m.stopSumMeters) && std::isfinite(m.touchdownSumMeters)
        && m.stopSumMeters>=0 && m.touchdownSumMeters>=0
        && m.stopSumMeters<=static_cast<double>(m.stops)*config.runway.lengthMeters
        && m.touchdownSumMeters<=static_cast<double>(m.touchdowns)*config.runway.lengthMeters;
}
bool Better(ValidationMetrics const& candidate,ValidationMetrics const& incumbent) noexcept
{
    return candidate.SuccessRate()>incumbent.SuccessRate()
        || (candidate.SuccessRate()==incumbent.SuccessRate() && candidate.MeanStop()<incumbent.MeanStop());
}
FlightModelStore::FlightModelStore(Config const& config,std::filesystem::path directory)
    : config_(config),directory_(std::move(directory)) {}
std::filesystem::path FlightModelStore::Latest() const { return directory_/"latest/flight_hybrid_v1.txt"; }
std::optional<ValidationMetrics> FlightModelStore::BestMetrics(bool allowTrainingChanges) const
{
    auto const record=ReadBest(config_,directory_,allowTrainingChanges);
    return record ? std::optional<ValidationMetrics>(record->metrics) : std::nullopt;
}
void FlightModelStore::LoadBest(FlightLearner& learner) const
{
    auto const record=ReadBest(config_,directory_);
    if(!record) throw std::runtime_error("FlightControl3D: no validated BEST; train and complete validation first");
    learner.Load(record->checkpoint);
    if(learner.Stats().updates!=record->metrics.update)
        throw std::runtime_error("FlightControl3D: BEST update/validation mismatch");
}
bool FlightModelStore::Promote(FlightLearner const& learner,ValidationMetrics const& metrics)
{
    if(!Validated(config_,metrics)) return false;
    if(metrics.update!=learner.Stats().updates) throw std::logic_error("FlightControl3D: validation must describe the frozen candidate");
    auto const best=ReadBest(config_,directory_,true);
    if(best && !Better(metrics,best->metrics)) return false;
    auto const bestDirectory=directory_/"best";
    std::filesystem::create_directories(bestDirectory);
    auto const pending=bestDirectory/"candidate.pending";
    learner.Save(pending);
    auto const checksum=Checksum(pending);
    auto const filename="flight_"+std::to_string(metrics.update)+"_"+checksum+".txt";
    auto const destination=bestDirectory/filename;
    if(std::filesystem::exists(destination))
    {
        if(Checksum(destination)!=checksum) throw std::runtime_error("FlightControl3D: existing BEST generation mismatch");
        std::filesystem::remove(pending);
    }
    else std::filesystem::rename(pending,destination);
    Json manifest{{"formatVersion",1},{"signature",config_.checkpointSignature},{"seed",config_.evaluation.seed},
        {"requiredEpisodes",config_.evaluation.episodes},{"update",metrics.update},{"episodes",metrics.episodes},
        {"touchdowns",metrics.touchdowns},{"stops",metrics.stops},{"crashes",metrics.crashes},{"timeouts",metrics.timeouts},
        {"touchdownSumMeters",metrics.touchdownSumMeters},{"stopSumMeters",metrics.stopSumMeters},
        {"score",metrics.Score()},{"checkpoint",filename},{"fnv1a64",checksum}};
    CommitPointer(bestDirectory/"current.json",manifest);
    return true;
}
} // namespace ppo::examples::flight
