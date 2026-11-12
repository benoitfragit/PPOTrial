#include "flight_learner.hpp"
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <vector>

namespace ppo::examples::flight {
namespace rlt = rl_tools;

void ComputeAdvantages(Transition* transitions, std::size_t count, TrainingSettings const& t) noexcept
{
    double gae = 0;
    for (std::size_t i=count; i>0; --i)
    {
        auto& row = transitions[i-1];
        gae = AdvantageDelta(row.reward,row.value,row.nextValue,row.terminated,t.gamma)
            + ((row.terminated || row.truncated) ? 0 : t.gamma*t.lambda*gae);
        row.advantage = gae;
        row.target = gae+row.value;
    }
}

struct FlightLearner::Impl
{
    Config config;
    std::mt19937 rng;
    std::vector<Transition> rows;
    std::vector<std::size_t> order;
    std::size_t collected{};
    LearningStats stats{};
    bool failed{};
    explicit Impl(Config const& c) : config(c), rng(c.training.seed), rows(c.training.rolloutSteps), order(rows.size())
    { std::iota(order.begin(),order.end(),0U); }
    virtual ~Impl() = default;
    virtual PolicyVector Evaluate(Observation const&) = 0;
    virtual double Value(Observation const&) = 0;
    virtual void Optimize() = 0;
    virtual void Write(std::ostream&) const = 0;
    virtual void Read(std::istream&) = 0;
    void Check() const
    { if (failed) throw std::runtime_error("FlightControl3D: learner failed; restart from a valid checkpoint"); }
};

namespace {
using Device = rlt::devices::DEVICE_FACTORY<>;
using Index = typename Device::index_t;
using Optimizer = rlt::nn::optimizers::Adam<rlt::nn::optimizers::adam::Specification<float,Index>>;

template<Index Width, Index Outputs>
using Structure = rlt::nn_models::mlp::StructureSpecification<float,Index,ObservationSize,Outputs,3,Width,
    rlt::nn::activation_functions::TANH,rlt::nn::activation_functions::IDENTITY,1,rlt::MatrixStaticTag>;
template<Index Width, Index Outputs>
using Network = rlt::nn_models::mlp::NeuralNetworkAdam<rlt::nn_models::mlp::AdamSpecification<Structure<Width,Outputs>>>;
template<Index Columns> using Matrix = rlt::MatrixStatic<rlt::matrix::Specification<float,Index,1,Columns>>;

void CheckObservation(Observation const& observation)
{
    for (float value : observation)
        if (!std::isfinite(value) || std::abs(value)>1.F) throw std::invalid_argument("FlightControl3D: observation is not normalized");
}

template<typename Net, typename Visitor>
void VisitLayers(Net& net, Visitor&& visit)
{
    visit(net.input_layer);
    for (auto& layer : net.hidden_layers) visit(layer);
    visit(net.output_layer);
}

template<typename Parameter, typename Visitor>
void VisitParameter(Parameter& parameter, Visitor&& visit)
{
    visit(parameter.parameters);
    visit(parameter.gradient_first_order_moment);
    visit(parameter.gradient_second_order_moment);
}

template<Index Width>
class Learner final : public FlightLearner::Impl
{
    Device device_{};
    Network<Width,PolicyOutputs> actor_{};
    Network<Width,1> critic_{};
    Optimizer actorOptimizer_{}, criticOptimizer_{};
    Matrix<ObservationSize> input_{};
    Matrix<PolicyOutputs> actorOutput_{}, actorDerivative_{};
    Matrix<1> valueOutput_{}, valueDerivative_{};
    typename Network<Width,PolicyOutputs>::template Buffer<1,rlt::MatrixStaticTag> actorBuffer_{};
    typename Network<Width,1>::template Buffer<1,rlt::MatrixStaticTag> criticBuffer_{};

    void Input(Observation const& observation)
    {
        CheckObservation(observation);
        for (Index i=0; i<ObservationSize; ++i) rlt::set(input_,0,i,observation[i]);
    }
    void Setup(Optimizer& optimizer, double rate)
    {
        optimizer.parameters.alpha = static_cast<float>(rate);
        optimizer.parameters.beta_1 = static_cast<float>(config.training.adamBeta1);
        optimizer.parameters.beta_2 = static_cast<float>(config.training.adamBeta2);
        optimizer.parameters.epsilon = static_cast<float>(config.training.adamEpsilon);
    }
    template<typename Net>
    void ClipGradients(Net& net)
    {
        double normSquared = 0;
        auto sum = [&](auto& layer)
        {
            SumMatrix(layer.weights.gradient,normSquared);
            SumMatrix(layer.biases.gradient,normSquared);
        };
        VisitLayers(net,sum);
        if (!std::isfinite(normSquared)) throw std::runtime_error("FlightControl3D: non-finite PPO gradient");
        double const scale = normSquared > 0 ? std::min(1.0,config.training.maxGradientNorm/std::sqrt(normSquared)) : 1;
        auto clip = [scale](auto& layer)
        {
            ScaleMatrix(layer.weights.gradient,scale);
            ScaleMatrix(layer.biases.gradient,scale);
        };
        VisitLayers(net,clip);
    }
    template<typename M> static void SumMatrix(M const& matrix, double& sum)
    {
        for (Index i=0; i<M::ROWS; ++i) for (Index j=0; j<M::COLS; ++j)
        { double const v=rlt::get(matrix,i,j); sum += v*v; }
    }
    template<typename M> static void ScaleMatrix(M& matrix, double scale)
    {
        for (Index i=0; i<M::ROWS; ++i) for (Index j=0; j<M::COLS; ++j)
            rlt::set(matrix,i,j,static_cast<float>(rlt::get(matrix,i,j)*scale));
    }
    template<typename Net> static void WriteNetwork(std::ostream& stream, Net const& net)
    {
        auto matrix = [&](auto const& m)
        {
            using M = std::decay_t<decltype(m)>;
            for (Index i=0; i<M::ROWS; ++i) for (Index j=0; j<M::COLS; ++j) stream << rlt::get(m,i,j) << ' ';
            stream << '\n';
        };
        auto layer = [&](auto const& l) { VisitParameter(l.weights,matrix); VisitParameter(l.biases,matrix); };
        VisitLayers(net,layer);
    }
    template<typename Net> static void ReadNetwork(std::istream& stream, Net& net)
    {
        auto matrix = [&](auto& m)
        {
            using M = std::decay_t<decltype(m)>;
            for (Index i=0; i<M::ROWS; ++i) for (Index j=0; j<M::COLS; ++j)
            {
                float value{};
                if (!(stream >> value) || !std::isfinite(value)) throw std::runtime_error("FlightControl3D: corrupt checkpoint parameter");
                rlt::set(m,i,j,value);
            }
        };
        auto layer = [&](auto& l) { VisitParameter(l.weights,matrix); VisitParameter(l.biases,matrix); };
        VisitLayers(net,layer);
    }
public:
    explicit Learner(Config const& c) : Impl(c)
    {
        rlt::init_weights(device_,actor_,rng); rlt::init_weights(device_,critic_,rng);
        rlt::reset_optimizer_state(device_,actorOptimizer_,actor_);
        rlt::reset_optimizer_state(device_,criticOptimizer_,critic_);
        Setup(actorOptimizer_,c.training.actorRate); Setup(criticOptimizer_,c.training.criticRate);
        // Zero output weights give an observation-independent, explicit initial exploration scale.
        for (Index i=0; i<PolicyOutputs; ++i)
        {
            for (Index j=0; j<Width; ++j) rlt::set(actor_.output_layer.weights.parameters,i,j,0.F);
            rlt::set(actor_.output_layer.biases.parameters,0,i,
                i>=ContinuousActions && i<2*ContinuousActions ? static_cast<float>(c.training.initialLogStd) : 0.F);
        }
        rlt::set_all(device_,input_,0.F);
        rlt::zero_gradient(device_,actor_); rlt::zero_gradient(device_,critic_);
        rlt::forward(device_,actor_,input_,actorOutput_);
        rlt::forward(device_,critic_,input_,valueOutput_);
    }
    PolicyVector Evaluate(Observation const& observation) override
    {
        Check(); Input(observation);
        rlt::evaluate(device_,actor_,input_,actorOutput_,actorBuffer_);
        PolicyVector output{};
        for (Index i=0; i<PolicyOutputs; ++i)
        {
            output[i] = rlt::get(actorOutput_,0,i);
            if (!std::isfinite(output[i])) throw std::runtime_error("FlightControl3D: invalid actor");
        }
        return output;
    }
    double Value(Observation const& observation) override
    {
        Check(); Input(observation);
        rlt::evaluate(device_,critic_,input_,valueOutput_,criticBuffer_);
        double const value = rlt::get(valueOutput_,0,0);
        if (!std::isfinite(value)) throw std::runtime_error("FlightControl3D: invalid critic");
        return value;
    }
    void Optimize() override
    {
        Check();
        if (collected != rows.size()) throw std::logic_error("FlightControl3D: incomplete rollout");
        try
        {
            ComputeAdvantages(rows.data(),collected,config.training);
            double mean=0, variance=0;
            for (auto const& row : rows) mean += row.advantage/static_cast<double>(collected);
            for (auto const& row : rows) variance += (row.advantage-mean)*(row.advantage-mean)/static_cast<double>(collected);
            double const sigma = std::sqrt(variance);
            for (auto& row : rows) row.advantage = sigma > std::numeric_limits<double>::epsilon() ? (row.advantage-mean)/sigma : 0;
            stats.policyLoss=0; stats.valueLoss=0; stats.entropy=0;
            double const samples = static_cast<double>(collected*config.training.epochs);
            for (std::size_t epoch=0; epoch<config.training.epochs; ++epoch)
            {
                std::iota(order.begin(),order.end(),0U);
                std::shuffle(order.begin(),order.end(),rng);
                for (std::size_t start=0; start<collected; start+=config.training.minibatch)
                {
                    rlt::zero_gradient(device_,actor_); rlt::zero_gradient(device_,critic_);
                    for (std::size_t j=0; j<config.training.minibatch; ++j)
                    {
                        auto const& row = rows[order[start+j]];
                        Input(row.observation);
                        rlt::forward(device_,actor_,input_,actorOutput_);
                        PolicyVector output{};
                        for (Index k=0; k<PolicyOutputs; ++k) output[k]=rlt::get(actorOutput_,0,k);
                        auto const loss=ClippedLoss(output,row.action,row.advantage,config.training);
                        if (!std::isfinite(loss.loss)) throw std::runtime_error("FlightControl3D: invalid PPO loss");
                        for (Index k=0; k<PolicyOutputs; ++k)
                            rlt::set(actorDerivative_,0,k,static_cast<float>(loss.derivative[k]/static_cast<double>(config.training.minibatch)));
                        rlt::backward(device_,actor_,input_,actorDerivative_,actorBuffer_);
                        rlt::forward(device_,critic_,input_,valueOutput_);
                        double const difference=rlt::get(valueOutput_,0,0)-row.target;
                        rlt::set(valueDerivative_,0,0,static_cast<float>(config.training.valueWeight*difference/static_cast<double>(config.training.minibatch)));
                        rlt::backward(device_,critic_,input_,valueDerivative_,criticBuffer_);
                        stats.policyLoss += loss.loss/samples;
                        stats.entropy += loss.entropy/samples;
                        stats.valueLoss += 0.5*difference*difference/samples;
                    }
                    ClipGradients(actor_); ClipGradients(critic_);
                    rlt::step(device_,actorOptimizer_,actor_); rlt::step(device_,criticOptimizer_,critic_);
                }
            }
            if (!std::isfinite(stats.policyLoss) || !std::isfinite(stats.valueLoss)) throw std::runtime_error("FlightControl3D: invalid training diagnostics");
            ++stats.updates; collected=0;
        }
        catch (...) { failed=true; throw; }
    }
    void Write(std::ostream& stream) const override
    {
        Check();
        stream << "PPO_FLIGHT_HYBRID_V1 " << ObservationSize << ' ' << ContinuousActions << ' ' << Width << ' ' << stats.updates << '\n';
        stream << std::quoted(config.checkpointSignature) << '\n' << std::setprecision(std::numeric_limits<float>::max_digits10);
        stream << actorOptimizer_.age << ' ' << criticOptimizer_.age << '\n' << rng << '\n';
        WriteNetwork(stream,actor_); WriteNetwork(stream,critic_);
    }
    void Read(std::istream& stream) override
    {
        Check();
        std::string magic, signature;
        std::size_t observations{}, actions{}, width{};
        std::uint64_t updates{};
        Index actorAge{},criticAge{};
        std::mt19937 candidateRng;
        if (!(stream >> magic >> observations >> actions >> width >> updates >> std::quoted(signature) >> actorAge >> criticAge >> candidateRng)
            || magic!="PPO_FLIGHT_HYBRID_V1" || observations!=ObservationSize || actions!=ContinuousActions || width!=Width
            || signature!=config.checkpointSignature || actorAge==0 || criticAge==0)
            throw std::runtime_error("FlightControl3D: incompatible/corrupt checkpoint header");
        // RLtools static matrices contain self-pointers: never use C++ copies.
        // Stage independent networks on the heap (including on Windows' small stack).
        auto candidateActor=std::make_unique<Network<Width,PolicyOutputs>>();
        auto candidateCritic=std::make_unique<Network<Width,1>>();
        rlt::forward(device_,actor_,input_,actorOutput_);
        rlt::forward(device_,critic_,input_,valueOutput_);
        rlt::copy(device_,device_,actor_,*candidateActor);
        rlt::copy(device_,device_,critic_,*candidateCritic);
        ReadNetwork(stream,*candidateActor); ReadNetwork(stream,*candidateCritic);
        stream >> std::ws;
        if (!stream.eof()) throw std::runtime_error("FlightControl3D: trailing checkpoint data");
        rlt::copy(device_,device_,*candidateActor,actor_);
        rlt::copy(device_,device_,*candidateCritic,critic_);
        rng=candidateRng;
        actorOptimizer_.age=actorAge; criticOptimizer_.age=criticAge;
        stats={}; stats.updates=updates; collected=0;
    }
};
} // namespace

FlightLearner::FlightLearner(Config const& c)
{
    Validate(c);
    if (c.training.hiddenWidth==32) impl_=std::make_unique<Learner<32>>(c);
    else if (c.training.hiddenWidth==64) impl_=std::make_unique<Learner<64>>(c);
    else impl_=std::make_unique<Learner<128>>(c);
}
FlightLearner::~FlightLearner() = default;
PolicyVector FlightLearner::Evaluate(Observation const& observation) { return impl_->Evaluate(observation); }
PolicyAction FlightLearner::Choose(Observation const& observation, bool deterministic)
{ return Sample(impl_->Evaluate(observation),impl_->config.training,impl_->rng,deterministic); }
bool FlightLearner::Ready() const noexcept { return impl_->collected==impl_->rows.size(); }
std::size_t FlightLearner::Collected() const noexcept { return impl_->collected; }
LearningStats const& FlightLearner::Stats() const noexcept { return impl_->stats; }
void FlightLearner::Optimize() { impl_->Optimize(); }

void FlightLearner::Record(Observation const& observation, PolicyAction const& action, StepResult const& step, Observation const& next)
{
    impl_->Check();
    if (Ready()) throw std::logic_error("FlightControl3D: rollout full");
    if (!std::isfinite(step.reward) || !std::isfinite(action.logProbability) || (step.terminated && step.truncated))
        throw std::invalid_argument("FlightControl3D: invalid transition");
    auto& row=impl_->rows[impl_->collected];
    row={observation,action,step.reward,impl_->Value(observation),step.terminated ? 0 : impl_->Value(next),0,0,step.terminated,step.truncated};
    ++impl_->collected;
}

void FlightLearner::Save(std::filesystem::path const& path) const
{
    if (impl_->collected!=0) throw std::logic_error("FlightControl3D: save at an update boundary");
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    auto temporary=path; temporary += ".tmp";
    { std::ofstream stream(temporary,std::ios::trunc); impl_->Write(stream); stream.flush();
      if (!stream) throw std::runtime_error("FlightControl3D: checkpoint write failed"); }
    auto backup=path; backup += ".previous";
    if (std::filesystem::exists(backup)) std::filesystem::remove(backup);
    bool const previous=std::filesystem::exists(path);
    if (previous) std::filesystem::rename(path,backup);
    try { std::filesystem::rename(temporary,path); }
    catch (...) { if (previous) std::filesystem::rename(backup,path); throw; }
}
void FlightLearner::Load(std::filesystem::path const& path)
{
    if (impl_->collected!=0) throw std::logic_error("FlightControl3D: load at an update boundary");
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("FlightControl3D: checkpoint not found");
    impl_->Read(stream);
}
} // namespace ppo::examples::flight
