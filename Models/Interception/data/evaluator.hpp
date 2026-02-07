#pragma once

#include "interface.hpp"

#include <string>
#include <memory>
#include <filesystem>

namespace rl {

/// <summary>
/// This class is written to evaluate an agent 
/// using a custom and fixed metric
/// </summary>
class Evaluator:
    public ppo::model::IEvaluator 
{
public:
    Evaluator(void);

    ~Evaluator(void);

    Evaluator(Evaluator&) = delete;
    Evaluator(Evaluator&&) = delete;

    Evaluator& operator=(Evaluator&) = delete;
    Evaluator& operator=(Evaluator&&) = delete;

    void Start(std::string const& name, std::filesystem::path const& modelpath) final;
    void Stop(void) final;

    ppo::model::sEvaluatorState GetSnapshot(void) const noexcept final;

private:
    struct Impl;
    std::unique_ptr<Impl> p_impl{ nullptr };
};

} /* rl */