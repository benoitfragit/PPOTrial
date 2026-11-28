/**
 * @file PPOKit/src/editor/ppo_help.cpp
 * @brief Help texts of the PPO hyper-parameters, metrics and reward views.
 */
#include "ppokit/editor/ppo_help.hpp"

namespace ppokit::editor {

std::vector<HelpEntry> const& PpoParameterHelp()
{
    static std::vector<HelpEntry> const entries{
        {"gamma", "Discount factor (gamma)",
            "Weight of future rewards: a reward k steps ahead counts gamma^k. The horizon the agent plans for is "
            "about 1 / (1 - gamma) steps.",
            "0.95 to 0.999. 0.99 is about 100 steps.",
            "Too low: the agent ignores delayed outcomes (landing, mission end). Too high: noisier advantages "
            "and slower learning on long episodes."},
        {"lambda", "GAE lambda",
            "Bias/variance trade-off of the advantages: 0 trusts the critic after one step, 1 uses the whole "
            "observed return.",
            "0.9 to 0.98; 0.95 is a common default.",
            "Lower it when advantages are noisy and learning is erratic; raise it when the critic is poor "
            "(low explained variance)."},
        {"clip", "PPO clip range (epsilon)",
            "Largest change of the action probability ratio that still pushes the policy in one update. Keeps "
            "each update close to the policy that collected the data.",
            "0.1 to 0.3; 0.2 is the usual value.",
            "If clip_fraction stays above about 0.3 or approx_kl grows, lower it (or the learning rate). If "
            "learning is very slow and clip_fraction is near 0, it can be raised."},
        {"actor_learning_rate", "Actor learning rate",
            "Adam step size of the policy network.",
            "1e-4 to 3e-3; 3e-4 is a safe start.",
            "Too high: approx_kl spikes, entropy collapses, the return drops after improving. Too low: flat "
            "curves for many updates."},
        {"critic_learning_rate", "Critic learning rate",
            "Adam step size of the value network.",
            "Same order as the actor, often 1 to 3 times higher.",
            "Raise it if explained variance stays near 0 while the rewards are informative; lower it if "
            "value_loss oscillates."},
        {"entropy_coefficient", "Entropy bonus",
            "Reward for keeping the action distribution wide: it slows the collapse of exploration.",
            "0 to 0.01 for continuous actions.",
            "If action_std_mean collapses early and the return plateaus, add a little (0.001 to 0.01). Too much "
            "keeps the policy random."},
        {"value_coefficient", "Value loss weight",
            "Weight of the value loss in the one loss of an actor and a critic that share blocks "
            "(network.json \"share\"), optimized together. Unused when they are optimized apart.",
            "0.5 is the usual choice.",
            "Raise it if explained variance stays low; lower it if the value loss dominates the shared blocks and "
            "the policy stops improving."},
        {"max_gradient_norm", "Gradient norm limit",
            "Largest global norm of the gradients of one optimizer step; a larger one is scaled down to it "
            "(graph networks only). 0 turns it off.",
            "0 (off), or 0.5 to 1 to tame rare huge gradients, as many torch PPO implementations do.",
            "Try 0.5 if the losses spike and the policy collapses after a few bad updates. Too small slows "
            "learning."},
        {"epochs", "Epochs per update",
            "Passes over each rollout. More passes reuse the data more, but move further from the policy that "
            "collected it.",
            "3 to 10; 4 is a common default.",
            "If approx_kl grows within an update, reduce it. With small rollouts, a few more epochs can help."},
        {"minibatch", "Minibatch size",
            "Samples per optimizer step. Each epoch makes rollout / minibatch steps.",
            "64 to 4096, a divisor of the rollout size if possible.",
            "Small minibatches give noisy but many steps; large ones are smoother. Change it together with the "
            "learning rate."},
        {"initial_log_std", "Initial log standard deviation",
            "Spread of the Gaussian actions at the start of training (log scale): exp(-0.5) is about 0.6 in "
            "the action units before clipping.",
            "-1 to 0.",
            "Too low: little exploration from the start. Too high: actions mostly clipped at the bounds."},
        {"minimum_log_std", "Minimum log standard deviation",
            "Lower bound of the Gaussian log standard deviation, applied after every actor step: the policy "
            "keeps at least exp(minimum) of exploration.",
            "-5 (default) to -2.",
            "Raise it when the policy stops exploring too early."},
        {"maximum_log_std", "Maximum log standard deviation",
            "Upper bound of the Gaussian log standard deviation, applied after every actor step.",
            "2 (default); -1 keeps the actions well inside [-1, 1].",
            "Lower it when the actions saturate at their bounds."},
        {"seed", "Seed",
            "Fixes weights, sampling and shuffling: two runs with the same seed and settings are identical.",
            "Any integer.",
            "Compare settings over several seeds: one seed proves nothing about convergence."},
        {"rollout_steps", "Rollout steps",
            "Environment steps collected before each update.",
            "1024 to 8192 for a single agent.",
            "Longer rollouts give smoother advantages but fewer updates; they should cover several episodes."},
        {"target_kl", "Target KL",
            "KL budget of one minibatch: when its sampled KL passes a multiple of it, the actor step is skipped "
            "(the critic still learns).",
            "0 (off), or 0.01 to 0.05.",
            "Many skips (actor_kl_skips) mean the actor steps are too large: lower the actor learning rate."},
        {"hidden_width", "Hidden width",
            "Neurons per hidden layer of the networks.",
            "64 to 256.",
            "Wider networks fit more but learn slower per sample; change it only with a reason."},
        {"probe_every", "Probe every",
            "Updates between two probed updates: the Probes tab shows what each block outputs and how much "
            "gradient reaches it. Probes only read the networks: training is the same with or without them.",
            "10 (0: off).",
            "Probe more often to follow a problem closely; each probed update costs a little extra time."},
        {"blue_mobiles", "Blue aircraft",
            "Blue agents trained together by this run (one shared policy, one critic for the team).",
            "1 to 9.",
            "A policy trained with few agents may not hold with many: evaluate the counts you will fly."},
        {"red_mobiles", "Red aircraft",
            "Red opponents in the episodes of this run.",
            "0 to 10.",
            "0 trains patrol alone; add Red aircraft once the patrol is learned."},
    };
    return entries;
}

std::vector<HelpEntry> const& PpoMetricHelp()
{
    static std::vector<HelpEntry> const entries{
        {"policy_loss", "Policy loss",
            "Minus the clipped PPO surrogate: how much the update improved the probability of good actions.",
            "Small, around 0; its sign alone means little.",
            "Use it to spot NaN or explosions. Judge learning by the returns, not by this loss."},
        {"value_loss", "Value loss",
            "Mean squared error of the critic against the value targets.",
            "Decreases, then follows the scale of the returns.",
            "If it grows steadily, the critic cannot follow: check the reward scale or lower the critic learning "
            "rate."},
        {"entropy", "Entropy",
            "Width of the action distribution, summed over action components.",
            "Decreases slowly as the policy becomes confident.",
            "A fast collapse means exploration stops early: consider the entropy bonus."},
        {"approx_kl", "Approximate KL",
            "How far an update moved the policy from the one that collected the data.",
            "Below about 0.02 per update.",
            "Above 0.05 repeatedly: updates are too large; lower the learning rate, epochs or clip."},
        {"clip_fraction", "Clip fraction",
            "Share of samples whose probability ratio left the clip range.",
            "0.05 to 0.3.",
            "Near 0: updates may be too timid. Above 0.3: too aggressive."},
        {"explained_variance", "Explained variance",
            "How much of the variance of the value targets the critic explains, before the update: 1 perfect, 0 "
            "no better than a constant.",
            "Rises towards 0.5 to 0.9 when the critic learns.",
            "Stuck near 0 or negative: advantages are mostly noise; the critic needs a better input or rate."},
        {"advantage_abs_mean", "Mean |advantage|",
            "Mean absolute advantage of the learnable samples, before normalization.",
            "Follows the reward scale.",
            "A sudden jump usually follows a change of rewards or weights."},
        {"action_std_mean", "Mean action deviation",
            "Mean standard deviation of the Gaussian actions after the update.",
            "Decreases slowly.",
            "Near 0 early: exploration has collapsed. Growing: the policy is getting less certain."},
        {"learnable_rows", "Learnable rows",
            "Samples that trained the actor (available agents).",
            "Rollout size minus the steps of destroyed agents.",
            "Much lower than the critic rows: many agents are destroyed early."},
        {"critic_rows", "Critic rows",
            "Samples that trained the critic (every agent, alive or not).",
            "Agents x team steps of the rollout.",
            "Constant for a given rollout size."},
        {"mean_reward", "Mean reward",
            "Mean weighted reward per step of the rollout.",
            "Rises as the policy learns.",
            "Compare it only between runs with the same weights: a weight change moves it on its own."},
        {"max_approx_kl", "Largest approximate KL",
            "Largest approximate KL of the minibatches of the update.",
            "Below the target KL.",
            "Above it often: the update is too large for the data; lower the actor learning rate or epochs."},
        {"max_sampled_kl", "Largest sampled KL",
            "Largest sampled KL, minus the mean log ratio, of the minibatches; it decides the skipped actor "
            "steps. Small negative values are estimation noise.",
            "Close to the approximate KL, below the target KL.",
            "Far from the approximate KL: the estimate is noisy; prefer a larger minibatch."},
        {"actor_kl_skips", "Skipped actor steps",
            "Minibatches whose actor step was skipped because the sampled KL passed its limit.",
            "0 most of the time.",
            "Frequent skips waste data: lower the actor learning rate or raise the target KL slightly."},
    };
    return entries;
}

std::vector<HelpEntry> const& RewardViewHelp()
{
    static std::vector<HelpEntry> const entries{
        {"weighted_step_mean", "Per step (weighted)",
            "Mean contribution of each channel to the reward per step, with its weight: what the agent optimizes.",
            "Compare channels on the same scale.", "A channel that dominates drives the behaviour."},
        {"episode_mean", "Per episode",
            "Mean weighted sum of each channel per completed episode: its share of the episode return.",
            "Follow the return of one episode.", "Empty while no episode has ended in the window."},
        {"cumulative", "Cumulative",
            "Weighted sum of each channel since the start of the run.",
            "Grows or falls monotonically for signed channels.", "The slope is the per-step contribution."},
        {"step_mean", "Per step (raw)",
            "Mean raw value of each channel per step, before its weight.",
            "In the units of the channel.", "Use it to see a channel whose weight is 0."},
    };
    return entries;
}

std::vector<HelpEntry> const& ProbeHelp()
{
    static std::vector<HelpEntry> const entries{
        {"probe_std", "Output spread",
            "Standard deviation of the outputs of the block over the probed samples: how much it varies from one "
            "sample to the next.",
            "Around 0.1 to 1 for hidden blocks.",
            "Near 0: the block outputs a constant and tells the next blocks nothing. Growing without bound: "
            "activations explode; lower the learning rate or normalize the inputs."},
        {"probe_mean", "Output mean",
            "Mean of the outputs of the block.",
            "Close to 0 for tanh blocks, positive for ReLU blocks.",
            "A mean drifting far from 0 often comes with saturated or dead units."},
        {"probe_dead", "Dead units",
            "Share of outputs exactly zero. For a ReLU block these units receive no gradient: they stopped "
            "learning.",
            "Below 0.5 for ReLU blocks; 0 for tanh blocks.",
            "Rising towards 1: the block dies. Lower the learning rate, normalize the inputs, or use tanh."},
        {"probe_saturated", "Saturated units",
            "Share of outputs with |x| >= 0.99. A saturated tanh or sigmoid unit passes almost no gradient.",
            "Below 0.2.",
            "High: inputs are too large or the weights grew. Normalize the inputs or lower the learning rate."},
        {"probe_gradient_norm", "Gradient size",
            "Mean L2 norm of the gradient of the block's parameters per optimizer step: how strongly the block is "
            "being changed.",
            "Steady values, of similar order across the blocks of a network.",
            "Near 0: nothing reaches the block (vanishing gradient). Very large or growing: updates may explode."},
    };
    return entries;
}

HelpEntry const* FindHelp(std::string_view name)
{
    for (auto const* list : {&PpoParameterHelp(), &PpoMetricHelp(), &RewardViewHelp(), &ProbeHelp()})
    {
        for (HelpEntry const& entry : *list)
        {
            if (name == entry.name)
            {
                return &entry;
            }
        }
    }
    return nullptr;
}

} // namespace ppokit::editor
