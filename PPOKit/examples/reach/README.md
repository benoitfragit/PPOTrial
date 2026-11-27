# reach

A PPOKit project created by `ppokit-new reach` (or "New project" in the studio). It
already trains: a point in a plane moves to a target. Change it into your own task.

## Use it

Open this folder in `ppokit-studio`: the overview shows the observation, the actor and
the critic, GAE and the PPO update; click a box to edit it. Train, Evaluate and Inference
start the program `ppokit_reach` (build it once) and show its console and curves live.

The same from a terminal:

```
ppokit_reach check          # validates every file, says what is wrong in plain words
ppokit_reach train          # 100 updates; run it again to continue where it stopped
ppokit_reach eval           # plays the policy on the seeds of eval/smoke.json
ppokit_reach play           # plays a few episodes and prints them
```

## The project folder

| File | What it is |
|------|------------|
| `project.json` | name, and the program that trains, evaluates and plays the project |
| `task.json` | observation, actions, reward channels, scenario parameters |
| `network.json` | actor and critic (`network.layout.json`: where the blocks are drawn) |
| `training.json` | PPO and GAE parameters, probes, curriculum |
| `eval/smoke.json` | evaluation seeds and pass gates |
| `components/` | component DLLs the network uses |
| `checkpoints/` | `latest.ppokit` (resume), `policy.ppokit` (evaluate), `metrics.jsonl` (journal) |
| `task.hpp` | the task in C++: physics, observation, rewards |

`task.json` and `task.hpp` go together: the layout header `reach_layout.hpp` is generated
from `task.json` at build time, so a field added there is a named offset in C++.

## Make it yours

1. Describe your observation, actions and rewards in `task.json`.
2. Rewrite `Reset` and `Step` in `task.hpp`. Keep the episode ends honest: `Terminated`
   when the task is over (success or failure), `Truncated` when time runs out.
3. Build, then Train, Evaluate and Inference from the studio.

A passing unit test does not mean the policy learned: only an evaluation, on seeds
training never sees, says that.
