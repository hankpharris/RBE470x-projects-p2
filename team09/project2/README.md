# Your goal #

In this scenario, you must plan the route of your agent from the top-left
corner to the exit. However, your route is obstructed - you need to use the bomb
to create a path to the exit.

## Variant 1: Alone in the world ##

In the first variant of this scenario, the world is deterministic and your agent
is alone in the environment.

## Variant 2: Random monster ##

In the second variant of this scenario, a stupid monster is present. The monster
chooses its next cell uniformly at random among the possible reachable cells.

## Variant 3: Self-preserving monster ##

In the third variant of this scenario, a smarter monster is present:
- The monster goes straight until it has reached an obstacle
- When it reaches an obstacle, it changes direction at random among the cells
  that are walkable and are not an explosion (if an agent, monster or character,
  touches an explosion, it dies)
- If the 8-distance of your agent to the monster is 1, the monster attacks your
  agent immediately and kills it

## Variant 4: Aggressive monster ##

In the fourth variant of this scenario, an aggressive monster is present:
- The monster goes straight until it has reached an obstacle
- When it reaches an obstacle, it changes direction at random among the cells
  that are walkable and are not an explosion (if an agent, monster or character,
  touches an explosion, it dies)
- If the 8-distance of your agent to the monster is 2, the monster moves towards
  your agent and attempts to kill it

## Variant 5: Stupid and Aggressive monsters together ##

In the fifth variant of this scenario, two monsters are present: an aggressive
one and a stupid one.

## Reinforcement Learning (Stable Baselines3) ##

A Gym-compatible environment for variant 5 is provided in `team09/project2/sb3_variant5_env.py`
so you can train reinforcement learning agents with Stable Baselines3.

1. Install the dependencies (for example inside a virtualenv):

       pip install stable-baselines3 gymnasium numpy

2. Train a PPO agent (this will create/update the model at the given path):

       python run_sb3_variant5.py train models/variant5_ppo.zip --timesteps 50000

3. Evaluate a saved model (use `--render` to print the board to the console):

       python run_sb3_variant5.py eval models/variant5_ppo.zip --episodes 5 --render

You can customise rewards, observation encoding, or action definitions inside
`sb3_variant5_env.py` to experiment with alternative training setups. The
default configuration already adds shaping rewards that encourage moving toward
the exit, placing bombs, and destroying walls/monsters so policies learn to
clear paths more quickly.

To watch the trained agent in the pygame GUI:

1. Place your trained model at `team09/project2/models/variant5_ppo.zip`
   (original location) or `team09/models/variant5_ppo.zip`, or export an
   alternative path via the `SB3_MODEL_PATH` environment variable.
2. Run `python variant5.py` as usual and press `Enter` to advance frames. The
   script now pulls in `team09/ppo_testcharacter.py`, which loads the PPO
   policy automatically.
