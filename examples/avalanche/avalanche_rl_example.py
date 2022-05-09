"""
This is an experimental integration between AvalancheRL and tella.
It is not finished.
AvalancheRL makes additional assumptions:
1. Have access to a pytorch model/optimizer
2. Run the training loop for the agent
which means any tella agent will need to be updated to work with AvalancheRL.
"""

import tella
from typing import *
from avalanche_rl.training.strategies.rl_base_strategy import RLBaseStrategy
from avalanche_rl.training.strategies.buffers import Rollout
from avalanche_rl.benchmarks.generators.rl_benchmark_generators import (
    gym_benchmark_generator,
)
import torch
import numpy as np
import gym
import logging

logger = logging.getLogger(__name__)


class TellaRandomAgent(tella.ContinualRLAgent):
    def __init__(
        self,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: Optional[str] = None,
    ) -> None:
        super().__init__(
            rng_seed, observation_space, action_space, num_envs, config_file
        )

        self.model = torch.nn.Linear(4, 2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def choose_actions(
        self, observations: List[Optional[tella.Observation]]
    ) -> List[Optional[tella.Action]]:
        return [
            None if obs is None else self.action_space.sample() for obs in observations
        ]

    def receive_transitions(self, transitions: List[Optional[tella.Transition]]):
        pass


class TellaToAvalancheWrapper(RLBaseStrategy):
    def __init__(self, agent: tella.agents.ContinualRLAgent, **kwargs):
        agent.model.get_action = self.get_action
        super().__init__(model=agent.model, optimizer=agent.optimizer, **kwargs)
        self.tella_agent = agent

    def update(self, rollouts: List[Rollout]):
        self.loss = self.model(torch.rand(4)).mean()

    def sample_rollout_action(self, observations: torch.Tensor) -> np.ndarray:
        return torch.tensor(
            self.tella_agent.choose_actions(list(observations.detach().numpy()))
        )

    def get_action(self, obs, task_label):
        return torch.tensor(self.tella_agent.choose_actions(list(obs.detach().numpy())))


def main():
    scenario = gym_benchmark_generator(
        ["CartPole-v1"], n_parallel_envs=1, eval_envs=["CartPole-v1"], n_experiences=1
    )

    env = gym.make("CartPole-v1")
    agent = TellaRandomAgent(0, env.observation_space, env.action_space, 1, None)
    strategy = TellaToAvalancheWrapper(
        agent,
        per_experience_steps=100,
        eval_every=100,
        eval_episodes=10,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    for experience in scenario.train_stream:
        print("Start of experience ", experience.current_experience)
        print("Current Env ", experience.env)
        print("Current Task", experience.task_label, type(experience.task_label))
        results.append(strategy.train(experience, scenario.test_stream))

    print("Training completed")
    eval_episodes = 100
    print(f"\nEvaluating on {eval_episodes} episodes!")
    print(strategy.eval(scenario.test_stream))


if __name__ == "__main__":
    main()
