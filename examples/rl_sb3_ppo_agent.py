import logging
import typing

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor

import tella

logger = logging.getLogger("SB3-PPO")


class Sb3PpoAgent(tella.ContinualRLAgent):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        metric: typing.Optional[tella.RLMetricAccumulator] = None,
    ) -> None:
        super(Sb3PpoAgent, self).__init__(
            observation_space, action_space, num_envs, metric
        )
        logger.info(
            f"Constructed with observation_space={observation_space} "
            f"action_space={action_space} num_envs={num_envs}"
        )
        self.sb3_ppo = PPO("MlpPolicy", "CartPole-v1")
        self.n_steps = 0

    def block_start(self, is_learning_allowed: bool) -> None:
        super().block_start(is_learning_allowed)
        logger.info(
            f"About to start a new {'learning' if is_learning_allowed else 'eval'} block"
        )
        self.sb3_ppo.policy.set_training_mode(is_learning_allowed)

    def task_start(self, task_name: typing.Optional[str]) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task. task_name={task_name}"
        )

    def learn_task_variant(
        self, task_variant: tella.AbstractRLTaskVariant
    ) -> tella.Metrics:
        logger.info("\tConsuming task variant")

        # Some setup code from stable-baselines3 on_policy_algorithm.py
        self.n_steps = 0
        self.sb3_ppo.rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.sb3_ppo.use_sde:
            self.sb3_ppo.policy.reset_noise(self.num_envs)

        metrics = super().learn_task_variant(task_variant)

        self.sb3_ppo.rollout_buffer.compute_returns_and_advantage()  # TODO
        self.sb3_ppo.train()
        self.sb3_ppo.policy.set_training_mode(
            False
        )  # Was set to True in .train() method

        return metrics

    def choose_actions(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        logger.debug(f"\t\t\tReturn {len(observations)} actions")

        if (
            self.sb3_ppo.use_sde
            and self.sb3_ppo.sde_sample_freq > 0
            and self.n_steps % self.sb3_ppo.sde_sample_freq == 0
        ):
            # Sample a new noise matrix
            self.sb3_ppo.policy.reset_noise(self.num_envs)

        obs_array = np.array([obs for obs in observations if obs is not None])

        with torch.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(obs_array, self.sb3_ppo.device)
            actions, values, log_probs = self.sb3_ppo.policy.forward(obs_tensor)
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )
        clipped_actions = clipped_actions.tolist()

        return_actions = []
        for obs in observations:
            if obs is None:
                return_actions.append(None)
            else:
                return_actions.append(clipped_actions.pop(0))

        self.n_steps += 1

        return return_actions

    def receive_transitions(self, step: tella.Transition):
        s, a, r, done, s_prime = step
        self.sb3_ppo.rollout_buffer.add(
            s,
            a,
            r,
        )  # TODO

    def task_end(self, task_name: typing.Optional[str]) -> None:
        logger.info(f"\tDone interacting with task. task_name={task_name}")

    def block_end(self, is_learning_allowed: bool) -> None:
        if is_learning_allowed:
            logger.info("Done with learning block")
        else:
            logger.info("Done with evaluation block")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella.rl_cli(Sb3PpoAgent)
