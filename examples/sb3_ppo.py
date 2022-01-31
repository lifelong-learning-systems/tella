import logging
import typing
import gym
import tella
from stable_baselines3 import PPO
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import is_image_space
import numpy as np
import torch

logger = logging.getLogger("SB3 PPO")


class _DummyEnv(gym.Env):
    def __init__(self, observation_space, action_space) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.observations = [self.observation_space.sample()]

    def reset(self):
        # NOTE: should only be called once during PPO._setup_learn()
        return self.observations.pop()


def transpose_image(image: np.ndarray) -> np.ndarray:
    """
    Transpose an image or batch of images (re-order channels).

    :param image:
    :return:
    """
    if len(image.shape) == 3:
        return np.transpose(image, (2, 0, 1))
    return np.transpose(image, (0, 3, 1, 2))


class SB3PPOAgent(tella.ContinualRLAgent):
    def __init__(
        self,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: typing.Optional[str] = None,
    ) -> None:
        super().__init__(
            rng_seed, observation_space, action_space, num_envs, config_file
        )
        self.obs_is_image = is_image_space(observation_space)

        # TODO what are the hyperparameters here?
        self.ppo = PPO("MlpPolicy", _DummyEnv(observation_space, action_space))
        self.ppo._setup_learn(np.inf, eval_env=None)

        self.steps_since_last_train = 0
        self.last_dones = [False] * self.num_envs

    def task_variant_start(
        self, task_name: typing.Optional[str], variant_name: typing.Optional[str]
    ) -> None:
        self.last_dones = [False] * self.num_envs
        self.ppo.rollout_buffer.reset()
        self.steps_since_last_train = 0
        if self.ppo.use_sde:
            self.ppo.policy.reset_noise(self.num_envs)

    def choose_actions(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        self._last_obs = np.array(observations)
        if self.obs_is_image:
            self._last_obs = transpose_image(self._last_obs)

        if (
            self.is_learning_allowed
            and self.ppo.use_sde
            and self.ppo.sde_sample_freq > 0
            and self.steps_since_last_train % self.sde_sample_freq == 0
        ):
            # Sample a new noise matrix
            self.ppo.policy.reset_noise(self.num_envs)

        with torch.no_grad():
            # Convert to pytorch tensor or to TensorDict
            obs_tensor = obs_as_tensor(self._last_obs, self.ppo.device)
            actions, self.last_values, self.last_log_probs = self.ppo.policy.forward(
                obs_tensor
            )
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions

        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )

        return clipped_actions

    def receive_transitions(
        self, transitions: typing.List[typing.Optional[tella.Transition]]
    ) -> None:
        if not self.is_learning_allowed:
            return

        assert all(t is not None for t in transitions)

        self.steps_since_last_train += 1

        obss, actions, rewards, dones, next_obss = list(
            map(np.array, zip(*transitions))
        )

        if self.obs_is_image:
            obss = transpose_image(obss)
            next_obss = transpose_image(next_obss)

        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        for idx, done in enumerate(dones):
            if done:
                # FIXME: this doesn't account for time limits like original SB3 code
                terminal_obs = self.ppo.policy.obs_to_tensor(next_obss[idx])[0]
                with torch.no_grad():
                    terminal_value = self.ppo.policy.predict_values(terminal_obs)[0]
                rewards[idx] += self.ppo.gamma * terminal_value

        self.ppo.rollout_buffer.add(
            obss,
            actions,
            rewards,
            self.last_dones,
            self.last_values,
            self.last_log_probs,
        )
        self.last_dones = dones

        if self.steps_since_last_train >= self.ppo.n_steps:
            with torch.no_grad():
                # Compute value for the last timestep
                values = self.ppo.policy.predict_values(
                    obs_as_tensor(next_obss, self.ppo.device)
                )

            self.ppo.rollout_buffer.compute_returns_and_advantage(
                last_values=values, dones=dones
            )

            self.ppo.train()

            self.steps_since_last_train = 0
            self.ppo.rollout_buffer.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella.rl_cli(SB3PPOAgent)
