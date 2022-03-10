import logging
import typing
import gym
import tella
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import is_image_space
import numpy as np
import torch

logger = logging.getLogger("SB3 PPO")

BASE_HYPERPARAMETERS = {
    "policy": "MlpPolicy",
}


class _DummyEnv(gym.Env):
    def __init__(self, observation_space, action_space) -> None:
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.observations = [np.zeros(self.observation_space.shape)]

    def reset(self):
        # NOTE: should only be called once during PPO._setup_learn()
        return self.observations.pop()


def transpose_image(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return np.transpose(image, (2, 0, 1))
    assert len(image.shape) == 4, image.shape
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

        logging.info(f"Seeding {rng_seed}")
        # NOTE: without wrapping rng_seed get this error "ValueError: Seed must be between 0 and 2**32 - 1"
        torch.use_deterministic_algorithms(True)
        self.ppo = PPO(
            env=DummyVecEnv(
                [
                    lambda: _DummyEnv(observation_space, action_space)
                    for _ in range(num_envs)
                ]
            ),
            seed=rng_seed % (2**32),
            verbose=1,
            device="cpu",
            **BASE_HYPERPARAMETERS,
        )
        self.ppo._setup_learn(np.inf, eval_env=None)

        self.steps_since_last_train = 0
        self.last_dones = [False] * self.num_envs
        self.iteration = 0
        self.num_timesteps = 0

    def task_variant_start(
        self, task_name: typing.Optional[str], variant_name: typing.Optional[str]
    ) -> None:
        self.last_dones = [True] * self.num_envs
        self.ppo.rollout_buffer.reset()
        self.steps_since_last_train = 0
        if self.ppo.use_sde:
            self.ppo.policy.reset_noise(self.num_envs)
        self.ppo.policy.set_training_mode(False)

    def choose_actions(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        obs = np.array([o for o in observations if o is not None])
        if self.obs_is_image:
            obs = transpose_image(obs)

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
            obs_tensor = obs_as_tensor(obs, self.ppo.device)
            actions, self.last_values, self.last_log_probs = self.ppo.policy.forward(
                obs_tensor,
                deterministic=False,  # not self.is_learning_allowed
            )
        actions = actions.cpu().numpy()

        # Rescale and perform action
        clipped_actions = actions

        # Clip the actions to avoid out of bound error
        if isinstance(self.action_space, gym.spaces.Box):
            clipped_actions = np.clip(
                actions, self.action_space.low, self.action_space.high
            )

        clipped_actions = list(clipped_actions)
        return [None if o is None else clipped_actions.pop(0) for o in observations]

    def receive_transitions(
        self, transitions: typing.List[typing.Optional[tella.Transition]]
    ) -> None:
        if not self.is_learning_allowed:
            return

        if any(t is None for t in transitions):
            logger.warning(
                "Ignoring transitions with Nones. To fix this:"
                " Either ensure `task_variant.num_steps % num_envs == 0`,"
                " or turn off vectorization."
            )
            return

        self.steps_since_last_train += 1
        self.num_timesteps += len(transitions)

        obss, actions, rewards, dones, next_obss = list(
            map(np.array, zip(*transitions))
        )

        if self.obs_is_image:
            obss = transpose_image(obss)
            next_obss = transpose_image(next_obss)

        if isinstance(self.action_space, gym.spaces.Discrete):
            # Reshape in case of discrete action
            actions = actions.reshape(-1, 1)

        next_obss_t = obs_as_tensor(next_obss, self.ppo.device)
        with torch.no_grad():
            # Compute value for the last timestep
            next_obss_values = self.ppo.policy.predict_values(next_obss_t)

        for idx, done in enumerate(dones):
            # FIXME: how do we get access to truncated data from TimeLimit
            truncated = False
            if done and truncated:
                rewards[idx] += self.ppo.gamma * next_obss_values[idx].item()

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
            self.ppo.rollout_buffer.compute_returns_and_advantage(
                last_values=next_obss_values, dones=dones
            )

            self.iteration += 1

            self.ppo.logger.record("time/iterations", self.iteration)
            self.ppo.logger.record("time/total_timesteps", self.num_timesteps)
            self.ppo.logger.dump(step=self.num_timesteps)

            self.ppo.train()

            self.ppo.policy.set_training_mode(False)
            self.steps_since_last_train = 0
            self.ppo.rollout_buffer.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella.rl_cli(SB3PPOAgent)
