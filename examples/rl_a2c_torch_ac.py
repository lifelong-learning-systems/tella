import logging
import random
import typing
from functools import reduce
from operator import imul

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from owvae.replay import Experience_Replay
# from owvae.wake_model import SimpleMinigridAgent

import torch_ac
from torch_ac.utils import DictList
import os, sys

sys.path.append("rl-starter-files")
import utils
import model

import tella
from tella.curriculum import (
    AbstractCurriculum,
    AbstractRLTaskVariant,
    EpisodicTaskVariant,
    simple_learn_block,
    simple_eval_block,
)


logger = logging.getLogger("Example A2C Agent")


class Object(object):
    def reset(self):
        pass


# Hyperparameters
sample_size = 8
buffer_limit = 10000
discount_factor = 0.98
optimizer_params = {}
optimizer_params["learning_rate"] = 0.01
optimizer_params["weight_decay"] = 0.0
optimizer_params["decay_steps"] = 100000
optimizer_params["decay_gamma"] = 1.0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


class MinimalRlA2CAgent(tella.ContinualRLAgent):
    def __init__(
        self,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: typing.Optional[str] = None,
    ) -> None:
        super(MinimalRlA2CAgent, self).__init__(
            rng_seed, observation_space, action_space, num_envs, config_file
        )

        assert isinstance(
            action_space, gym.spaces.Discrete
        ), "This A2C agent requires discrete action spaces"

        logger.info(
            f"Constructed with observation_space={observation_space} "
            f"action_space={action_space} num_envs={num_envs}"
        )

        logger.info(f"RNG seed provided: {rng_seed}")
        torch.manual_seed(rng_seed)

        # Set the input and output dimensions based on observation and action spaces
        input_dim = reduce(imul, observation_space.shape)
        output_dim = action_space.n
        layer_dims = (input_dim, 128, output_dim)

        # self.wake_model = SimpleMinigridAgent(layer_dims, optimizer_params, discount_factor=discount_factor)

        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(observation_space)
        self.wake_model = model.ACModel(obs_space, action_space, True, False)
        fake_env = Object()
        fake_env.observation_space = observation_space
        fake_env.action_space = action_space

        self.a2c_algo = torch_ac.A2CAlgo(
            [fake_env],
            self.wake_model,
            device,
            12,
            0.99,
            0.001,
            0.95,
            0.01,
            0.5,
            0.5,
            3,
            0.99,
            1e-8,
            self.preprocess_obss,
        )
        # self.a2c_memory = Experience_Replay(buffer_size=buffer_limit, sample_size=sample_size)
        self.current_frame_count = 0
        self.training = None
        self.num_eps_done = 0
        import pdb

        pdb.set_trace()

    def block_start(self, is_learning_allowed: bool) -> None:
        super().block_start(is_learning_allowed)
        if is_learning_allowed:
            logger.info("About to start a new learning block")
            self.training = True
        else:
            logger.info("About to start a new evaluation block")
            self.training = False

    def task_start(self, task_name: typing.Optional[str]) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task. task_name={task_name}"
        )

    def task_variant_start(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tAbout to start interacting with a new task variant. "
            f"task_name={task_name} variant_name={variant_name}"
        )

    def learn_task_variant(self, task_variant: tella.AbstractRLTaskVariant) -> None:
        logger.info("\tConsuming task variant")
        return super().learn_task_variant(task_variant)

    def choose_actions(
        self, observations: typing.List[typing.Optional[tella.Observation]]
    ) -> typing.List[typing.Optional[tella.Action]]:
        logger.debug(f"\t\t\tReturn {len(observations)} actions")

        # process current observation to receive action
        preprocessed_obs = self.a2c_algo.preprocess_obss(
            observations, device=self.a2c_algo.device
        )
        with torch.no_grad():
            if self.a2c_algo.acmodel.recurrent:
                dist, value, memory = self.a2c_algo.acmodel(
                    preprocessed_obs,
                    self.a2c_algo.memory * self.a2c_algo.mask.unsqueeze(1),
                )
                self.a2c_algo.dist = dist
                self.a2c_algo.value = value
                self.a2c_algo.memory = memory
            else:
                dist, value = self.a2c_algo.acmodel(preprocessed_obs)
                self.a2c_algo.dist = dist
                self.a2c_algo.value = value
        # with torch.no_grad():
        #     dist, value = self.a2c_algo.acmodel(preprocessed_obs)
        #     self.a2c_algo.dist = dist
        #     self.a2c_algo.value = value

        self.a2c_algo.action = dist.sample()

        return [self.a2c_algo.action]

    def receive_transitions(
        self, transitions: typing.List[typing.Optional[tella.Transition]]
    ):
        if not self.is_learning_allowed:
            return
        for transition in transitions:
            if transition is not None:
                self.receive_transition(transition)

    def receive_transition(self, transition: tella.Transition):
        prev_obs, action, reward, done, obs = transition

        self.a2c_algo.obss[self.current_frame_count] = prev_obs
        self.a2c_algo.obs = obs
        if self.a2c_algo.acmodel.recurrent:
            self.a2c_algo.memories[self.current_frame_count] = self.a2c_algo.memory
        self.a2c_algo.masks[self.current_frame_count] = self.a2c_algo.mask
        self.a2c_algo.mask = 1 - torch.tensor(
            [done], device=self.a2c_algo.device, dtype=torch.float
        )
        self.a2c_algo.actions[self.current_frame_count] = self.a2c_algo.action
        self.a2c_algo.values[self.current_frame_count] = self.a2c_algo.value
        if self.a2c_algo.reshape_reward is not None:
            self.a2c_algo.rewards[self.current_frame_count] = torch.tensor(
                [
                    self.a2c_algo.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ],
                device=self.device,
            )
        else:
            self.a2c_algo.rewards[self.current_frame_count] = torch.tensor(
                reward, device=self.a2c_algo.device
            )
        self.a2c_algo.log_probs[self.current_frame_count] = self.a2c_algo.dist.log_prob(
            action
        )

        # Update log values

        self.a2c_algo.log_episode_return += torch.tensor(
            reward, device=self.a2c_algo.device, dtype=torch.float
        )
        self.a2c_algo.log_episode_reshaped_return += self.a2c_algo.rewards[
            self.current_frame_count
        ]
        self.a2c_algo.log_episode_num_frames += torch.ones(
            self.a2c_algo.num_procs, device=self.a2c_algo.device
        )

        # if done:
        #     self.a2c_algo.log_done_counter += 1
        #     self.a2c_algo.log_return.append(self.a2c_algo.log_episode_return[self.current_frame_count].item())
        #     self.a2c_algo.log_reshaped_return.append(self.a2c_algo.log_episode_reshaped_return[self.current_frame_count].item())
        #     self.a2c_algo.log_num_frames.append(self.a2c_algo.log_episode_num_frames[self.current_frame_count].item())
        #
        # self.a2c_algo.log_episode_return *= self.a2c_algo.mask
        # self.a2c_algo.log_episode_reshaped_return *= self.a2c_algo.mask
        # self.a2c_algo.log_episode_num_frames *= self.a2c_algo.mask

        if self.current_frame_count == self.a2c_algo.num_frames_per_proc - 1:
            # Add advantage and return to experiences

            preprocessed_obs = self.preprocess_obss(
                [self.a2c_algo.obs], device=self.a2c_algo.device
            )
            with torch.no_grad():
                if self.a2c_algo.acmodel.recurrent:
                    _, next_value, _ = self.a2c_algo.acmodel(
                        preprocessed_obs,
                        self.a2c_algo.memory * self.a2c_algo.mask.unsqueeze(1),
                    )
                else:
                    _, next_value = self.a2c_algo.acmodel(preprocessed_obs)
            # with torch.no_grad():
            #     _, next_value = self.a2c_algo.acmodel(preprocessed_obs)

            for i in reversed(range(self.a2c_algo.num_frames_per_proc)):
                next_mask = (
                    self.a2c_algo.masks[i + 1]
                    if i < self.a2c_algo.num_frames_per_proc - 1
                    else self.a2c_algo.mask
                )
                next_value = (
                    self.a2c_algo.values[i + 1]
                    if i < self.a2c_algo.num_frames_per_proc - 1
                    else next_value
                )
                next_advantage = (
                    self.a2c_algo.advantages[i + 1]
                    if i < self.a2c_algo.num_frames_per_proc - 1
                    else 0
                )

                delta = (
                    self.a2c_algo.rewards[i]
                    + self.a2c_algo.discount * next_value * next_mask
                    - self.a2c_algo.values[i]
                )
                self.a2c_algo.advantages[i] = (
                    delta
                    + self.a2c_algo.discount
                    * self.a2c_algo.gae_lambda
                    * next_advantage
                    * next_mask
                )

            # Define experiences:
            #   the whole experience is the concatenation of the experience
            #   of each process.
            # In comments below:
            #   - T is self.num_frames_per_proc,
            #   - P is self.num_procs,
            #   - D is the dimensionality.

            exps = DictList()
            exps.obs = self.a2c_algo.obss
            if self.a2c_algo.acmodel.recurrent:
                # T x P x D -> P x T x D -> (P * T) x D
                exps.memory = self.a2c_algo.memories.transpose(0, 1).reshape(
                    -1, *self.a2c_algo.memories.shape[2:]
                )
                # T x P -> P x T -> (P * T) x 1
                exps.mask = self.a2c_algo.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
            # for all tensors below, T x P -> P x T -> P * T
            exps.action = self.a2c_algo.actions.transpose(0, 1).reshape(-1)
            exps.value = self.a2c_algo.values.transpose(0, 1).reshape(-1)
            exps.reward = self.a2c_algo.rewards.transpose(0, 1).reshape(-1)
            exps.advantage = self.a2c_algo.advantages.transpose(0, 1).reshape(-1)
            exps.returnn = exps.value + exps.advantage
            exps.log_prob = self.a2c_algo.log_probs.transpose(0, 1).reshape(-1)

            # Preprocess experiences

            exps.obs = self.preprocess_obss(exps.obs, device=self.a2c_algo.device)

            logger.debug(f"Training Model")
            self.a2c_algo.update_parameters(exps)

            self.current_frame_count = 0

        # exps, logs1 = self.a2c_algo.collect_experiences()

        # self.a2c_memory.update(
        #    (s.flatten(), a, r / 100.0, s_prime.flatten(), 0.0 if done else 1.0)
        # )

        else:

            self.current_frame_count += 1

        logger.debug(f"\t\t\tReceived transition done={done}")

        # Handle end-of-episode matters: training, logging, and annealing
        if done:
            self.num_eps_done += 1

            logger.info(
                f"\t\t"
                f"n_episode: {self.num_eps_done}, "
                # f"score: {self.metric.calculate()['MeanEpisodeReward']:.1f}, "
                # f"n_buffer: {self.a2c_memory.size()}%"
            )

    def task_variant_end(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        logger.info(
            f"\tDone interacting with task variant. "
            f"task_name={task_name} variant_name={variant_name}"
        )

    def task_end(self, task_name: typing.Optional[str]) -> None:
        logger.info(f"\tDone interacting with task. task_name={task_name}")

    def block_end(self, is_learning_allowed: bool) -> None:
        if is_learning_allowed:
            logger.info("Done with learning block")
        else:
            logger.info("Done with evaluation block")


class EasySimpleCrossingCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    tella._curriculums.minigrid.simple.EasySimpleCrossing,
                    num_episodes=1_000,
                    task_label="EasySimpleCrossing",
                    variant_label="Default",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )
        yield simple_eval_block(
            [
                EpisodicTaskVariant(
                    tella._curriculums.minigrid.simple.EasySimpleCrossing,
                    num_episodes=1,
                    task_label="EasySimpleCrossing",
                    variant_label="Default",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


class EasyDistShiftCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    tella._curriculums.minigrid.simple.EasyDistShift1,
                    num_episodes=1_000,
                    task_label="EasyDistShift",
                    variant_label="Default",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )
        yield simple_eval_block(
            [
                EpisodicTaskVariant(
                    tella._curriculums.minigrid.simple.EasyDistShift1,
                    num_episodes=1,
                    task_label="EasyDistShift",
                    variant_label="Default",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


class EasyDynamicObstaclesCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    tella._curriculums.minigrid.simple.EasyDynamicObstacles,
                    num_episodes=1_000,
                    task_label="EasyDynamicObstacles",
                    variant_label="Default",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )
        yield simple_eval_block(
            [
                EpisodicTaskVariant(
                    tella._curriculums.minigrid.simple.EasyDynamicObstacles,
                    num_episodes=1,
                    task_label="EasyDynamicObstacles",
                    variant_label="Default",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tella._curriculums.curriculum_registry[
        "EasySimpleCrossing"
    ] = EasySimpleCrossingCurriculum
    tella._curriculums.curriculum_registry["EasyDistShift"] = EasyDistShiftCurriculum
    tella._curriculums.curriculum_registry[
        "EasyDynamicObstacles"
    ] = EasyDynamicObstaclesCurriculum
    tella.rl_cli(MinimalRlA2CAgent)
