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


logger = logging.getLogger("Example PPO Agent")


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


class MinimalRlPPOAgent(tella.ContinualRLAgent):
    def __init__(
        self,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: typing.Optional[str] = None,
    ) -> None:
        super(MinimalRlPPOAgent, self).__init__(
            rng_seed, observation_space, action_space, num_envs, config_file
        )

        assert isinstance(
            action_space, gym.spaces.Discrete
        ), "This PPO agent requires discrete action spaces"

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

        self.ppo_algo = torch_ac.PPOAlgo(
            envs=[fake_env],
            acmodel=self.wake_model,
            device=device,
            num_frames_per_proc=12,
            discount=0.99,
            lr=0.00034, # not default
            gae_lambda=1.0, # not default
            entropy_coef=9.5e-8, # not default
            value_loss_coef=0.85, # not default
            max_grad_norm=0.6, # not default
            recurrence=4,
            # adam_eps=1e-8,
            # clip_eps=0.2,
            epochs=20, # not default
            batch_size=128, # not default
            preprocess_obss=self.preprocess_obss,
        )
        # self.ppo_memory = Experience_Replay(buffer_size=buffer_limit, sample_size=sample_size)
        self.current_frame_count = 0
        self.training = None
        self.num_eps_done = 0
        # import pdb

        # pdb.set_trace()

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
        preprocessed_obs = self.ppo_algo.preprocess_obss(
            observations, device=self.ppo_algo.device
        )
        with torch.no_grad():
            if self.ppo_algo.acmodel.recurrent:
                dist, value, memory = self.ppo_algo.acmodel(
                    preprocessed_obs,
                    self.ppo_algo.memory * self.ppo_algo.mask.unsqueeze(1),
                )
                self.ppo_algo.dist = dist
                self.ppo_algo.value = value
                self.ppo_algo.memory = memory
            else:
                dist, value = self.ppo_algo.acmodel(preprocessed_obs)
                self.ppo_algo.dist = dist
                self.ppo_algo.value = value
        # with torch.no_grad():
        #     dist, value = self.ppo_algo.acmodel(preprocessed_obs)
        #     self.ppo_algo.dist = dist
        #     self.ppo_algo.value = value

        self.ppo_algo.action = dist.sample()

        return [self.ppo_algo.action]

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

        self.ppo_algo.obss[self.current_frame_count] = prev_obs
        self.ppo_algo.obs = obs
        if self.ppo_algo.acmodel.recurrent:
            self.ppo_algo.memories[self.current_frame_count] = self.ppo_algo.memory
        self.ppo_algo.masks[self.current_frame_count] = self.ppo_algo.mask
        self.ppo_algo.mask = 1 - torch.tensor(
            [done], device=self.ppo_algo.device, dtype=torch.float
        )
        self.ppo_algo.actions[self.current_frame_count] = self.ppo_algo.action
        self.ppo_algo.values[self.current_frame_count] = self.ppo_algo.value
        if self.ppo_algo.reshape_reward is not None:
            self.ppo_algo.rewards[self.current_frame_count] = torch.tensor(
                [
                    self.ppo_algo.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ],
                device=self.device,
            )
        else:
            self.ppo_algo.rewards[self.current_frame_count] = torch.tensor(
                reward, device=self.ppo_algo.device
            )
        self.ppo_algo.log_probs[self.current_frame_count] = self.ppo_algo.dist.log_prob(
            action
        )

        # Update log values

        self.ppo_algo.log_episode_return += torch.tensor(
            reward, device=self.ppo_algo.device, dtype=torch.float
        )
        self.ppo_algo.log_episode_reshaped_return += self.ppo_algo.rewards[
            self.current_frame_count
        ]
        self.ppo_algo.log_episode_num_frames += torch.ones(
            self.ppo_algo.num_procs, device=self.ppo_algo.device
        )

        # if done:
        #     self.ppo_algo.log_done_counter += 1
        #     self.ppo_algo.log_return.append(self.ppo_algo.log_episode_return[self.current_frame_count].item())
        #     self.ppo_algo.log_reshaped_return.append(self.ppo_algo.log_episode_reshaped_return[self.current_frame_count].item())
        #     self.ppo_algo.log_num_frames.append(self.ppo_algo.log_episode_num_frames[self.current_frame_count].item())
        #
        # self.ppo_algo.log_episode_return *= self.ppo_algo.mask
        # self.ppo_algo.log_episode_reshaped_return *= self.ppo_algo.mask
        # self.ppo_algo.log_episode_num_frames *= self.ppo_algo.mask

        if self.current_frame_count == self.ppo_algo.num_frames_per_proc - 1:
            # Add advantage and return to experiences

            preprocessed_obs = self.preprocess_obss(
                [self.ppo_algo.obs], device=self.ppo_algo.device
            )
            with torch.no_grad():
                if self.ppo_algo.acmodel.recurrent:
                    _, next_value, _ = self.ppo_algo.acmodel(
                        preprocessed_obs,
                        self.ppo_algo.memory * self.ppo_algo.mask.unsqueeze(1),
                    )
                else:
                    _, next_value = self.ppo_algo.acmodel(preprocessed_obs)
            # with torch.no_grad():
            #     _, next_value = self.ppo_algo.acmodel(preprocessed_obs)

            for i in reversed(range(self.ppo_algo.num_frames_per_proc)):
                next_mask = (
                    self.ppo_algo.masks[i + 1]
                    if i < self.ppo_algo.num_frames_per_proc - 1
                    else self.ppo_algo.mask
                )
                next_value = (
                    self.ppo_algo.values[i + 1]
                    if i < self.ppo_algo.num_frames_per_proc - 1
                    else next_value
                )
                next_advantage = (
                    self.ppo_algo.advantages[i + 1]
                    if i < self.ppo_algo.num_frames_per_proc - 1
                    else 0
                )

                delta = (
                    self.ppo_algo.rewards[i]
                    + self.ppo_algo.discount * next_value * next_mask
                    - self.ppo_algo.values[i]
                )
                self.ppo_algo.advantages[i] = (
                    delta
                    + self.ppo_algo.discount
                    * self.ppo_algo.gae_lambda
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
            exps.obs = self.ppo_algo.obss
            if self.ppo_algo.acmodel.recurrent:
                # T x P x D -> P x T x D -> (P * T) x D
                exps.memory = self.ppo_algo.memories.transpose(0, 1).reshape(
                    -1, *self.ppo_algo.memories.shape[2:]
                )
                # T x P -> P x T -> (P * T) x 1
                exps.mask = self.ppo_algo.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
            # for all tensors below, T x P -> P x T -> P * T
            exps.action = self.ppo_algo.actions.transpose(0, 1).reshape(-1)
            exps.value = self.ppo_algo.values.transpose(0, 1).reshape(-1)
            exps.reward = self.ppo_algo.rewards.transpose(0, 1).reshape(-1)
            exps.advantage = self.ppo_algo.advantages.transpose(0, 1).reshape(-1)
            exps.returnn = exps.value + exps.advantage
            exps.log_prob = self.ppo_algo.log_probs.transpose(0, 1).reshape(-1)

            # Preprocess experiences

            exps.obs = self.preprocess_obss(exps.obs, device=self.ppo_algo.device)

            logger.debug(f"Training Model")
            self.ppo_algo.update_parameters(exps)

            self.current_frame_count = 0

        # exps, logs1 = self.ppo_algo.collect_experiences()

        # self.ppo_memory.update(
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
                # f"n_buffer: {self.ppo_memory.size()}%"
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
    tella.rl_cli(MinimalRlPPOAgent)
