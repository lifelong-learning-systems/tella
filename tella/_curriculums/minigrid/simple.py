"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import typing

import gym
import numpy as np
from gym_minigrid.envs import DistShift1, SimpleCrossingEnv
from gym_minigrid.wrappers import ImgObsWrapper, StateBonus, ActionBonus

from .envs import CustomDynamicObstaclesS6N1
from ...curriculum import *


class RestrictedActions(gym.Wrapper):
    def __init__(self, env: gym.Env, num_actions: int) -> None:
        super().__init__(env)
        assert isinstance(self.action_space, gym.spaces.Discrete)
        self.action_space = gym.spaces.Discrete(num_actions)


class _EasyMiniGrid(gym.Wrapper):
    def __init__(self, env_class: typing.Type[gym.Env]) -> None:
        env = env_class()
        env = ImgObsWrapper(env)
        env = StateBonus(env)
        env = ActionBonus(env)
        env = RestrictedActions(env, num_actions=3)
        super().__init__(env)


class EasySimpleCrossing(_EasyMiniGrid):
    def __init__(self):
        super().__init__(SimpleCrossingEnv)


class EasyDistShift1(_EasyMiniGrid):
    def __init__(self):
        super().__init__(DistShift1)


class EasyDynamicObstacles(_EasyMiniGrid):
    def __init__(self):
        super().__init__(CustomDynamicObstaclesS6N1)


TASKS = [
    (EasySimpleCrossing, "EasySimpleCrossing", "Default"),
    (EasyDistShift1, "EasyDistShift", "Default"),
    (EasyDynamicObstacles, "EasyDynamicObstacles", "Default"),
]


class SimpleMiniGridCurriculum(InterleavedEvalCurriculum[AbstractRLTaskVariant]):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        for cls, task_label, variant_label in self.rng.permutation(TASKS):
            yield LearnBlock(
                [
                    TaskBlock(
                        task_label,
                        [
                            EpisodicTaskVariant(
                                cls,
                                task_label=task_label,
                                variant_label=variant_label,
                                num_episodes=5,
                                rng_seed=self.rng.bit_generator.random_raw(),
                            )
                        ],
                    )
                ]
            )

    def eval_block(self) -> AbstractEvalBlock[AbstractRLTaskVariant]:
        return simple_eval_block(
            EpisodicTaskVariant(
                cls,
                task_label=task_label,
                variant_label=variant_label,
                num_episodes=5,
                rng_seed=self.eval_rng_seed,
            )
            for cls, task_label, variant_label in TASKS
        )


class SimpleStepMiniGridCurriculum(SimpleMiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        for cls, task_label, variant_label in self.rng.permutation(TASKS):
            yield LearnBlock(
                [
                    TaskBlock(
                        task_label,
                        [
                            EpisodicTaskVariant(
                                cls,
                                task_label=task_label,
                                variant_label=variant_label,
                                num_episodes=5,
                                rng_seed=self.rng.bit_generator.random_raw(),
                                step_limit=5000,
                            )
                        ],
                    )
                ]
            )
