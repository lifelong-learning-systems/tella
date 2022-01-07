"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

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
import gym
import typing
import numpy as np
from tella.curriculum import *
from gym_minigrid.envs import DistShift1, DynamicObstaclesEnv, SimpleCrossingEnv
from gym_minigrid.wrappers import ImgObsWrapper, StateBonus, ActionBonus


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
        super().__init__(DynamicObstaclesEnv)


TASKS = [
    (EasySimpleCrossing, "EasySimpleCrossing", "Default"),
    (EasyDistShift1, "EasyDistShift", "Default"),
    (EasyDynamicObstacles, "EasyDynamicObstacles", "Default"),
]


class SimpleMiniGridCurriculum(InterleavedEvalCurriculum[AbstractRLTaskVariant]):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        rng = np.random.default_rng(self.rng_seed)
        for cls, task_label, variant_label in rng.permutation(TASKS):
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
                            )
                        ],
                    )
                ]
            )

    def eval_block(self) -> AbstractEvalBlock[AbstractRLTaskVariant]:
        rng = np.random.default_rng(self.rng_seed)
        return simple_eval_block(
            EpisodicTaskVariant(
                cls,
                task_label=task_label,
                variant_label=variant_label,
                num_episodes=5,
            )
            for cls, task_label, variant_label in rng.permutation(TASKS)
        )
