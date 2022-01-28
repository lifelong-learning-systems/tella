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
from gym_minigrid.envs import (
    DistShift1,
    DistShift2,
    DoorKeyEnv,
    DoorKeyEnv5x5,
    DoorKeyEnv6x6,
    SimpleCrossingEnv,
    SimpleCrossingS9N2Env,
    SimpleCrossingS9N3Env,
)
from gym_minigrid.wrappers import ImgObsWrapper

from ...curriculum import (
    InterleavedEvalCurriculum,
    AbstractLearnBlock,
    LearnBlock,
    AbstractEvalBlock,
    simple_eval_block,
    TaskBlock,
    AbstractRLTaskVariant,
    EpisodicTaskVariant,
)
from .envs import (
    CustomDynamicObstaclesS6N1,
    CustomDynamicObstaclesS8N2,
    CustomDynamicObstaclesS10N3,
    CustomFetchEnv5x5T1N2,
    CustomFetchEnv8x8T1N2,
    CustomFetchEnv16x16T2N4,
    CustomUnlock5x5,
    CustomUnlock7x7,
    CustomUnlock9x9,
    DistShift3,
)


class MiniGridReducedActionSpaceWrapper(gym.ActionWrapper):
    """Reduce the action space in environment to help learning."""

    def __init__(self, env: gym.Env, num_actions: int = 6) -> None:
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert num_actions <= env.action_space.n
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(num_actions)

    def action(self, act):
        if act >= self.action_space.n:
            raise gym.error.InvalidAction()
        return act


class MiniGridLavaPenaltyWrapper(gym.Wrapper):
    """Penalize agent for stepping in lava."""

    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        # Check if there is lava in front of the agent
        front_cell = self.env.grid.get(*self.env.front_pos)
        not_clear = front_cell and front_cell.type == "lava"

        # Update the agent's position/direction
        obs, reward, done, info = self.env.step(action)

        # If the agent tried to walk over lava
        if action == self.env.actions.forward and not_clear:
            reward = -1
            done = True
            return obs, reward, done, info

        return obs, reward, done, info


class _MiniGridEnv(gym.Wrapper):
    def __init__(self, env_class: typing.Type[gym.Env]) -> None:
        super().__init__(env_class())
        self.env = ImgObsWrapper(self.env)
        self.env = MiniGridReducedActionSpaceWrapper(self.env)


class _MiniGridLavaEnv(_MiniGridEnv):
    def __init__(self, env_class: typing.Type[gym.Env]) -> None:
        super().__init__(env_class)
        self.env = MiniGridLavaPenaltyWrapper(self.env)


def _wrap_minigrid_env(
    env_class: typing.Type[gym.Env], wrapper_class: typing.Type[_MiniGridEnv]
):
    class _WrappedEnv(wrapper_class):
        def __init__(self):
            super().__init__(env_class)

    return _WrappedEnv


SimpleCrossingS9N1 = _wrap_minigrid_env(SimpleCrossingEnv, _MiniGridEnv)
SimpleCrossingS9N2 = _wrap_minigrid_env(SimpleCrossingS9N2Env, _MiniGridEnv)
SimpleCrossingS9N3 = _wrap_minigrid_env(SimpleCrossingS9N3Env, _MiniGridEnv)
DistShiftR2 = _wrap_minigrid_env(DistShift1, _MiniGridLavaEnv)
DistShiftR5 = _wrap_minigrid_env(DistShift2, _MiniGridLavaEnv)
DistShiftR3 = _wrap_minigrid_env(DistShift3, _MiniGridLavaEnv)
DynObstaclesS6N1 = _wrap_minigrid_env(CustomDynamicObstaclesS6N1, _MiniGridEnv)
DynObstaclesS8N2 = _wrap_minigrid_env(CustomDynamicObstaclesS8N2, _MiniGridEnv)
DynObstaclesS10N3 = _wrap_minigrid_env(CustomDynamicObstaclesS10N3, _MiniGridEnv)
CustomFetchS5T1N2 = _wrap_minigrid_env(CustomFetchEnv5x5T1N2, _MiniGridEnv)
CustomFetchS8T1N2 = _wrap_minigrid_env(CustomFetchEnv8x8T1N2, _MiniGridEnv)
CustomFetchS16T2N4 = _wrap_minigrid_env(CustomFetchEnv16x16T2N4, _MiniGridEnv)
CustomUnlockS5 = _wrap_minigrid_env(CustomUnlock5x5, _MiniGridEnv)
CustomUnlockS7 = _wrap_minigrid_env(CustomUnlock7x7, _MiniGridEnv)
CustomUnlockS9 = _wrap_minigrid_env(CustomUnlock9x9, _MiniGridEnv)
DoorKeyS5 = _wrap_minigrid_env(DoorKeyEnv5x5, _MiniGridEnv)
DoorKeyS6 = _wrap_minigrid_env(DoorKeyEnv6x6, _MiniGridEnv)
DoorKeyS8 = _wrap_minigrid_env(DoorKeyEnv, _MiniGridEnv)


TASKS = [
    (SimpleCrossingS9N1, "SimpleCrossing", "S9N1"),
    (SimpleCrossingS9N2, "SimpleCrossing", "S9N2"),
    (SimpleCrossingS9N3, "SimpleCrossing", "S9N3"),
    (DistShiftR2, "DistShift", "R2"),
    (DistShiftR5, "DistShift", "R5"),
    (DistShiftR3, "DistShift", "R3"),
    (DynObstaclesS6N1, "DynObstacles", "S6N1"),
    (DynObstaclesS8N2, "DynObstacles", "S8N2"),
    (DynObstaclesS10N3, "DynObstacles", "S10N3"),
    (CustomFetchS5T1N2, "CustomFetch", "S5T1N2"),
    (CustomFetchS8T1N2, "CustomFetch", "S8T1N2"),
    (CustomFetchS16T2N4, "CustomFetch", "S16T2N4"),
    (CustomUnlockS5, "CustomUnlock", "S5"),
    (CustomUnlockS7, "CustomUnlock", "S7"),
    (CustomUnlockS9, "CustomUnlock", "S9"),
    (DoorKeyS5, "DoorKey", "S5"),
    (DoorKeyS6, "DoorKey", "S6"),
    (DoorKeyS8, "DoorKey", "S8"),
]


class _MiniGridCurriculum(InterleavedEvalCurriculum[AbstractRLTaskVariant]):
    def eval_block(self) -> AbstractEvalBlock[AbstractRLTaskVariant]:
        rng = np.random.default_rng(self.eval_rng_seed)
        return simple_eval_block(
            EpisodicTaskVariant(
                cls,
                task_label=task_label,
                variant_label=variant_label,
                num_episodes=100,
                rng_seed=rng.bit_generator.random_raw(),
            )
            for cls, task_label, variant_label in TASKS
        )


class MiniGridCondensed(_MiniGridCurriculum):
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
                                num_episodes=1000,
                                rng_seed=self.rng.bit_generator.random_raw(),
                            )
                        ],
                    )
                ]
            )


class MiniGridDispersed(_MiniGridCurriculum):
    def __init__(self, rng_seed: int, num_repetitions: int = 3):
        super().__init__(rng_seed)
        self.num_repetitions = num_repetitions

    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        for _ in range(self.num_repetitions):
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
                                    num_episodes=1000 // self.num_repetitions,
                                    rng_seed=self.rng.bit_generator.random_raw(),
                                )
                            ],
                        )
                    ]
                )


def _make_single_learning_task_minigrid_curriculum(
    task_cls: typing.Type[gym.Env],
    task_label: str,
    variant_label: str,
):
    class _NewCurriculum(_MiniGridCurriculum):
        def learn_blocks(
            self,
        ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
            yield LearnBlock(
                [
                    TaskBlock(
                        task_label,
                        [
                            EpisodicTaskVariant(
                                task_cls,
                                task_label=task_label,
                                variant_label=variant_label,
                                num_episodes=1000,
                                rng_seed=self.rng.bit_generator.random_raw(),
                            )
                        ],
                    )
                ]
            )

    return _NewCurriculum


MiniGridSimpleCrossingS9N1 = _make_single_learning_task_minigrid_curriculum(
    SimpleCrossingS9N1, "SimpleCrossing", "S9N1"
)
MiniGridSimpleCrossingS9N2 = _make_single_learning_task_minigrid_curriculum(
    SimpleCrossingS9N2, "SimpleCrossing", "S9N2"
)
MiniGridSimpleCrossingS9N3 = _make_single_learning_task_minigrid_curriculum(
    SimpleCrossingS9N3, "SimpleCrossing", "S9N3"
)
MiniGridDistShiftR2 = _make_single_learning_task_minigrid_curriculum(
    DistShiftR2, "DistShift", "R2"
)
MiniGridDistShiftR5 = _make_single_learning_task_minigrid_curriculum(
    DistShiftR5, "DistShift", "R5"
)
MiniGridDistShiftR3 = _make_single_learning_task_minigrid_curriculum(
    DistShiftR3, "DistShift", "R3"
)
MiniGridDynObstaclesS6N1 = _make_single_learning_task_minigrid_curriculum(
    DynObstaclesS6N1, "DynObstacles", "S6N1"
)
MiniGridDynObstaclesS8N2 = _make_single_learning_task_minigrid_curriculum(
    DynObstaclesS8N2, "DynObstacles", "S8N2"
)
MiniGridDynObstaclesS10N3 = _make_single_learning_task_minigrid_curriculum(
    DynObstaclesS10N3, "DynObstacles", "S10N3"
)
MiniGridCustomFetchS5T1N2 = _make_single_learning_task_minigrid_curriculum(
    CustomFetchS5T1N2, "CustomFetch", "S5T1N2"
)
MiniGridCustomFetchS8T1N2 = _make_single_learning_task_minigrid_curriculum(
    CustomFetchS8T1N2, "CustomFetch", "S8T1N2"
)
MiniGridCustomFetchS16T2N4 = _make_single_learning_task_minigrid_curriculum(
    CustomFetchS16T2N4, "CustomFetch", "S16T2N4"
)
MiniGridCustomUnlockS5 = _make_single_learning_task_minigrid_curriculum(
    CustomUnlockS5, "CustomUnlock", "S5"
)
MiniGridCustomUnlockS7 = _make_single_learning_task_minigrid_curriculum(
    CustomUnlockS7, "CustomUnlock", "S7"
)
MiniGridCustomUnlockS9 = _make_single_learning_task_minigrid_curriculum(
    CustomUnlockS9, "CustomUnlock", "S9"
)
MiniGridDoorKeyS5 = _make_single_learning_task_minigrid_curriculum(
    DoorKeyS5, "DoorKey", "S5"
)
MiniGridDoorKeyS6 = _make_single_learning_task_minigrid_curriculum(
    DoorKeyS6, "DoorKey", "S6"
)
MiniGridDoorKeyS8 = _make_single_learning_task_minigrid_curriculum(
    DoorKeyS8, "DoorKey", "S8"
)
