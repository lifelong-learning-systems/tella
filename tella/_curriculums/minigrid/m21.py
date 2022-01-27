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

from ...curriculum import *
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

    def __init__(self, env: gym.Env, num_actions: int) -> None:
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
        env = env_class()
        env = ImgObsWrapper(env)
        env = MiniGridReducedActionSpaceWrapper(env, num_actions=6)
        super().__init__(env)

    @classmethod
    def make(cls, env_class: typing.Type[gym.Env]):
        class _SubClass(cls):
            def __init__(self):
                super().__init__(env_class)
        return _SubClass


class _MiniGridLavaEnv(gym.Wrapper):
    def __init__(self, env_class: typing.Type[gym.Env]) -> None:
        env = env_class()
        env = ImgObsWrapper(env)
        env = MiniGridReducedActionSpaceWrapper(env, num_actions=6)
        env = MiniGridLavaPenaltyWrapper(env)
        super().__init__(env)

    @classmethod
    def make(cls, env_class: typing.Type[gym.Env]):
        class _SubClass(cls):
            def __init__(self):
                super().__init__(env_class)
        return _SubClass


SimpleCrossingS9N1 = _MiniGridEnv.make(SimpleCrossingEnv)
SimpleCrossingS9N2 = _MiniGridEnv.make(SimpleCrossingS9N2Env)
SimpleCrossingS9N3 = _MiniGridEnv.make(SimpleCrossingS9N3Env)
DistShiftR2 = _MiniGridLavaEnv.make(DistShift1)
DistShiftR5 = _MiniGridLavaEnv.make(DistShift2)
DistShiftR3 = _MiniGridLavaEnv.make(DistShift3)
DynObstaclesS6N1 = _MiniGridEnv.make(CustomDynamicObstaclesS6N1)
DynObstaclesS8N2 = _MiniGridEnv.make(CustomDynamicObstaclesS8N2)
DynObstaclesS10N3 = _MiniGridEnv.make(CustomDynamicObstaclesS10N3)
CustomFetchS5T1N2 = _MiniGridEnv.make(CustomFetchEnv5x5T1N2)
CustomFetchS8T1N2 = _MiniGridEnv.make(CustomFetchEnv8x8T1N2)
CustomFetchS16T2N4 = _MiniGridEnv.make(CustomFetchEnv16x16T2N4)
CustomUnlockS5 = _MiniGridEnv.make(CustomUnlock5x5)
CustomUnlockS7 = _MiniGridEnv.make(CustomUnlock7x7)
CustomUnlockS9 = _MiniGridEnv.make(CustomUnlock9x9)
DoorKeyS5 = _MiniGridEnv.make(DoorKeyEnv5x5)
DoorKeyS6 = _MiniGridEnv.make(DoorKeyEnv6x6)
DoorKeyS8 = _MiniGridEnv.make(DoorKeyEnv)


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


class MiniGridSimpleCrossingS9N1(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "SimpleCrossing",
                    [
                        EpisodicTaskVariant(
                            SimpleCrossingS9N1,
                            task_label="SimpleCrossing",
                            variant_label="S9N1",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridSimpleCrossingS9N2(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "SimpleCrossing",
                    [
                        EpisodicTaskVariant(
                            SimpleCrossingS9N2,
                            task_label="SimpleCrossing",
                            variant_label="S9N2",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridSimpleCrossingS9N3(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "SimpleCrossing",
                    [
                        EpisodicTaskVariant(
                            SimpleCrossingS9N3,
                            task_label="SimpleCrossing",
                            variant_label="S9N3",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridDistShiftR2(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "DistShift",
                    [
                        EpisodicTaskVariant(
                            DistShiftR2,
                            task_label="DistShift",
                            variant_label="R2",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridDistShiftR5(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "DistShift",
                    [
                        EpisodicTaskVariant(
                            DistShiftR5,
                            task_label="DistShift",
                            variant_label="R5",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridDistShiftR3(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "DistShift",
                    [
                        EpisodicTaskVariant(
                            DistShiftR3,
                            task_label="DistShift",
                            variant_label="R3",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridDynObstaclesS6N1(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "DynObstacles",
                    [
                        EpisodicTaskVariant(
                            DynObstaclesS6N1,
                            task_label="DynObstacles",
                            variant_label="S6N1",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridDynObstaclesS8N2(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "DynObstacles",
                    [
                        EpisodicTaskVariant(
                            DynObstaclesS8N2,
                            task_label="DynObstacles",
                            variant_label="S8N2",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridDynObstaclesS10N3(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "DynObstacles",
                    [
                        EpisodicTaskVariant(
                            DynObstaclesS10N3,
                            task_label="DynObstacles",
                            variant_label="S10N3",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridCustomFetchS5T1N2(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "CustomFetch",
                    [
                        EpisodicTaskVariant(
                            CustomFetchS5T1N2,
                            task_label="CustomFetch",
                            variant_label="S5T1N2",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridCustomFetchS8T1N2(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "CustomFetch",
                    [
                        EpisodicTaskVariant(
                            CustomFetchS8T1N2,
                            task_label="CustomFetch",
                            variant_label="S8T1N2",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridCustomFetchS16T2N4(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "CustomFetch",
                    [
                        EpisodicTaskVariant(
                            CustomFetchS16T2N4,
                            task_label="CustomFetch",
                            variant_label="S16T2N4",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridCustomUnlockS5(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "CustomUnlock",
                    [
                        EpisodicTaskVariant(
                            CustomUnlockS5,
                            task_label="CustomUnlock",
                            variant_label="S5",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridCustomUnlockS7(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "CustomUnlock",
                    [
                        EpisodicTaskVariant(
                            CustomUnlockS7,
                            task_label="CustomUnlock",
                            variant_label="S7",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridCustomUnlockS9(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "CustomUnlock",
                    [
                        EpisodicTaskVariant(
                            CustomUnlockS9,
                            task_label="CustomUnlock",
                            variant_label="S9",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridDoorKeyS5(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "DoorKey",
                    [
                        EpisodicTaskVariant(
                            DoorKeyS5,
                            task_label="DoorKey",
                            variant_label="S5",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridDoorKeyS6(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "DoorKey",
                    [
                        EpisodicTaskVariant(
                            DoorKeyS6,
                            task_label="DoorKey",
                            variant_label="S6",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridDoorKeyS8(_MiniGridCurriculum):
    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    "DoorKey",
                    [
                        EpisodicTaskVariant(
                            DoorKeyS8,
                            task_label="DoorKey",
                            variant_label="S8",
                            num_episodes=1000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )
