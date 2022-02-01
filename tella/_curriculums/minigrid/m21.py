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


class SimpleCrossingS9N1(_MiniGridEnv):
    def __init__(self):
        super().__init__(SimpleCrossingEnv)


class SimpleCrossingS9N2(_MiniGridEnv):
    def __init__(self):
        super().__init__(SimpleCrossingS9N2Env)


class SimpleCrossingS9N3(_MiniGridEnv):
    def __init__(self):
        super().__init__(SimpleCrossingS9N3Env)


class DistShiftR2(_MiniGridLavaEnv):
    def __init__(self):
        super().__init__(DistShift1)


class DistShiftR5(_MiniGridLavaEnv):
    def __init__(self):
        super().__init__(DistShift2)


class DistShiftR3(_MiniGridLavaEnv):
    def __init__(self):
        super().__init__(DistShift3)


class DynObstaclesS6N1(_MiniGridEnv):
    def __init__(self):
        super().__init__(CustomDynamicObstaclesS6N1)


class DynObstaclesS8N2(_MiniGridEnv):
    def __init__(self):
        super().__init__(CustomDynamicObstaclesS8N2)


class DynObstaclesS10N3(_MiniGridEnv):
    def __init__(self):
        super().__init__(CustomDynamicObstaclesS10N3)


class CustomFetchS5T1N2(_MiniGridEnv):
    def __init__(self):
        super().__init__(CustomFetchEnv5x5T1N2)


class CustomFetchS8T1N2(_MiniGridEnv):
    def __init__(self):
        super().__init__(CustomFetchEnv8x8T1N2)


class CustomFetchS16T2N4(_MiniGridEnv):
    def __init__(self):
        super().__init__(CustomFetchEnv16x16T2N4)


class CustomUnlockS5(_MiniGridEnv):
    def __init__(self):
        super().__init__(CustomUnlock5x5)


class CustomUnlockS7(_MiniGridEnv):
    def __init__(self):
        super().__init__(CustomUnlock7x7)


class CustomUnlockS9(_MiniGridEnv):
    def __init__(self):
        super().__init__(CustomUnlock9x9)


class DoorKeyS5(_MiniGridEnv):
    def __init__(self):
        super().__init__(DoorKeyEnv5x5)


class DoorKeyS6(_MiniGridEnv):
    def __init__(self):
        super().__init__(DoorKeyEnv6x6)


class DoorKeyS8(_MiniGridEnv):
    def __init__(self):
        super().__init__(DoorKeyEnv)


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
    DEFAULT_LEARN_BLOCK_LENGTH = 1000
    DEFAULT_EVAL_BLOCK_LENGTH = 100

    def eval_block(self) -> AbstractEvalBlock[AbstractRLTaskVariant]:
        rng = np.random.default_rng(self.eval_rng_seed)
        return simple_eval_block(
            EpisodicTaskVariant(
                cls,
                task_label=task_label,
                variant_label=variant_label,
                num_episodes=self.DEFAULT_EVAL_BLOCK_LENGTH,
                rng_seed=rng.bit_generator.random_raw(),
            )
            for cls, task_label, variant_label in TASKS
        )

    def episode_limit_from_config(self, task_label: str, variant_label: str):
        """
        Set variable lengths for task blocks based on optional configuration file.

        Expecting a yaml following this format:
        learn:  # All learning block limits belong under this header
            default length: 999  # A new default task length can be specified using this key
            CustomFetchS16T2N4: 1234  # Task lengths can be set for a specific task variant
            SimpleCrossing: 42  # Or task lengths can be set for all variants of a task type
        """
        default_length = self.DEFAULT_LEARN_BLOCK_LENGTH
        if "learn" in self.config:
            learn_config = self.config["learn"]

            # If length given for task + variant, use that
            task_variant_label = task_label + variant_label
            if task_variant_label in learn_config:
                length = learn_config[task_variant_label]
                # TODO: move config validation elsewhere
                assert isinstance(length, int)
                return length

            # Otherwise, if length given for task, use that
            if task_label in learn_config:
                length = learn_config[task_label]
                # TODO: move config validation elsewhere
                assert isinstance(length, int)
                return length

            # If requested, overwrite default value
            if "default length" in learn_config:
                default_length = learn_config["default length"]
                # TODO: move config validation elsewhere
                assert isinstance(default_length, int)

        # Otherwise, resort to default
        return default_length


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
                                num_episodes=self.episode_limit_from_config(
                                    task_label, variant_label
                                ),
                                rng_seed=self.rng.bit_generator.random_raw(),
                            )
                        ],
                    )
                ]
            )


class MiniGridDispersed(_MiniGridCurriculum):
    DEFAULT_LEARN_BLOCKS = 3

    def __init__(
        self,
        rng_seed: int,
        config_file: typing.Optional[str] = None,
    ):
        super().__init__(rng_seed, config_file)
        self.num_learn_blocks = self.config.get(
            "num learn blocks", self.DEFAULT_LEARN_BLOCKS
        )

    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        for num_block in range(self.num_learn_blocks):
            for cls, task_label, variant_label in self.rng.permutation(TASKS):

                num_total_episodes = self.episode_limit_from_config(
                    task_label, variant_label
                )
                num_episodes_this_block = num_total_episodes // self.num_learn_blocks
                # If total episodes does not evenly divide into num. blocks, increment up to mod.
                if num_block < (num_total_episodes % self.num_learn_blocks):
                    num_episodes_this_block += 1

                yield LearnBlock(
                    [
                        TaskBlock(
                            task_label,
                            [
                                EpisodicTaskVariant(
                                    cls,
                                    task_label=task_label,
                                    variant_label=variant_label,
                                    num_episodes=num_episodes_this_block,
                                    rng_seed=self.rng.bit_generator.random_raw(),
                                )
                            ],
                        )
                    ]
                )


class _STECurriculum(_MiniGridCurriculum):
    TASK_CLASS: typing.Type[gym.Env] = None
    TASK_LABEL: str = None
    VARIANT_LABEL: str = None

    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        yield LearnBlock(
            [
                TaskBlock(
                    self.TASK_LABEL,
                    [
                        EpisodicTaskVariant(
                            self.TASK_CLASS,
                            task_label=self.TASK_LABEL,
                            variant_label=self.VARIANT_LABEL,
                            num_episodes=self.episode_limit_from_config(
                                self.TASK_LABEL, self.VARIANT_LABEL
                            ),
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                    ],
                )
            ]
        )


class MiniGridSimpleCrossingS9N1(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = SimpleCrossingS9N1, "SimpleCrossing", "S9N1"


class MiniGridSimpleCrossingS9N2(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = SimpleCrossingS9N2, "SimpleCrossing", "S9N2"


class MiniGridSimpleCrossingS9N3(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = SimpleCrossingS9N3, "SimpleCrossing", "S9N3"


class MiniGridDistShiftR2(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = DistShiftR2, "DistShift", "R2"


class MiniGridDistShiftR5(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = DistShiftR5, "DistShift", "R5"


class MiniGridDistShiftR3(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = DistShiftR3, "DistShift", "R3"


class MiniGridDynObstaclesS6N1(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = DynObstaclesS6N1, "DynObstacles", "S6N1"


class MiniGridDynObstaclesS8N2(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = DynObstaclesS8N2, "DynObstacles", "S8N2"


class MiniGridDynObstaclesS10N3(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = DynObstaclesS10N3, "DynObstacles", "S10N3"


class MiniGridCustomFetchS5T1N2(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = CustomFetchS5T1N2, "CustomFetch", "S5T1N2"


class MiniGridCustomFetchS8T1N2(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = CustomFetchS8T1N2, "CustomFetch", "S8T1N2"


class MiniGridCustomFetchS16T2N4(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = CustomFetchS16T2N4, "CustomFetch", "S16T2N4"


class MiniGridCustomUnlockS5(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = CustomUnlockS5, "CustomUnlock", "S5"


class MiniGridCustomUnlockS7(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = CustomUnlockS7, "CustomUnlock", "S7"


class MiniGridCustomUnlockS9(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = CustomUnlockS9, "CustomUnlock", "S9"


class MiniGridDoorKeyS5(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = DoorKeyS5, "DoorKey", "S5"


class MiniGridDoorKeyS6(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = DoorKeyS6, "DoorKey", "S6"


class MiniGridDoorKeyS8(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = DoorKeyS8, "DoorKey", "S8"
