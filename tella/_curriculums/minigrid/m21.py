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
    DoorKeyEnv5x5,
    DoorKeyEnv6x6,
    SimpleCrossingEnv,
    SimpleCrossingS9N2Env,
    SimpleCrossingS9N3Env,
)
from gym_minigrid.wrappers import ImgObsWrapper

from ...curriculum import (
    InterleavedEvalCurriculum,
    EvalBlock,
    LearnBlock,
    TaskVariant,
    TaskBlock,
    simple_eval_block,
    ValidationError,
)
from .envs import (
    CustomDynamicObstaclesS6N1,
    CustomDynamicObstaclesS8N2,
    CustomDynamicObstaclesS10N3,
    CustomFetchEnv5x5T1N2,
    CustomFetchEnv8x8T1N2,
    CustomFetchEnv10x10T2N4,
    CustomUnlock5x5,
    CustomUnlock7x7,
    CustomUnlock9x9,
    DistShift3,
    DoorKeyEnv7x7,
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


class CustomFetchS10T2N4(_MiniGridEnv):
    def __init__(self):
        super().__init__(CustomFetchEnv10x10T2N4)


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


class DoorKeyS7(_MiniGridEnv):
    def __init__(self):
        super().__init__(DoorKeyEnv7x7)


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
    (CustomFetchS10T2N4, "CustomFetch", "S10T2N4"),
    (CustomUnlockS5, "CustomUnlock", "S5"),
    (CustomUnlockS7, "CustomUnlock", "S7"),
    (CustomUnlockS9, "CustomUnlock", "S9"),
    (DoorKeyS5, "DoorKey", "S5"),
    (DoorKeyS6, "DoorKey", "S6"),
    (DoorKeyS7, "DoorKey", "S7"),
]


class _MiniGridCurriculum(InterleavedEvalCurriculum):
    DEFAULT_BLOCK_LENGTH_UNIT = "episodes"
    DEFAULT_LEARN_BLOCK_LENGTH = 1000
    DEFAULT_EVAL_BLOCK_LENGTH = 100

    def eval_block(self) -> EvalBlock:
        rng = np.random.default_rng(self.eval_rng_seed)
        return simple_eval_block(
            TaskVariant(
                cls,
                task_label=task_label,
                variant_label=variant_label,
                num_episodes=self.DEFAULT_EVAL_BLOCK_LENGTH,
                rng_seed=rng.bit_generator.random_raw(),
            )
            for cls, task_label, variant_label in TASKS
        )

    def _block_limit_from_config(
        self, task_label: str, variant_label: str
    ) -> typing.Dict[str, int]:
        """
        Set variable lengths for task blocks based on optional configuration file.

        Expecting a yaml following this format:
        ---
        learn:  # All learning block limits belong under this header
            default length: 999  # A new default task length can be specified using this key
            CustomFetchS10T2N4: 1234  # Task lengths can be set for a specific task variant
            SimpleCrossing: 42  # Or task lengths can be set for all variants of a task type

        Blocks can alternately be limited by total steps taken
        ---
        learn:
            default unit: steps  # A new default task length unit can be specified using this key
            DynObstacles:  # Task length units can be provided explicitly
                length: 500
                unit: steps
            CustomUnlock: 686  # Or task length can be provided using default units
        """
        learn_config = self.config.get("learn", {})

        # If requested, overwrite default values
        length = learn_config.get("default length", self.DEFAULT_LEARN_BLOCK_LENGTH)
        unit = learn_config.get("default unit", self.DEFAULT_BLOCK_LENGTH_UNIT)

        # If length given for task + variant, use that
        task_variant_label = task_label + variant_label

        label = task_variant_label
        if task_variant_label not in learn_config and task_label in learn_config:
            label = task_label

        config = learn_config.get(label, {})
        if isinstance(config, dict):
            length = config.get("length", length)
            unit = config.get("unit", unit)
        elif isinstance(config, int):
            length = config

        # Length and unit will be used as a kwarg for a task variant, so construct a dict
        returned_kwarg = {f"num_{unit}": length}
        return returned_kwarg

    def validate(self) -> None:
        if not isinstance(self.config, typing.Dict):
            raise ValidationError(
                f"Configuration must be a dictionary, not {self.config}."
            )

        # Strict enforcement of valid keys to prevent silent errors from typos
        for key, value in self.config.items():

            if key == "num learn blocks":
                # Permitting this argument even in MiniGridCondensed for config file reuse
                if not isinstance(value, int) or value < 1:
                    raise ValidationError(
                        f"Num learn blocks must be a positive integer, not {value}."
                    )

            elif key == "learn":
                if not isinstance(value, typing.Dict):
                    raise ValidationError(
                        f"Learn blocks config must be a dictionary, not {value}."
                    )

            else:
                raise ValidationError(f"Unexpected config key, {key}.")

        valid_labels = list({t for _, t, v in TASKS}) + [t + v for _, t, v in TASKS]
        if "learn" in self.config:
            for key, value in self.config["learn"].items():

                if key == "default length":
                    if not isinstance(value, int) or value < 1:
                        raise ValidationError(
                            f"Task default length must be a positive integer, not {value}."
                        )

                elif key == "default unit":
                    if value not in ("episodes", "steps"):
                        raise ValidationError(
                            f"Task default steps must be episodes or steps, not {value}."
                        )

                elif key in valid_labels:
                    if isinstance(value, int):
                        if value < 1:
                            raise ValidationError(
                                f"Task length must be positive, not {value}"
                            )
                    elif isinstance(value, typing.Dict):
                        for task_key, task_value in value.items():
                            if task_key == "length":
                                if not isinstance(task_value, int) or task_value < 1:
                                    raise ValidationError(
                                        f"Task length must be a positive integer, not {task_value}."
                                    )
                            elif task_key == "unit":
                                if task_value not in ("episodes", "steps"):
                                    raise ValidationError(
                                        f"Task unit must be episodes or steps, not {task_value}."
                                    )
                            else:
                                raise ValidationError(
                                    f"Task config key must be length or unit, not {task_key}."
                                )
                    else:
                        raise ValidationError(
                            f"Task config must be either an integer or a "
                            f"dictionary with keys (length, unit), not {value}."
                        )

                else:
                    raise ValidationError(f"Unexpected task config key, {key}")


class MiniGridCondensed(_MiniGridCurriculum):
    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        for cls, task_label, variant_label in self.rng.permutation(TASKS):
            yield LearnBlock(
                [
                    TaskBlock(
                        task_label,
                        [
                            TaskVariant(
                                cls,
                                task_label=task_label,
                                variant_label=variant_label,
                                **self._block_limit_from_config(
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

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        num_learn_blocks = self.config.get(
            "num learn blocks", self.DEFAULT_LEARN_BLOCKS
        )
        for num_block in range(num_learn_blocks):
            for cls, task_label, variant_label in self.rng.permutation(TASKS):

                # Get task limit from config, but apply as total limit over all learn blocks
                ((task_length_kw, total_task_length),) = self._block_limit_from_config(
                    task_label, variant_label
                ).items()  # unpacking known format, single kwarg as dict

                task_limit_this_block = total_task_length // num_learn_blocks
                # If total task length does not evenly divide into num. blocks, distribute the
                #   `remainder` = (total_task_length % num_learn_blocks) over the first
                #   `remainder` blocks
                if num_block < (total_task_length % num_learn_blocks):
                    task_limit_this_block += 1

                # Then repack as a dict for **kwarg unpacking
                task_length_limit = {task_length_kw: task_limit_this_block}

                yield LearnBlock(
                    [
                        TaskBlock(
                            task_label,
                            [
                                TaskVariant(
                                    cls,
                                    task_label=task_label,
                                    variant_label=variant_label,
                                    **task_length_limit,
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

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        yield LearnBlock(
            [
                TaskBlock(
                    self.TASK_LABEL,
                    [
                        TaskVariant(
                            self.TASK_CLASS,
                            task_label=self.TASK_LABEL,
                            variant_label=self.VARIANT_LABEL,
                            **self._block_limit_from_config(
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


class MiniGridCustomFetchS10T2N4(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = CustomFetchS10T2N4, "CustomFetch", "S10T2N4"


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


class MiniGridDoorKeyS7(_STECurriculum):
    TASK_CLASS, TASK_LABEL, VARIANT_LABEL = DoorKeyS7, "DoorKey", "S7"
