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
import argparse
import typing

import gym
import numpy as np
from gym_minigrid.envs import (
    DistShift1,
    DistShift2,
    DoorKeyEnv,
    DoorKeyEnv5x5,
    DoorKeyEnv6x6,
    DynamicObstaclesEnv,
    DynamicObstaclesEnv5x5,
    DynamicObstaclesEnv6x6,
    SimpleCrossingEnv,
    SimpleCrossingS9N2Env,
    SimpleCrossingS9N3Env,
)
from gym_minigrid.wrappers import ImgObsWrapper
from tella._curriculums.minigrid.envs import (
    CustomFetchEnv5x5T1N2,
    CustomFetchEnv8x8T1N2,
    CustomFetchEnv16x16T2N4,
    CustomUnlock5x5,
    CustomUnlock7x7,
    CustomUnlock9x9,
    DistShift3,
)
from tella.curriculum import *


class MiniGridReducedActionSpaceWrapper(gym.ActionWrapper):
    """Reduce the action space in environment to help learning."""

    def __init__(self, env: gym.Env, num_actions: int) -> None:
        super().__init__(env)
        assert isinstance(self.action_space, gym.spaces.Discrete)
        self.action_space = gym.spaces.Discrete(num_actions)

    def action(self, act):
        if act >= self.action_space.n:
            act = 0
        return act


class MiniGridMovementActionWrapper(gym.ActionWrapper):
    """Remap pickup, drop, and toggle actions to movements."""

    def __init__(self, env):
        super().__init__(env)

    def action(self, act):
        return act % 3


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

        # If the agent tried to walk over an obstacle or wall
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


class _MiniGridLavaEnv(gym.Wrapper):
    def __init__(self, env_class: typing.Type[gym.Env]) -> None:
        env = env_class()
        env = ImgObsWrapper(env)
        env = MiniGridReducedActionSpaceWrapper(env, num_actions=6)
        env = MiniGridLavaPenaltyWrapper(env)
        super().__init__(env)


class _MiniGridDynObsEnv(gym.Wrapper):
    def __init__(self, env_class: typing.Type[gym.Env]) -> None:
        env = env_class()
        env = ImgObsWrapper(env)
        env = MiniGridReducedActionSpaceWrapper(env, num_actions=6)
        env = MiniGridMovementActionWrapper(env)
        super().__init__(env)


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


class DynObstaclesS5N2(_MiniGridDynObsEnv):
    def __init__(self):
        super().__init__(DynamicObstaclesEnv5x5)


class DynObstaclesS6N3(_MiniGridDynObsEnv):
    def __init__(self):
        super().__init__(DynamicObstaclesEnv6x6)


class DynObstaclesS8N4(_MiniGridDynObsEnv):
    def __init__(self):
        super().__init__(DynamicObstaclesEnv)


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
    (DynObstaclesS5N2, "DynObstacles", "S5N2"),
    (DynObstaclesS6N3, "DynObstacles", "S6N3"),
    (DynObstaclesS8N4, "DynObstacles", "S8N4"),
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


class MiniGridCondensed(InterleavedEvalCurriculum[AbstractRLTaskVariant]):
    def __init__(self, rng_seed: int = 0):
        super().__init__(rng_seed)
        self.rng = np.random.default_rng(rng_seed)

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
                num_episodes=100,
            )
            for cls, task_label, variant_label in TASKS
        )


class MiniGridDispersed(InterleavedEvalCurriculum[AbstractRLTaskVariant]):
    def __init__(self, rng_seed: int = 0, num_repetitions: int = 3):
        super().__init__(rng_seed)
        self.rng = np.random.default_rng(rng_seed)
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
                num_episodes=100,
            )
            for cls, task_label, variant_label in TASKS
        )


def main():
    parser = argparse.ArgumentParser(description="M21 curriculum parser")

    parser.add_argument(
        "-c",
        "--curriculum",
        required=True,
        default=None,
        type=str,
        choices=["MiniGridCondensed", "MiniGridDispersed"],
        help="Curriculum name. Defaults to None.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        help="Seed value for task sequence. Defaults to 0.",
    )
    parser.add_argument(
        "-n",
        "--num-repetitions",
        default=3,
        type=int,
        help="Number of reptitions for dispersed curriculum. Defaults to 3.",
    )

    args = parser.parse_args()

    if args.curriculum == "MiniGridCondensed":
        curriculum = MiniGridCondensed(rng_seed=args.seed)
    elif args.curriculum == "MiniGridDispersed":
        curriculum = MiniGridDispersed(
            rng_seed=args.seed, num_repetitions=args.num_repetitions
        )
    else:
        print(f"Invalid curriculum name: {args.curriculum}")
        return

    for i, block in enumerate(curriculum.learn_blocks_and_eval_blocks()):
        for task_block in block.task_blocks():
            for task_variant in task_block.task_variants():
                print(
                    f"Block {i}, learning_allowed={block.is_learning_allowed}, "
                    f"task_variant={task_variant.task_label}_{task_variant.variant_label}, "
                    f"num_episodes={task_variant.total_episodes}"
                )


if __name__ == "__main__":
    main()
