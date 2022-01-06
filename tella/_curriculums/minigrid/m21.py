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
import argparse
import typing

import numpy as np
from gym_minigrid.envs import (DistShift1, DistShift2, DoorKeyEnv,
                               DoorKeyEnv5x5, DoorKeyEnv6x6,
                               DynamicObstaclesEnv, DynamicObstaclesEnv5x5,
                               DynamicObstaclesEnv6x6, SimpleCrossingEnv,
                               SimpleCrossingS9N2Env, SimpleCrossingS9N3Env)
from tella._curriculums.minigrid.envs import (CustomFetchEnv5x5T1N2,
                                              CustomFetchEnv8x8T1N2,
                                              CustomFetchEnv16x16T2N4,
                                              CustomUnlock5x5, CustomUnlock7x7,
                                              CustomUnlock9x9, DistShift3)
from tella.curriculum import *

TASKS = [
    (SimpleCrossingEnv, "SimpleCrossing", "S9N1"),
    (SimpleCrossingS9N2Env, "SimpleCrossing", "S9N2"),
    (SimpleCrossingS9N3Env, "SimpleCrossing", "S9N3"),
    (DistShift1, "DistShift", "1"),
    (DistShift2, "DistShift", "2"),
    (DistShift3, "DistShift", "3"),
    (DynamicObstaclesEnv5x5, "DynamicObstacles", "S5N2"),
    (DynamicObstaclesEnv6x6, "DynamicObstacles", "S6N3"),
    (DynamicObstaclesEnv, "DynamicObstacles", "S8N4"),
    (CustomFetchEnv5x5T1N2, "CustomFetch", "S5T1N2"),
    (CustomFetchEnv8x8T1N2, "CustomFetch", "S8T1N2"),
    (CustomFetchEnv16x16T2N4, "CustomFetch", "S16T2N4"),
    (CustomUnlock5x5, "CustomUnlock", "S5"),
    (CustomUnlock7x7, "CustomUnlock", "S7"),
    (CustomUnlock9x9, "CustomUnlock", "S9"),
    (DoorKeyEnv5x5, "DoorKey", "S5"),
    (DoorKeyEnv6x6, "DoorKey", "S6"),
    (DoorKeyEnv, "DoorKey", "S8"),
]

class MiniGridCondensed(InterleavedEvalCurriculum[AbstractRLTaskVariant]):
    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def learn_blocks(self,) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
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
    def __init__(self, num_repetitions: int = 3, seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)
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
                                    num_episodes=1000//self.num_repetitions,
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

    parser.add_argument("-c", "--curriculum", required=True, default=None, type=str, choices=[
                        "MiniGridCondensed", "MiniGridDispersed"], help="Curriculum name. Defaults to None.")
    parser.add_argument("-s", "--seed", default=0, type=int,
                        help="Seed value for task sequence. Defaults to 0.")
    parser.add_argument("-n", "--num-repetitions", default=3, type=int,
                        help="Number of reptitions for dispersed curriculum. Defaults to 3.")

    args = parser.parse_args()

    if args.curriculum == "MiniGridCondensed":
        curriculum = MiniGridCondensed(seed=args.seed)
    elif args.curriculum == "MiniGridDispersed":
        curriculum = MiniGridDispersed(
            seed=args.seed, num_repetitions=args.num_repetitions)
    else:
        print(f"Invalid curriculum name: {args.curriculum}")
        return

    for i, block in enumerate(curriculum.learn_blocks_and_eval_blocks()):
        for task_block in block.task_blocks():
            for task_variant in task_block.task_variants():
                print(
                    f"Block {i}, learning_allowed={block.is_learning_allowed()}, "
                    f"task_variant={task_variant.task_label}_{task_variant.variant_label}, "
                    f"num_episodes={task_variant.total_episodes}"
                )


if __name__ == "__main__":
    main()
