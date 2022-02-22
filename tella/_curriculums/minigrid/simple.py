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
from .m21 import SimpleCrossingS9N1, DistShiftR2, DynObstaclesS6N1
from ...curriculum import (
    InterleavedEvalCurriculum,
    LearnBlock,
    EvalBlock,
    TaskBlock,
    TaskVariant,
    simple_eval_block,
)


TASKS = [
    (SimpleCrossingS9N1, "SimpleCrossing", "S9N1"),
    (DistShiftR2, "DistShift", "R2"),
    (DynObstaclesS6N1, "DynObstacles", "S6N1"),
]


class SimpleMiniGridCurriculum(InterleavedEvalCurriculum):
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
                                num_episodes=5,
                                rng_seed=self.rng.bit_generator.random_raw(),
                            )
                        ],
                    )
                ]
            )

    def eval_block(self) -> EvalBlock:
        return simple_eval_block(
            TaskVariant(
                cls,
                task_label=task_label,
                variant_label=variant_label,
                num_episodes=5,
                rng_seed=self.eval_rng_seed,
            )
            for cls, task_label, variant_label in TASKS
        )


class SimpleStepMiniGridCurriculum(SimpleMiniGridCurriculum):
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
                                rng_seed=self.rng.bit_generator.random_raw(),
                                num_steps=5000,
                            )
                        ],
                    )
                ]
            )
