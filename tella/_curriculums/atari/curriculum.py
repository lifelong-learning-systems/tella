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

import numpy as np

from ...curriculum import (
    InterleavedEvalCurriculum,
    simple_learn_block,
    simple_eval_block,
    TaskVariant,
    LearnBlock,
    EvalBlock,
)
from .environments import ATARI_TASKS


class AtariCurriculum(InterleavedEvalCurriculum):
    TASKS = list(ATARI_TASKS)

    def eval_block(self) -> EvalBlock:
        rng = np.random.default_rng(self.eval_rng_seed)
        return simple_eval_block(
            TaskVariant(
                ATARI_TASKS[task_label],
                task_label=task_label,
                num_episodes=100,
                rng_seed=rng.bit_generator.random_raw(),
            )
            for task_label in self.TASKS
        )

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        for task_label in self.rng.permutation(self.TASKS):
            yield simple_learn_block(
                [
                    TaskVariant(
                        ATARI_TASKS[task_label],
                        task_label=task_label,
                        num_steps=10_000,
                        rng_seed=self.rng.bit_generator.random_raw(),
                    )
                ]
            )


class BreakoutAndPong(AtariCurriculum):
    TASKS = ["Breakout", "Pong"]
