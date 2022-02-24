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

from gym.envs.classic_control import CartPoleEnv
from gym.wrappers.time_limit import TimeLimit

from ..curriculum import (
    AbstractCurriculum,
    Block,
    LearnBlock,
    EvalBlock,
    simple_learn_block,
    simple_eval_block,
    TaskVariant,
    InterleavedEvalCurriculum,
)


class _CartPoleV0(TimeLimit):
    def __init__(self):
        super().__init__(CartPoleEnv(), max_episode_steps=200)


class SimpleCartPoleCurriculum(AbstractCurriculum):
    def learn_blocks_and_eval_blocks(self) -> typing.Iterable[Block]:
        yield simple_learn_block(
            [
                TaskVariant(
                    _CartPoleV0,
                    num_episodes=5,
                    task_label="CartPole",
                    variant_label="Default",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )
        yield simple_eval_block(
            [
                TaskVariant(
                    _CartPoleV0,
                    num_episodes=1,
                    task_label="CartPole",
                    variant_label="Default",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


class CartPole1000Curriculum(InterleavedEvalCurriculum):
    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        for _ in range(10):
            yield simple_learn_block(
                [
                    TaskVariant(
                        _CartPoleV0,
                        num_episodes=100,
                        task_label="CartPole",
                        variant_label="Default",
                        rng_seed=self.rng.bit_generator.random_raw(),
                    )
                ]
            )

    def eval_block(self) -> EvalBlock:
        return simple_eval_block(
            [
                TaskVariant(
                    _CartPoleV0,
                    num_episodes=10,
                    task_label="CartPole",
                    variant_label="Default",
                    rng_seed=self.eval_rng_seed,
                )
            ]
        )
