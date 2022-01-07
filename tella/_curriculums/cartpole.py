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
from gym.envs.classic_control import CartPoleEnv
from gym.wrappers.time_limit import TimeLimit
from tella.curriculum import *


class _CartPoleV0(TimeLimit):
    def __init__(self):
        super().__init__(CartPoleEnv(), max_episode_steps=200)


class SimpleCartPoleCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
        rng_seed: int,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    _CartPoleV0,
                    num_episodes=5,
                    task_label="CartPole",
                    variant_label="Default",
                )
            ]
        )
        yield simple_eval_block(
            [
                EpisodicTaskVariant(
                    _CartPoleV0,
                    num_episodes=1,
                    task_label="CartPole",
                    variant_label="Default",
                )
            ]
        )


class CartPole1000Curriculum(InterleavedEvalCurriculum[AbstractRLTaskVariant]):
    def learn_blocks(
        self,
        rng_seed: int,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        for _ in range(10):
            yield simple_learn_block(
                [
                    EpisodicTaskVariant(
                        _CartPoleV0,
                        num_episodes=100,
                        task_label="CartPole",
                        variant_label="Default",
                    )
                ]
            )

    def eval_block(
        self,
        rng_seed: int,
    ) -> AbstractEvalBlock[AbstractRLTaskVariant]:
        return simple_eval_block(
            [
                EpisodicTaskVariant(
                    _CartPoleV0,
                    num_episodes=10,
                    task_label="CartPole",
                    variant_label="Default",
                )
            ]
        )
