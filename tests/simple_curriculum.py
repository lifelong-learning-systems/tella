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
from gym.envs.classic_control import CartPoleEnv
from tella.curriculum import AbstractCurriculum, AbstractLearnBlock, AbstractEvalBlock
from tella.curriculum import AbstractRLTaskVariant, EpisodicTaskVariant
from tella.curriculum import simple_learn_block, simple_eval_block


class SimpleRLCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    variant_label="Variant1",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )
        yield simple_eval_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


class MultiEpisodeRLCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=5,
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=4,
                    variant_label="Variant1",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )
        yield simple_eval_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=3,
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


class LearnOnlyCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    variant_label="Variant1",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )


class EvalOnlyCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        yield simple_eval_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    variant_label="Variant1",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )
