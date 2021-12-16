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

import typing
import numpy as np
from tella.curriculum import *
from gym_minigrid.envs import DistShift1, DynamicObstaclesEnv, SimpleCrossingEnv
from gym_minigrid.wrappers import ImgObsWrapper


class SimpleCrossingImg(ImgObsWrapper):
    def __init__(self):
        super().__init__(SimpleCrossingEnv())


class DistShift1Img(ImgObsWrapper):
    def __init__(self):
        super().__init__(DistShift1())


class DynamicObstaclesImg(ImgObsWrapper):
    def __init__(self):
        super().__init__(DynamicObstaclesEnv())


TASK_VARIANTS = [
    (SimpleCrossingImg, "SimpleCrossing", "Default"),
    (DistShift1Img, "DistShift", "Default"),
    (DynamicObstaclesImg, "DynamicObstacles", "Default"),
]


class SimpleMiniGridCurriculum(InterleavedEvalCurriculum[AbstractRLTaskVariant]):
    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def learn_blocks(
        self,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        task_variants = [
            EpisodicTaskVariant(
                cls,
                task_label=task_label,
                variant_label=variant_label,
                num_episodes=5,
            )
            for cls, task_label, variant_label in TASK_VARIANTS.copy()
        ]
        self.rng.shuffle(task_variants)
        for task_variant in task_variants:
            yield simple_learn_block([task_variant])

    def eval_block(self) -> AbstractEvalBlock[AbstractRLTaskVariant]:
        return simple_eval_block(
            [
                EpisodicTaskVariant(
                    cls,
                    task_label=task_label,
                    variant_label=variant_label,
                    num_episodes=1,
                )
                for cls, task_label, variant_label in TASK_VARIANTS.copy()
            ]
        )
