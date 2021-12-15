"""
Copyright © 2021 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to
deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import abc
import typing
from .task_variant import TaskVariantType
from .curriculum import (
    AbstractLearnBlock,
    AbstractCurriculum,
    AbstractLearnBlock,
    AbstractEvalBlock,
    AbstractTaskBlock,
)


class InterleavedEvalCurriculum(AbstractCurriculum[TaskVariantType]):
    """
    One possible version of a curriculum where a single evaluation block
    is interleaved between a sequence of learning blocks.

    This class implements :meth:`Curriculum.blocks()`, and expects the user
    to implement two new methods:

        1. learn_blocks(), which returns the sequence of :class:`LearnBlock`.
        2. eval_block(), which returns the single :class:`EvalBlock` to be
            interleaved between each :class:`LearnBlock`.

    """

    @abc.abstractmethod
    def learn_blocks(self) -> typing.Iterable[AbstractLearnBlock[TaskVariantType]]:
        """
        :return: An iterable of :class:`LearnBlock`.
        """

    @abc.abstractmethod
    def eval_block(self) -> AbstractEvalBlock[TaskVariantType]:
        """
        :return: The single :class:`EvalBlock` to interleave between each
            individual :class:`LearnBlock` returned from
            :meth:`InterleavedEvalCurriculum.learn_blocks`.
        """

    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[TaskVariantType]", "AbstractEvalBlock[TaskVariantType]"
        ]
    ]:
        yield self.eval_block()
        for block in self.learn_blocks():
            yield block
            yield self.eval_block()


class TaskBlock(AbstractTaskBlock):
    """
    A simple subclass of :class:`AbstractTaskBlock` that accepts the task variants
    in the constructor.
    """

    def __init__(self, task_variants: typing.Iterable[TaskVariantType]) -> None:
        super().__init__()
        self._task_variants = task_variants

    def task_variants(self) -> typing.Iterable[TaskVariantType]:
        return self._task_variants


class LearnBlock(AbstractLearnBlock):
    """
    A simple subclass of :class:`AbstractLearnBlock` that accepts the task blocks
    in the constructor.
    """

    def __init__(self, task_blocks: typing.Iterable[TaskBlock]) -> None:
        super().__init__()
        self._task_blocks = task_blocks

    def task_blocks(self) -> typing.Iterable["AbstractTaskBlock[TaskVariantType]"]:
        return self._task_blocks


class EvalBlock(AbstractEvalBlock):
    """
    A simple subclass of :class:`AbstractEvalBlock` that accepts the task blocks
    in the constructor.
    """

    def __init__(self, task_blocks: typing.Iterable[TaskBlock]) -> None:
        super().__init__()
        self._task_blocks = task_blocks

    def task_blocks(self) -> typing.Iterable["AbstractTaskBlock[TaskVariantType]"]:
        return self._task_blocks


def simple_learn_block(
    task_variants: typing.Iterable[TaskVariantType],
) -> AbstractLearnBlock[TaskVariantType]:
    """
    Constucts a learn block with a single task block with the variants passed in.

    :param task_variants: The iterable of TaskVariantType to include in the learn block.
    :return: A :class:`LearnBlock` with a single :class:`TaskBlock` that
        contains the `task_variants` parameter.
    """
    return LearnBlock([TaskBlock(task_variants)])


def simple_eval_block(
    task_variants: typing.Iterable[TaskVariantType],
) -> AbstractEvalBlock[TaskVariantType]:
    """
    Constucts a eval block with a single task block with the variants passed in.

    :param task_variants: The iterable of TaskVariantType to include in the learn block.
    :return: A :class:`EvalBlock` with a single :class:`TaskBlock` that
        contains the `task_variants` parameter.
    """
    return EvalBlock([TaskBlock(task_variants)])
