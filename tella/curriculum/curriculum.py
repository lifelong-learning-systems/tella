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

import typing
import abc
from .task_variant import TaskVariantType


class AbstractCurriculum(abc.ABC, typing.Generic[TaskVariantType]):
    """
    Represents a lifelong/continual learning curriculum. A curriculum is simply
    a sequence of :class:`AbstractLearnBlock`s and :class:`AbstractEvalBlock`s.
    """

    @abc.abstractmethod
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[TaskVariantType]", "AbstractEvalBlock[TaskVariantType]"
        ]
    ]:
        """
        :return: An Iterable of Learn Blocks and Eval Blocks.
        """


class AbstractLearnBlock(abc.ABC, typing.Generic[TaskVariantType]):
    """
    Represents a sequence of 1 or more :class:`AbstractTaskBlock`, where the
    data can be used for learning.
    """

    def is_learning_allowed(self) -> bool:
        return True

    @abc.abstractmethod
    def task_blocks(self) -> typing.Iterable["AbstractTaskBlock[TaskVariantType]"]:
        """
        :return: An Iterable of Task Blocks
        """


class AbstractEvalBlock(abc.ABC, typing.Generic[TaskVariantType]):
    """
    Represents a sequence of 1 or more :class:`AbstractTaskBlock`, where the
    data can NOT be used for learning.
    """

    def is_learning_allowed(self) -> bool:
        return False

    @abc.abstractmethod
    def task_blocks(self) -> typing.Iterable["AbstractTaskBlock[TaskVariantType]"]:
        """
        :return: An Iterable of Task Blocks
        """


class AbstractTaskBlock(abc.ABC, typing.Generic[TaskVariantType]):
    """
    Represents a sequence of 1 or more Task Variants (represented by the
    generic type `TaskVariantType`.)
    """

    def task_variants(self) -> typing.Iterable[TaskVariantType]:
        """
        :return: An Iterable of :class:`TaskVariantType`.
        """
