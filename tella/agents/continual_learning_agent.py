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
import abc
from ..curriculum.task_variant import TaskVariantType

Metrics = typing.Dict[str, float]


class ContinualLearningAgent(abc.ABC, typing.Generic[TaskVariantType]):
    """
    The base class for a continual learning agent. A CL Agent is an agent that
    can consume some task variant of a generic type (:class:`TaskVariantType`).

    The only requirement is to implement :meth:`ContinualLearningAgent.consume_task_variant()`,
    which takes an object of type :class:`TaskVariantType`.
    """

    def __init__(self) -> None:
        super().__init__()
        self.is_learning_allowed: bool = False

    def block_start(self, is_learning_allowed: bool) -> None:
        """
        Signifies a new block (either learning or evaluation) is about to start.

        The next method called would be :meth:`ContinualLearningAgent.task_start()`.

        NOTE: the attribute :attr:`ContinualLearningAgent.is_learning_allowed` is
            also set outside of this method.

        :param is_learning_allowed: Whether the block is a learning block or
            an evaluation block.
        """
        pass

    def task_start(
        self,
        task_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a new task is about to start. `task_info`
        may contain task id/label or task parameters.

        The next method called would be :meth:`ContinualLearningAgent.task_variant_start()`.

        :param task_name: An optional value indicating the name of the task
        """
        pass

    def task_variant_start(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a new task variant is about to start. `task_info`
        may contain task id/label or task parameters.

        The next method called would be :meth:`ContinualLearningAgent.consume_experience()`.

        :param task_name: An optional value indicating the name of the task
        :param variant_name: An optional value indicating the name of the task variant
        """
        pass

    @abc.abstractmethod
    def consume_task_variant(self, task_variant: TaskVariantType) -> Metrics:
        """
        Passes an object of type :class:`TaskVariantType` to the agent to consume for learning.

        The next method called would be :meth:`ContinualLearningAgent.task_variant_end()`.

        :return: Dictionary with string keys, and float values. It represents
            some generic metrics that were calculated across the experiences
        """

    @abc.abstractmethod
    def eval_task_variant(self, task_variant: TaskVariantType) -> Metrics:
        """
        Passes an object of type :class:`TaskVariantType` to the agent to consume for evaluation.

        The next method called would be :meth:`ContinualLearningAgent.task_variant_end()`.

        :return: Dictionary with string keys, and float values. It represents
            some generic metrics that were calculated across the experiences
        """

    def task_variant_end(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a task variant has just ended.

        The next method called would be :meth:`Agent.task_variant_start()` if there
        are more task variants in the block, otherwise :meth:`Agent.task_end()`.

        :param task_name: An optional value indicating the name of the task
        :param variant_name: An optional value indicating the name of the task variant
        """
        pass

    def task_end(
        self,
        task_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a task has just ended.

        The next method called would be :meth:`Agent.task_start()` if there
        are more tasks in the block, otherwise :meth:`Agent.block_end()`.

        :param task_name: An optional value indicating the name of the task
        """
        pass

    def block_end(self, is_learning_allowed: bool) -> None:
        """
        Signifies the end of a block.

        The next method called would be :meth:`Agent.block_start()`
        if there are more blocks, otherwise the program would end.

        :param is_learning_allowed: The same data passed into the last :meth:`Agent.block_end()`.
        """
        pass
