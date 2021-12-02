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
