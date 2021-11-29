import typing
import abc
from .experiences.experience import GenericExperience


class Curriculum(abc.ABC, typing.Generic[GenericExperience]):
    """
    Represents a lifelong/continual learning curriculum. A curriculum is simply
    a sequence of :class:`Block`s.
    """

    @abc.abstractmethod
    def blocks(
        self,
    ) -> typing.Iterable["Block[GenericExperience]"]:
        """
        :return: An Iterable of :class:`Block`.
        """


class Block(abc.ABC, typing.Generic[GenericExperience]):
    """
    This represents a sequence of :class:`Experience`.

    Additionally a block can either allow learning and collection of
    data to happen (i.e. `block.is_learning_allowed() == True`), or not
    (i.e. `block.is_learning_allowed() == False`).

    .. Example Usage:

        block: Block = ...
        for experience: Experience in block.experiences():
            ... # Do something with each experience

    """

    @abc.abstractmethod
    def is_learning_allowed(self) -> bool:
        """
        :return: True if learning is allowed to happen, False otherwise.
        """

    @abc.abstractmethod
    def experiences(self) -> typing.Iterable[GenericExperience]:
        """
        :return: An Iterable of :class:`Experience`.
        """


class LearnBlock(Block[GenericExperience]):
    """
    A helper class to create learning blocks. Experiences are passed into the
    constructor, and is_learning_allowed always returns True.
    """

    def __init__(self, experiences: typing.Iterable[GenericExperience]) -> None:
        self._experiences = experiences

    def experiences(self) -> typing.Iterable[GenericExperience]:
        return self._experiences

    def is_learning_allowed(self) -> bool:
        return True


class EvalBlock(Block[GenericExperience]):
    """
    A helper class to create evaluation blocks. Experiences are passed into the
    constructor, and is_learning_allowed always returns False.
    """

    def __init__(self, experiences: typing.Iterable[GenericExperience]) -> None:
        self._experiences = experiences

    def experiences(self) -> typing.Iterable[GenericExperience]:
        return self._experiences

    def is_learning_allowed(self) -> bool:
        return False


class InterleavedEvalCurriculum(Curriculum[GenericExperience]):
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
    def learn_blocks(self) -> typing.Iterable[LearnBlock[GenericExperience]]:
        """
        :return: An iterable of :class:`LearnBlock`.
        """

    @abc.abstractmethod
    def eval_block(self) -> EvalBlock[GenericExperience]:
        """
        :return: The single :class:`EvalBlock` to interleave between each
            individual :class:`LearnBlock` returned from
            :meth:`InterleavedEvalCurriculum.learn_blocks`.
        """

    def blocks(self) -> typing.Iterable[Block[GenericExperience]]:
        yield self.eval_block()
        for block in self.learn_blocks():
            yield block
            yield self.eval_block()
