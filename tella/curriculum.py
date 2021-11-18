import typing
import abc

T = typing.TypeVar("T")
S = typing.TypeVar("S")
I = typing.TypeVar("I")


class Experience(abc.ABC, typing.Generic[S, T, I]):
    """
    An experience transforms some generic parameter U into a iterable of
    generic parameters T.

    For RL this can be thought of as taking an agent (parameter U) and producing
    an iterable of transitions (obs/action/rewards).

    For classification this can be thought of as taking None (parameter U) and
    producing an iterable of labelled data.
    """

    @abc.abstractmethod
    def validate(self) -> None:
        """
        A method to validate that the experience is set up properly.

        This should raise an Exception if the experience is not set up properly.
        """
        pass

    @abc.abstractmethod
    def info(self) -> I:
        pass

    @abc.abstractmethod
    def generate(self, s: S) -> typing.Iterable[T]:
        pass


class Block(abc.ABC):
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
    def experiences(self) -> typing.Iterable[Experience]:
        """
        :return: An Iterable of :class:`Experience`.
        """


class Curriculum(abc.ABC):
    """
    Represents a lifelong/continual learning curriculum. A curriculum is simply
    a sequence of :class:`Block`s.
    """

    @abc.abstractmethod
    def blocks(self) -> typing.Iterable[Block]:
        """
        :return: An Iterable of :class:`Block`.
        """


class LearnBlock(Block):
    """
    A helper class to create learning blocks. Experiences are passed into the
    constructor, and is_learning_allowed always returns True.
    """

    def __init__(self, experiences: typing.Iterable[Experience]) -> None:
        self._experiences = experiences

    def experiences(self) -> typing.Iterable[Experience]:
        return self._experiences

    def is_learning_allowed(self) -> bool:
        return True


class EvalBlock(Block):
    """
    A helper class to create evaluation blocks. Experiences are passed into the
    constructor, and is_learning_allowed always returns False.
    """

    def __init__(self, experiences: typing.Iterable[Experience]) -> None:
        self._experiences = experiences

    def experiences(self) -> typing.Iterable[Experience]:
        return self._experiences

    def is_learning_allowed(self) -> bool:
        return False


class InterleavedEvalCurriculum(Curriculum):
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
    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        """
        :return: An iterable of :class:`LearnBlock`.
        """

    @abc.abstractmethod
    def eval_block(self) -> EvalBlock:
        """
        :return: The single :class:`EvalBlock` to interleave between each
            individual :class:`LearnBlock` returned from
                :meth:`InterleavedEvalCurriculum.learn_blocks`.
        """

    def blocks(self) -> typing.Iterable[Block]:
        yield self.eval_block()
        for block in self.learn_blocks():
            yield block
            yield self.eval_block()
