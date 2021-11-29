import typing
import abc

InputType = typing.TypeVar("InputType")
OutputType = typing.TypeVar("OutputType")
InfoType = typing.TypeVar("InfoType")


class Curriculum(abc.ABC, typing.Generic[InputType, OutputType, InfoType]):
    """
    Represents a lifelong/continual learning curriculum. A curriculum is simply
    a sequence of :class:`Block`s.
    """

    @abc.abstractmethod
    def blocks(
        self,
    ) -> typing.Iterable["Block[InputType, OutputType, InfoType]"]:
        """
        :return: An Iterable of :class:`Block`.
        """


class Block(abc.ABC, typing.Generic[InputType, OutputType, InfoType]):
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
    def experiences(
        self,
    ) -> typing.Iterable["Experience[InputType, OutputType, InfoType]"]:
        """
        :return: An Iterable of :class:`Experience`.
        """


class Experience(abc.ABC, typing.Generic[InputType, OutputType, InfoType]):
    """
    An experience transforms some object of type `InputType` into an object
    with type `OutputType`. Additionally, an experience has some InfoTypermation
    associated with it, of type `InfoType`.

    For RL, an experience can be thought of as taking an agent as InputType,
    producing an Iterable of MDP transitions as OutputType, and giving
    and :class:`gym.Env` as the InfoType.

    For image classification, an experience could be a batch size integer as
    InputType, Batch of image/label data as OutputType, and giving
    the dataset object as InputType.
    """

    @abc.abstractmethod
    def validate(self) -> None:
        """
        A method to validate that the experience is set up properly.

        This should raise an Exception if the experience is not set up properly.
        """

    @abc.abstractmethod
    def info(self) -> InfoType:
        """
        :return: The object of type `InfoType` associated with this experience.
        """

    @abc.abstractmethod
    def generate(self, inp: InputType) -> OutputType:
        """
        The main method to generate the Experience data.

        :param inp: The object of type InputType.
        :return: The data for the experience.
        """


class LearnBlock(Block[InputType, OutputType, InfoType]):
    """
    A helper class to create learning blocks. Experiences are passed into the
    constructor, and is_learning_allowed always returns True.
    """

    def __init__(
        self,
        experiences: typing.Iterable[Experience[InputType, OutputType, InfoType]],
    ) -> None:
        self._experiences = experiences

    def experiences(
        self,
    ) -> typing.Iterable[Experience[InputType, OutputType, InfoType]]:
        return self._experiences

    def is_learning_allowed(self) -> bool:
        return True


class EvalBlock(Block[InputType, OutputType, InfoType]):
    """
    A helper class to create evaluation blocks. Experiences are passed into the
    constructor, and is_learning_allowed always returns False.
    """

    def __init__(
        self,
        experiences: typing.Iterable[Experience[InputType, OutputType, InfoType]],
    ) -> None:
        self._experiences = experiences

    def experiences(
        self,
    ) -> typing.Iterable[Experience[InputType, OutputType, InfoType]]:
        return self._experiences

    def is_learning_allowed(self) -> bool:
        return False


class InterleavedEvalCurriculum(Curriculum[InputType, OutputType, InfoType]):
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
    def learn_blocks(
        self,
    ) -> typing.Iterable[LearnBlock[InputType, OutputType, InfoType]]:
        """
        :return: An iterable of :class:`LearnBlock`.
        """

    @abc.abstractmethod
    def eval_block(
        self,
    ) -> EvalBlock[InputType, OutputType, InfoType]:
        """
        :return: The single :class:`EvalBlock` to interleave between each
            individual :class:`LearnBlock` returned from
                :meth:`InterleavedEvalCurriculum.learn_blocks`.
        """

    def blocks(
        self,
    ) -> typing.Iterable[Block[InputType, OutputType, InfoType]]:
        yield self.eval_block()
        for block in self.learn_blocks():
            yield block
            yield self.eval_block()
