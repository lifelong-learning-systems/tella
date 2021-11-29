import abc
import typing


InputType = typing.TypeVar("InputType")
OutputType = typing.TypeVar("OutputType")
InfoType = typing.TypeVar("InfoType")


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


GenericExperience = typing.TypeVar("GenericExperience", bound=Experience)
