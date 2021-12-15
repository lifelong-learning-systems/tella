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


InputType = typing.TypeVar("InputType")
ExperienceType = typing.TypeVar("ExperienceType")
InfoType = typing.TypeVar("InfoType")


class AbstractTaskVariant(abc.ABC, typing.Generic[InputType, ExperienceType, InfoType]):
    """
    A TaskVariant abstractly represents some amount of experience in a single task.

    We represent a TaskVariant as something that takes an input object (`InputType`) and produces
    an generic experience `ExperienceType` object. Additionally, a TaskVariant
    has some information (`InfoType`) associated with it.

    This representation allows us to represent both RL and SL tasks/experiences.

    For RL, a TaskVariant can be thought of as taking an agent as InputType,
    producing an Iterable of Step Data as ExperienceType, and giving
    and :class:`gym.Env` as the InfoType.

    For SL, a TaskVariant could take a batch size integer as
    InputType, and produce a Batch of image/label data as ExperienceType.
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
    def generate(self, inp: InputType) -> ExperienceType:
        """
        The main method to generate the Experience data.

        :param inp: The object of type InputType.
        :return: The data for the experience.
        """

    @property
    @abc.abstractmethod
    def task_label(self) -> str:
        """
        :return: The task label associated with this task variant. All task variants
            with the same task should have the same task label.
        """

    @property
    @abc.abstractmethod
    def variant_label(self) -> str:
        """
        :return: The variant label associated with this task variant. All task variants
            with the same extrinsic parameters should have the same variant label.
        """


TaskVariantType = typing.TypeVar("TaskVariantType", bound=AbstractTaskVariant)
