"""
Copyright © 2021-2022 The Johns Hopkins University Applied Physics Laboratory LLC

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

import abc
import typing

import gym

from .curriculum import (
    Action,
    Observation,
    Transition,
)


class ContinualRLAgent:
    """
    tella's base class for a continual reinforcement learning agent. This class
    implements placeholder event handlers for the start and end of each component in a
    :class:`AbstractCurriculum <tella.curriculum.AbstractCurriculum>`, as well as
    abstract methods for interacting with each of the curriculum's :class:`gym.Env`.

    The six optional methods for curriculum events are:

    * :meth:`ContinualRLAgent.block_start` and :meth:`ContinualRLAgent.block_end`
    * :meth:`ContinualRLAgent.task_start` and :meth:`ContinualRLAgent.task_end`
    * :meth:`ContinualRLAgent.task_variant_start` and :meth:`ContinualRLAgent.task_variant_end`

    The two required methods for interacting with environments are:

    * :meth:`ContinualRLAgent.choose_actions()`
    * :meth:`ContinualRLAgent.receive_transitions()`

    """

    def __init__(
        self,
        rng_seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        config_file: typing.Optional[str] = None,
    ) -> None:
        """
        :param rng_seed: Random number generator seed.
        :param observation_space: The observation space from the :class:`gym.Env`.
        :param action_space: The action space from the :class:`gym.Env`.
        :param num_envs: The number of environments that will be used for :class:`gym.vector.VectorEnv`.
        :param config_file: Path to a config file for the agent or None if no config.
        """
        self.rng_seed = rng_seed
        self.is_learning_allowed: bool = False
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.config_file = config_file

        # Set RNG seeds on observation and action spaces for .sample() method
        self.observation_space.seed(self.rng_seed)
        self.action_space.seed(self.rng_seed)

    def block_start(self, is_learning_allowed: bool) -> None:
        """
        Called at the start of a new :class:`Block <tella.curriculum.Block>`.

        The next method called will be :meth:`ContinualRLAgent.task_start()`.

        :param is_learning_allowed: Whether the block is a
            :class:`LearnBlock <tella.curriculum.LearnBlock>`
            or an :class:`EvalBlock <tella.curriculum.EvalBlock>`.

        .. NOTE::
            When using the tella CLI, the attribute `ContinualRLAgent.is_learning_allowed`
            is automatically updated before this method is called.
            (This occurs in :func:`tella.experiment.run`.)
        """
        pass

    def task_start(
        self,
        task_name: typing.Optional[str],
    ) -> None:
        """
        Called at the start of a new :class:`TaskBlock <tella.curriculum.TaskBlock>`.

        The next method called will be :meth:`ContinualRLAgent.task_variant_start()`.

        :param task_name: An optional value indicating the name of the task
        """
        pass

    def task_variant_start(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        """
        Called at the start of a new :class:`TaskVariant <tella.curriculum.TaskVariant>`.

        The task variant experiences will occur between this method and
        :meth:`ContinualRLAgent.task_variant_end()`.

        The next method called will be :meth:`ContinualRLAgent.choose_actions()`.

        :param task_name: An optional value indicating the name of the task
        :param variant_name: An optional value indicating the name of the task variant
        """
        pass

    def task_variant_end(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        """
        Called at the end of the current :class:`TaskVariant <tella.curriculum.TaskVariant>`.

        The next method called will be :meth:`ContinualRLAgent.task_variant_start()` if there
        are more task variants in the task block, otherwise :meth:`ContinualRLAgent.task_end()`.

        :param task_name: An optional value indicating the name of the task
        :param variant_name: An optional value indicating the name of the task variant
        """
        pass

    def task_end(
        self,
        task_name: typing.Optional[str],
    ) -> None:
        """
        Called at the end of the current :class:`TaskBlock <tella.curriculum.TaskBlock>`.

        The next method called will be :meth:`ContinualRLAgent.task_start()` if there
        are more tasks in the block, otherwise :meth:`ContinualRLAgent.block_end()`.

        :param task_name: An optional value indicating the name of the task
        """
        pass

    def block_end(self, is_learning_allowed: bool) -> None:
        """
        Called at the end of the current :class:`Block <tella.curriculum.Block>`.

        The next method called will be :meth:`ContinualRLAgent.block_start()`
        if there are more blocks in the curriculum, otherwise the program will end.

        :param is_learning_allowed: The same data passed into the previous call to
            :meth:`ContinualRLAgent.block_start()`.
        """
        pass

    @abc.abstractmethod
    def choose_actions(
        self, observations: typing.List[typing.Optional[Observation]]
    ) -> typing.List[typing.Optional[Action]]:
        """
        Asks the agent for an action for each observation in the list passed in.
        The observations will be consistent with whatever :class:`gym.vector.VectorEnv`
        returns. The actions should be passable to :meth:`gym.vector.VectorEnv.step`.

        ... e.g. ::

            observations = vector_env.reset()
            actions = agent.choose_actions(observations)
            ... = vector_env.step(actions)

        If there are environments that are done, but no more new steps can be taken
        due to limitations from the curriculums, a ``None`` will be passed in place of
        an observation. This is done to preserve the ordering of observations.

        In the case that ``observations[i] is None``, then the i'th action returned
        will be disregarded. The agent can instead return ``None`` in that place.

        ... i.e. ::

            observations = ...
            observations[2] = None
            actions = agent.choose_actions(observations)
            assert actions[2] is None

        The next method called will be :meth:`ContinualRLAgent.receive_transitions()`.

        :param observations: The observations from the environment.
        :return: Actions that can be passed to :meth:`gym.vector.VectorEnv.step()`.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def receive_transitions(
        self, transitions: typing.List[typing.Optional[Transition]]
    ) -> None:
        """
        Gives the transitions that result from calling :meth:`gym.Env.step()` with given actions.

        ... e.g. ::

            actions = agent.choose_actions(observations)
            next_obs, rewards, dones, infos = vector_env.step(actions)
            transitions = zip(observations, actions, rewards, dones, next_obs)
            agent.receive_transitions(transitions)

        The next method called will be :meth:`ContinualRLAgent.task_variant_end()` if all episodes
        have ended, otherwise :meth:`ContinualRLAgent.choose_actions()`.

        :param transitions: The transitions from the agent-environment interaction.
        """
        raise NotImplementedError
