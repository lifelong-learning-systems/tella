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
import gym

Observation = typing.TypeVar("Observation")
Action = typing.TypeVar("Action")


class Agent(abc.ABC):
    """
    The base Agent class for continual reinforcement learning.

    The only requirement is to implement :meth:`Agent.step()`.
    """

    def block_start(self, is_learning_allowed: bool) -> None:
        """
        Signifies a new block (either learning or evaluation) is about to start.

        The next method called would be :meth:`Agent.task_start()`.

        :param is_learning_allowed: Whether the block is a learning block or
            an evaluation block.
        """
        pass

    def task_start(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a new task is about to start. `task_info`
        may contain task id/label or task parameters.

        The next method called would be :meth:`Agent.episode_start()`.

        :param observation_space: The observation space from the :class:`gym.Env`.
        :param action_space: The action space from the :class:`gym.Env`.
        :param task_name: An optional value indicating the name of the task
        :param variant_name: An optional value indicating the name of the task variant
        """
        pass

    def episode_start(self) -> None:
        """
        Signifies a new episode is about to start.

        The next method called would be :meth:`Agent.step()`.
        """
        pass

    @abc.abstractmethod
    def step(self, observation: Observation) -> Action:
        """
        Asks the agent for an action given an observation from the environment.

        .. e.g.

            action = agent.step(obs)
            ... = env.step(action)

        The next method called would be :meth:`Agent.step_result()`.

        :param observation: The observation from the environment.
        :return: An action that can be passed to :meth:`gym.Env.step()`.
        """
        pass

    def step_result(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        done: bool,
        next_observation: Observation,
    ) -> bool:
        """
        Gives the result of calling :meth:`gym.Env.step()` with a given action.

        .. e.g.

            action = agent.step(obs)
            next_obs, reward, done, info = env.step(action)
            keep_going = agent.step_result(obs, action, reward, done, next_obs)
            done = done or not keep_going

        Also allows the agent to end episode early by returning False from this
        method. If True is returned, this indicates that the episode should continue
        unless done is True.

        The next method called would be :meth:`Agent.get_action()` if done is False,
        otherwise :meth:`Agent.episode_end()`.

        :return: A boolean indicating whether to continue with the episode.
        """
        return True

    def episode_end(self) -> None:
        """
        Signifies an episode has just ended.

        The next method called would be :meth:`Agent.episode_start()` if
        there are more episodes for the task, otherwise :meth:`Agent.task_end()`.
        """
        pass

    def task_end(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a task has just ended.

        The next method called would be :meth:`Agent.task_start()` if there
        are more tasks in the block, otherwise :meth:`Agent.block_end()`.

        :param observation_space: The observation space from the :class:`gym.Env`.
        :param action_space: The action space from the :class:`gym.Env`.
        :param task_name: An optional value indicating the name of the task
        :param variant_name: An optional value indicating the name of the task variant
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

    @abc.abstractmethod
    def save_internal_state(self, path: str) -> bool:
        """
        Tells the agent to save any internal parameters to `path`. This is
        to enable restoring agent to a previous state - i.e. checkpointing.

        NOTE: this can be in any format, the only requirement is that
        :meth:`Agent.restore_internal_state(path)` can load in data that was saved.

        This function will be called by code external to the Agent.

        :return: A boolean indicating whether file creation and saving was successful
        """
        return False

    @abc.abstractmethod
    def restore_internal_state(self, path: str) -> bool:
        """
        Tells the agent to restore internal paramters that were previously saved
        by :meth:`Agent.save_internal_state(path)`. This is to enable recovering
        from a previous crash by restoring state to a checkpoint.

        This function will be called by code external to the Agent.

        :return: A boolean indicating whether restoration was successful
        """
        return False
