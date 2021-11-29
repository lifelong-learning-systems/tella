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

    The only requirement is to implement :meth:`Agent.step_observe()`.
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space) -> None:
        """
        The constructor for the Agent.

        :param observation_space: The observation space from the :class:`gym.Env`.
        :param action_space: The action space from the :class:`gym.Env`.
        """
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space

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
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a new task is about to start. `task_info`
        may contain task id/label or task parameters.

        The next method called would be :meth:`Agent.episode_start()`.

        :param task_name: An optional value indicating the name of the task
        :param variant_name: An optional value indicating the name of the task variant
        """
        pass

    def episode_start(self) -> None:
        """
        Signifies a new episode is about to start.

        The next method called would be :meth:`Agent.step_observe()`.
        """
        pass

    @abc.abstractmethod
    def step_observe(self, observation: Observation) -> Action:
        """
        Asks the agent for an action given an observation from the environment.

        .. e.g.

            action = agent.step_observe(obs)
            ... = env.step(action)

        The next method called would be :meth:`Agent.step_reward()`.

        :param observation: The observation from the environment.
        :return: An action that can be passed to :meth:`gym.Env.step()`.
        """
        pass

    def step_reward(
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

            action = agent.step_observe(obs)
            next_obs, reward, done, info = env.step(action)
            keep_going = agent.step_reward(obs, action, reward, done, next_obs)
            done = done or not keep_going

        Also allows the agent to end episode early by returning False from this
        method. If True is returned, this indicates that the episode should continue
        unless done is True.

        The next method called would be :meth:`Agent.step_observe()` if done is False,
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
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a task has just ended.

        The next method called would be :meth:`Agent.task_start()` if there
        are more tasks in the block, otherwise :meth:`Agent.block_end()`.

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
