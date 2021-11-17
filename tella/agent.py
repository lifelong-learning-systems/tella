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

    The only requirement is to implement :meth:`Agent.get_action()`.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    def save(self, path: str, **kwargs) -> None:
        """
        Save the parameters to `path`.
        """
        pass

    def load(self, path: str, **kwargs) -> None:
        """
        Load paramters from `path`.
        """
        pass

    def handle_block_start(self, is_learning_allowed: bool, **kwargs) -> None:
        """
        Signifies a new block (either learning or evaluation) is about to start.

        The next method called would be :meth:`Agent.handle_task_start()`.

        :param is_learning_allowed: Whether the block is a learning block or
            an evaluation block.
        """
        pass

    def handle_task_start(
        self, task_info: typing.Optional[typing.Dict[str, typing.Any]], **kwargs
    ) -> None:
        """
        Signifies interaction with a new task is about to start. `task_info`
        may contain task id/label or task parameters.

        The next method called would be :meth:`Agent.handle_episode_start()`.

        :param task_info: TODO what is this?
        """
        pass

    def handle_episode_start(self, **kwargs) -> None:
        """
        Signifies a new episode is about to start.

        The next method called would be :meth:`Agent.get_action()`.
        """
        pass

    @abc.abstractmethod
    def get_action(self, observation: Observation, **kwargs) -> Action:
        """
        Asks the agent for an action given an observation from the environment.

        The next method called would be :meth:`Agent.handle_step_result()`.

        :param observation: The observation from the environment.
        :return: An action that can be passed to :meth:`gym.Env.step()`.
        """
        pass

    def handle_step_result(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        done: bool,
        next_observation: Observation,
        **kwargs,
    ) -> None:
        """
        Gives the result of calling :meth:`gym.Env.step()` with a given action.

        .. e.g.
            action = agent.get_action(obs)
            next_obs, reward, done, info = env.step(action)
            agent.handle_step_result(obs, action, reward, done, next_obs)

        The next method called would be :meth:`Agent.get_action()` if done is False,
        otherwise :meth:`Agent.handle_episode_end()`.
        """
        pass

    def handle_episode_end(self, **kwargs) -> None:
        """
        Signifies an episode has just ended.

        The next method called would be :meth:`Agent.handle_episode_start()` if
        there are more episodes for the task, otherwise :meth:`Agent.handle_task_end()`.
        """
        pass

    def handle_task_end(
        self, task_info: typing.Optional[typing.Dict[str, typing.Any]], **kwargs
    ) -> None:
        """
        Signifies interaction with a task has just ended.

        The next method called would be :meth:`Agent.handle_task_start()` if there
        are more tasks in the block, otherwise :meth:`Agent.handle_block_end()`.

        :param task_info: The same data passed into the last :meth:`Agent.handle_task_start()`.
        """
        pass

    def handle_block_end(self, is_learning_allowed: bool, **kwargs) -> None:
        """
        Signifies the end of a block.

        The next method called would be :meth:`Agent.handle_block_start()`
        if there are more blocks, otherwise the program would end.

        :param is_learning_allowed: The same data passed into the last :meth:`Agent.handle_block_end()`.
        """
        pass

    def learning_rollout(self, env: gym.Env, **kwargs) -> None:
        """
        Default implementation of interacting with the environment during a learning
        block.

        :param env: The :class:`gym.Env` to interact with.
        """
        self.handle_episode_start(**kwargs)
        obs = env.reset()
        done = False
        while not done:
            action = self.get_action(obs, **kwargs)
            next_obs, reward, done, info = env.step(action)
            self.handle_step_result(obs, next_obs, action, reward, done, **kwargs)
            if not done:
                obs = next_obs
        self.handle_episode_end(**kwargs)
