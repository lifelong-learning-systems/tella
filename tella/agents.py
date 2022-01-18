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
    AbstractRLTaskVariant,
    Action,
    Observation,
    TaskVariantType,
    Transition,
)


class ContinualLearningAgent(abc.ABC, typing.Generic[TaskVariantType]):
    """
    The base class for a continual learning agent. A CL Agent is an agent that
    can consume some task variant of a generic type (:class:`TaskVariantType`).

    The only requirement is to implement :meth:`ContinualLearningAgent.consume_task_variant()`,
    which takes an object of type :class:`TaskVariantType`.
    """

    def __init__(self, rng_seed: int) -> None:
        """
        Agent constructor.

        For experiment repeatability, all agents with non-deterministic methods are expected
        to seed their random number generators (RNG) based on the parameter provided here.

        :param rng_seed: The seed to be used in setting random number generators.
        """
        super().__init__()
        self.rng_seed = rng_seed
        self.is_learning_allowed: bool = False

    def block_start(self, is_learning_allowed: bool) -> None:
        """
        Signifies a new block (either learning or evaluation) is about to start.

        The next method called would be :meth:`ContinualLearningAgent.task_start()`.

        NOTE: the attribute :attr:`ContinualLearningAgent.is_learning_allowed` is
            also set outside of this method.

        :param is_learning_allowed: Whether the block is a learning block or
            an evaluation block.
        """
        pass

    def task_start(
        self,
        task_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a new task is about to start. `task_info`
        may contain task id/label or task parameters.

        The next method called would be :meth:`ContinualLearningAgent.task_variant_start()`.

        :param task_name: An optional value indicating the name of the task
        """
        pass

    def task_variant_start(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a new task variant is about to start. `task_info`
        may contain task id/label or task parameters.

        The next method called would be :meth:`ContinualLearningAgent.consume_experience()`.

        :param task_name: An optional value indicating the name of the task
        :param variant_name: An optional value indicating the name of the task variant
        """
        pass

    @abc.abstractmethod
    def learn_task_variant(self, task_variant: TaskVariantType) -> None:
        """
        Passes an object of type :class:`TaskVariantType` to the agent to consume for learning.

        The next method called would be :meth:`ContinualLearningAgent.task_variant_end()`.
        """

    @abc.abstractmethod
    def eval_task_variant(self, task_variant: TaskVariantType) -> None:
        """
        Passes an object of type :class:`TaskVariantType` to the agent to consume for evaluation.

        The next method called would be :meth:`ContinualLearningAgent.task_variant_end()`.
        """

    def task_variant_end(
        self,
        task_name: typing.Optional[str],
        variant_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a task variant has just ended.

        The next method called would be :meth:`Agent.task_variant_start()` if there
        are more task variants in the block, otherwise :meth:`Agent.task_end()`.

        :param task_name: An optional value indicating the name of the task
        :param variant_name: An optional value indicating the name of the task variant
        """
        pass

    def task_end(
        self,
        task_name: typing.Optional[str],
    ) -> None:
        """
        Signifies interaction with a task has just ended.

        The next method called would be :meth:`Agent.task_start()` if there
        are more tasks in the block, otherwise :meth:`Agent.block_end()`.

        :param task_name: An optional value indicating the name of the task
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


class ContinualRLAgent(ContinualLearningAgent[AbstractRLTaskVariant]):
    """
    The base class for a continual reinforcement learning agent. This class
    consumes an experience of type :class:`AbstractRLTaskVariant`.

    This class implements the :meth:`ContinualLearningAgent.learn_experience`
    and the :meth:`ContinualLearningAgent.eval_experience`, and exposes two
    new required methods for subclasses to implement:

        1. choose_actions, which :meth:`ContinualRLAgent.learn_experience`
            and :meth:`ContinualRLAgent.eval_experience` pass to
            :meth:`RLTaskVariant.generate`.
        2. receive_transitions, which :meth:`ContinualRLAgent.learn_experience`
            calls with the result of :meth:`RLTaskVariant.generate`.

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
        The constructor for the Continual RL agent.

        :param rng_seed: Random number generator seed.
        :param observation_space: The observation space from the :class:`gym.Env`.
        :param action_space: The action space from the :class:`gym.Env`.
        :param num_envs: The number of environments that will be used for :class:`gym.vector.VectorEnv`.
        :param config_file: Path to a config file for the agent or None if no config.
        """
        super().__init__(rng_seed)
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.config_file = config_file

        # Set RNG seeds on observation and action spaces for .sample() method
        self.observation_space.seed(self.rng_seed)
        self.action_space.seed(self.rng_seed)

    def learn_task_variant(self, task_variant: AbstractRLTaskVariant) -> None:
        """
        Passes :meth:`ContinualRLAgent.choose_action` to :meth:`RLTaskVariant.generate`
        to generate the iterable of :class:`MDPTransition`, then each transition is
        passed to :meth:`ContinualRLAgent.receive_transition` for learning.
        """
        for transition in task_variant.generate(self.choose_action):
            self.receive_transition(transition)

    def eval_task_variant(self, task_variant: AbstractRLTaskVariant) -> None:
        """
        Passes :meth:`ContinualRLAgent.choose_action` to :meth:`RLTaskVariant.generate`
        to generate the iterable of :class:`MDPTransition`.
        """
        # FIXME: RNN agents need to know when an episode ends, and so need to see some
        #  of the transition data even during eval. https://github.com/darpa-l2m/tella/issues/187
        for transition in task_variant.generate(self.choose_action):
            pass

    @abc.abstractmethod
    def choose_action(
        self, observations: typing.List[typing.Optional[Observation]]
    ) -> typing.List[typing.Optional[Action]]:
        """
        Asks the agent for an action for each observation in the list passed in.
        The observations will be consistent with whatever :class:`gym.vector.VectorEnv`
        returns. The actions should be passable to :meth:`gym.vector.VectorEnv.step`.

        .. e.g.

            observations = vector_env.reset()
            actions = agent.choose_action(observations)
            ... = vector_env.step(actions)

        If there are environments that are done, but no more new steps can be taken
        due to limitations from the curriculums, a None will be passed inplace of
        an observation. This is done to preserve the ordering of observations.

        In the case that `observations[i] is None`, then the i'th action returned
        should also be `None`.

        .. i.e.

            observations = ...
            observations[2] = None
            actions = agent.choose_action(observations)
            assert actions[2] is None

        :param observations: The observations from the environment.
        :return: Actions that can be passed to :meth:`gym.vector.VectorEnv.step()`.
        """
        pass

    @abc.abstractmethod
    def receive_transition(self, transition: Transition) -> None:
        """
        Gives the transition that results from calling :meth:`gym.Env.step()` with a given action.

        .. e.g.

            action = ...
            next_obs, reward, done, info = env.step(action)
            transition = (obs, action, reward, done, next_obs)
            agent.receive_transition(transition)

        NOTE: when using vectorized environments (i.e. when `Agent.choose_action`
        receives multiple observations, or when `self.num_envs > 1`),
        :meth:`Agent.step_transition` is called separately for each resulting
        transition. I.e. :meth:`Agent.step_transition` is called `self.num_envs` times.

        The next method called would be :meth:`Agent.choose_action()` if done is False,
        otherwise :meth:`Agent.task_variant_end()`.
        """
        # FIXME: This method should take an iterable as an argument as does `choose_action`.
        #  This would make their formatting match, https://github.com/darpa-l2m/tella/issues/189,
        #  and simplify tracking multiple environments, https://github.com/darpa-l2m/tella/issues/196
        pass
