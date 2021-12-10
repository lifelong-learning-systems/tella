import typing
import abc
import gym

from .metrics.rl import default_metrics, RLMetricAccumulator
from ..curriculum.rl_task_variant import (
    StepData,
    Observation,
    Action,
    AbstractRLTaskVariant,
)
from .continual_learning_agent import ContinualLearningAgent, Metrics


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
        2. view_transitions, which :meth:`ContinualRLAgent.learn_experience`
            calls with the result of :meth:`RLTaskVariant.generate`.

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        num_envs: int,
        metric: typing.Optional[RLMetricAccumulator] = None,
    ) -> None:
        """
        The constructor for the Continual RL agent.

        :param observation_space: The observation space from the :class:`gym.Env`.
        :param action_space: The action space from the :class:`gym.Env`.
        :param num_envs: The number of environments that will be used for :class:`gym.vector.VectorEnv`.
        """
        super().__init__()
        if metric is None:
            metric = default_metrics()
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_envs = num_envs
        self.metric = metric

    def learn_task_variant(self, task_variant: AbstractRLTaskVariant) -> Metrics:
        """
        Passes :meth:`ContinualRLAgent.choose_action` to :meth:`RLTaskVariant.generate`
        to generate the iterable of :class:`MDPTransition`, then each transition is
        passed to :meth:`ContinualRLAgent.view_transition` for learning.
        """
        for transition in task_variant.generate(self.choose_action):
            self.metric.track(transition)
            self.view_transition(transition)
        return self.metric.calculate()

    def eval_task_variant(self, task_variant: AbstractRLTaskVariant) -> Metrics:
        """
        Passes :meth:`ContinualRLAgent.choose_action` to :meth:`RLTaskVariant.generate`
        to generate the iterable of :class:`MDPTransition`.
        """
        for transition in task_variant.generate(self.choose_action):
            self.metric.track(transition)
        return self.metric.calculate()

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
        due to limitations from the curricula, a None will be passed inplace of
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
    def view_transition(self, step_data: StepData) -> None:
        """
        Gives the transition that results from calling :meth:`gym.Env.step()` with a given action.

        .. e.g.

            action = ...
            next_obs, reward, done, info = env.step(action)
            transition = (obs, action, reward, done, next_obs)
            agent.view_transition(transition)

        NOTE: when using vectorized environments (i.e. when `Agent.choose_action`
        receives multiple observations, or when `self.num_envs > 1`),
        :meth:`Agent.step_transition` is called separately for each resulting
        transition. I.e. :meth:`Agent.step_transition` is called `self.num_envs` times.

        The next method called would be :meth:`Agent.choose_action()` if done is False,
        otherwise :meth:`Agent.task_variant_end()`.
        """
        pass
