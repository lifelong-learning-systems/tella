import typing
import abc
import gym

from .metrics.rl import default_metrics, RLMetricAccumulator
from ..experiences.rl import MDPTransition, Observation, Action, RLExperience
from .continual_learning_agent import ContinualLearningAgent, Metrics


class ContinualRLAgent(ContinualLearningAgent[RLExperience]):
    """
    The base class for a continual reinforcement learning agent. This class
    consumes an experience of type :class:`RLExperience`.

    This class implements the :meth:`ContinualLearningAgent.consume_experience`,
    and exposes two new required methods for subclasses to implement:

        1. step_observe, which :meth:`ContinualRLAgent.consume_experience`
            passes to :meth:`RLExperience.generate`.
        2. step_transition, which :meth:`ContinualRLAgent.consume_experience`
            calls with the result of :meth:`RLExperience.generate`.

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

    def consume_experience(self, experience: RLExperience) -> Metrics:
        """
        Passes :meth:`ContinualRLAgent.step_observe` to :meth:`RLExperience.generate`
        to generate the iterable of :class:`MDPTransition`.

        If this is in a learning block (i.e. self.is_learning_allowed is True),
        then each transition is passed to :meth:`ContinualRLAgent.step_transition`.
        """
        for transition in experience.generate(self.step_observe):
            self.metric.track(transition)
            if self.is_learning_allowed:
                keep_going = self.step_transition(transition)
                if not keep_going:
                    break
        return self.metric.calculate()

    @abc.abstractmethod
    def step_observe(
        self, observations: typing.List[typing.Optional[Observation]]
    ) -> typing.List[typing.Optional[Action]]:
        """
        Asks the agent for an action for each observation in the list passed in.
        The observations will be consistent with whatever :class:`gym.vector.VectorEnv`
        returns. The actions should be passable to :meth:`gym.vector.VectorEnv.step`.

        .. e.g.

            observations = vector_env.reset()
            actions = agent.step_observe(observations)
            ... = vector_env.step(actions)

        If there are environments that are done, but no more new steps can be taken
        due to limitations from the curricula, a None will be passed inplace of
        an observation. This is done to preserve the ordering of observations.

        In the case that `observations[i] is None`, then the i'th action returned
        should also be `None`.

        .. i.e.

            observations = ...
            observations[2] = None
            actions = agent.step_observe(observations)
            assert actions[2] is None

        :param observations: The observations from the environment.
        :return: Actions that can be passed to :meth:`gym.vector.VectorEnv.step()`.
        """
        pass

    @abc.abstractmethod
    def step_transition(self, transition: MDPTransition) -> bool:
        """
        Gives the transition that results from calling :meth:`gym.Env.step()` with a given action.

        .. e.g.

            action = ...
            next_obs, reward, done, info = env.step(action)
            transition = (obs, action, reward, done, next_obs)
            keep_going = agent.step_transition(transition)

        Also allows the agent to stop consuming an experience early by returning
        False from this method. If True is returned, this indicates that the episode should continue
        unless done is True. See :meth:`ContinualRLAgent.consume_experience` for
        more details.

        NOTE: when using vectorized environments (i.e. when `Agent.step_observe`
        receives multiple observations, or when `self.num_envs > 1`),
        :meth:`Agent.step_transition` is called separately for each resulting
        transition. I.e. :meth:`Agent.step_transition` is called `self.num_envs` times.

        The next method called would be :meth:`Agent.step_observe()` if done is False,
        otherwise :meth:`Agent.episode_end()`.

        :return: A boolean indicating whether to continue with the episode.
        """
        pass
