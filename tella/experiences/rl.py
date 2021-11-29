import typing
import gym
from .experience import Experience
from ..validation import validate_params

Observation = typing.TypeVar("Observation")
Action = typing.TypeVar("Action")
Reward = float
Done = bool

MDPTransition = typing.Tuple[Observation, Action, Reward, Done, Observation]
"""
A tuple with data representing a transition of an MDP. The last item of the tuple
is the resulting observation that happens after applying action to the first
observation in the tuple.
"""

ActionFn = typing.Callable[
    [typing.List[typing.Optional[Observation]]], typing.List[typing.Optional[Action]]
]
"""
A function that takes a list of Observations and returns a list of Actions, one
for each observation.
"""

RLExperience = Experience[ActionFn, typing.Iterable[MDPTransition], gym.Env]
"""
An RLExperience is an Experience that takes an ActionFn as input, produces an
Iterable[MDPTransition], and returns a :class:`gym.Env` as the Information.
"""


class LimitedEpisodesExperience(RLExperience):
    """
    Represents an experience that consists of a set number of episodes in a :class:`gym.Env`.

    This is a subclass of the :class:`RLExperience`, which is defined as an :class:`Experience`
    that takes an :type:`ActionFn` and returns an iterable of :type:`Transition`.
    """

    def __init__(
        self,
        task_cls: typing.Type[gym.Env],
        *,
        num_episodes: int,
        num_envs: typing.Optional[int] = None,
        params: typing.Optional[typing.Dict] = None,
    ) -> None:
        if num_envs is None:
            num_envs = 1
        if params is None:
            params = {}
        assert num_envs > 0

        self._task_cls = task_cls
        self._params = params
        self._num_episodes = num_episodes
        self._num_envs = num_envs
        self._env = None

    def validate(self) -> None:
        return validate_params(self._task_cls, list(self._params.keys()))

    def _make_env(self) -> gym.Env:
        return self._task_cls(**self._params)

    def info(self) -> gym.Env:
        if self._env is None:
            vector_env_cls = gym.vector.AsyncVectorEnv
            if self._num_envs == 1:
                vector_env_cls = gym.vector.SyncVectorEnv
            self._env = vector_env_cls([self._make_env for _ in range(self._num_envs)])
        return self._env

    def generate(self, action_fn: ActionFn) -> typing.Iterable[MDPTransition]:
        env = self.info()
        num_episodes_left = self._num_episodes
        observations = env.reset()
        while num_episodes_left > 0:
            # FIXME: this always uses all environments in the vector env. this will leak extra episodes to the consumer
            actions = action_fn(observations)
            next_observations, rewards, dones, infos = env.step(actions)
            for i in range(self._num_envs):
                # FIXME: if done[i] == True, then we need to return info[i]["terminal_observation"]
                yield (
                    observations[i],
                    actions[i],
                    rewards[i],
                    dones[i],
                    next_observations[i],
                )
            observations = next_observations
            num_episodes_left -= sum(dones)
        self._env.close()
        self._env = None
