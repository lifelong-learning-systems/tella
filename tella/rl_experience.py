import typing
import gym
from .curriculum import Experience
from .validation import validate_params

Observation = typing.TypeVar("Observation")
Action = typing.TypeVar("Action")
Reward = float
Done = bool
Transition = typing.Tuple[Observation, Action, Reward, Done, Observation]
"""
A transition is a tuple with data representing a transition of an MDP.
"""

ActionFn = typing.Callable[[Observation], Action]
"""
A function that takes an Observation and returns an Action.
"""

RLExperience = Experience[ActionFn, Transition, gym.Env]
"""
An RLExperience is an Experience that takes an ActionFn and produces an iterable
of Transitions
"""


class RLEpisodeExperience(RLExperience):
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
        params: typing.Optional[typing.Dict] = None,
    ) -> None:
        if params is None:
            params = {}
        self._task_cls = task_cls
        self._params = params
        self._num_episodes = num_episodes
        self._env = None

    def validate(self) -> None:
        return validate_params(self._task_cls, list(self._params.keys()))

    def info(self) -> gym.Env:
        if self._env is None:
            self._env = self._task_cls(**self._params)
        return self._env

    def generate(self, action_fn: ActionFn) -> typing.Iterable[Transition]:
        env = self.info()
        for _i_episode in range(self._num_episodes):
            obs = env.reset()
            done = False
            while not done:
                action = action_fn(obs)
                next_obs, reward, done, info = env.step(action)
                yield obs, action, reward, done, next_obs
                obs = next_obs
        self._env.close()
        self._env = None
