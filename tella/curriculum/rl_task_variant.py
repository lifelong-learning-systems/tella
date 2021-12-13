import typing
import gym
from ..env import L2MEnv
from .task_variant import AbstractTaskVariant
from ..validation import validate_params

Observation = typing.TypeVar("Observation")
Action = typing.TypeVar("Action")
Reward = float
Done = bool
NextObservation = Observation

StepData = typing.Tuple[Observation, Action, Reward, Done, NextObservation]
"""
A tuple with data containing data from a single step in an MDP.
The last item of the tuple is the observation resulting from applying the action
to the observation (i.e. Next observation).
"""

ActionFn = typing.Callable[
    [typing.List[typing.Optional[Observation]]], typing.List[typing.Optional[Action]]
]
"""
A function that takes a list of Observations and returns a list of Actions, one
for each observation.
"""

AbstractRLTaskVariant = AbstractTaskVariant[
    ActionFn, typing.Iterable[StepData], gym.Env
]
"""
An AbstractRLTaskVariant is an TaskVariant that takes an ActionFn as input
and produces an Iterable[StepData]. It also  returns a :class:`gym.Env` as the
Information.
"""


class EpisodicTaskVariant(AbstractRLTaskVariant):
    """
    Represents a TaskVariant that consists of a set number of episodes in a
    :class:`gym.Env`.

    This is a concrete subclass of the :class:`AbstractRLTaskVariant`,
    that takes an :type:`ActionFn` and returns an iterable of :type:`StepData`.
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

    def total_episodes(self):
        return self._num_episodes
        
    def validate(self) -> None:
        return validate_params(self._task_cls, list(self._params.keys()))

    def _make_env(self) -> gym.Env:
        #return self._task_cls(**self._params)
        return L2MEnv(self._task_cls(**self._params), self.data_logger,self.logger_info) 

    def set_logger_info(self, data_logger,block_num, block_type, exp_num):
        self.data_logger = data_logger
        self.logger_info = {
            'block_num': block_num,
            'block_type': block_type,
            'task_params': self._params,
            'task_name' : self._task_cls.__name__,
            'worker_id': 'dummy',
            'exp_num':exp_num
        }

    def info(self) -> gym.Env:
        if self._env is None:
            vector_env_cls = gym.vector.AsyncVectorEnv
            if self._num_envs == 1:
                vector_env_cls = gym.vector.SyncVectorEnv
            self._env = vector_env_cls([self._make_env for _ in range(self._num_envs)])
        return self._env

    def generate(self, action_fn: ActionFn) -> typing.Iterable[StepData]:
        env = self.info()

        num_episodes_finished = 0

        # data to keep track of which observations to mask out (set to None)
        episode_ids = list(range(self._num_envs))
        next_episode_id = episode_ids[-1] + 1

        observations = env.reset()
        while num_episodes_finished < self._num_episodes:
            # mask out any environments that have episode id above max episodes
            mask = [ep_id >= self._num_episodes for ep_id in episode_ids]

            # replace masked environment observations with None
            masked_observations = _where(mask, None, observations)

            # query for the actions
            actions = action_fn(masked_observations)

            # replace masked environment actions with random action
            unmasked_actions = _where(mask, env.single_action_space.sample(), actions)

            # step in the VectorEnv
            next_observations, rewards, dones, infos = env.step(unmasked_actions)

            # yield all the non masked transitions
            for i in range(self._num_envs):
                if not mask[i]:
                    # FIXME: if done[i] == True, then we need to return info[i]["terminal_observation"]
                    yield (
                        observations[i],
                        actions[i],
                        rewards[i],
                        dones[i],
                        next_observations[i],
                    )

                # increment episode ids if episode ended
                if dones[i]:
                    episode_ids[i] = next_episode_id
                    next_episode_id += 1

            observations = next_observations
            num_episodes_finished += sum(dones)
        self._env.close()
        self._env = None


def _where(
    condition: typing.List[bool], replace_value: typing.Any, original_list: typing.List
) -> typing.List:
    """
    Replaces elements in `original_list[i]` with `replace_value` where the `condition[i]`
    is True.

    :param condition: List of booleans indicating where to put replace_value
    :param replace_value: The value to insert into the list
    :param original_list: The list of values to modify
    :return: A new list with replace_value inserted where condition elements are True
    """
    return [
        replace_value if condition[i] else original_list[i]
        for i in range(len(condition))
    ]
