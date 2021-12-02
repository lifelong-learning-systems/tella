import pytest
import typing
import gym
from tella.task_variants.rl import EpisodicTaskVariant


class DummyEnv(gym.Env):
    def __init__(self, a: int, b: float, c: str) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.observation_space = gym.spaces.Discrete(5)
        self.action_space = gym.spaces.Discrete(2)
        self.i = 0

    def reset(self):
        self.i = 0
        return self.observation_space.sample()

    def step(
        self, action: int
    ) -> typing.Tuple[int, float, bool, typing.Dict[str, typing.Any]]:
        self.i += 1
        done = self.i >= 5
        return self.observation_space.sample(), 0.0, done, {}


def random_action(
    observations: typing.List[typing.Optional[int]],
) -> typing.List[typing.Optional[int]]:
    return [None if obs is None else 0 for obs in observations]


@pytest.mark.parametrize("num_envs", [1, 2, 3, 4])
def test_num_episodes(num_envs: int):
    for num_episodes in [1, 2, 3, 4, 5, 6, 7, 8]:
        exp = EpisodicTaskVariant(
            DummyEnv,
            num_episodes=num_episodes,
            num_envs=num_envs,
            params={"a": 1, "b": 3.0, "c": "a"},
        )
        steps = list(exp.generate(random_action))
        assert len(steps) == 5 * num_episodes
        assert (
            sum([done for obs, action, reward, done, next_obs in steps]) == num_episodes
        )


def test_validate():
    pass


def test_terminal_observations():
    pass
