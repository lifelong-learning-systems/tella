import pytest
import typing
import gym
from tella.curriculum import EpisodicTaskVariant


class DummyEnv(gym.Env):
    def __init__(
        self, a: int, b: float, c: str, observations=None, max_steps=5
    ) -> None:
        super().__init__()
        if observations is None:
            observations = []
        self.a = a
        self.b = b
        self.c = c
        self.observation_space = gym.spaces.Discrete(5)
        self.action_space = gym.spaces.Discrete(2)
        self.observations = observations
        self.i = 0
        self.max_steps = max_steps

    def reset(self):
        self.i = 0
        return (
            self.observation_space.sample()
            if len(self.observations) == 0
            else self.observations.pop(0)
        )

    def step(
        self, action: int
    ) -> typing.Tuple[int, float, bool, typing.Dict[str, typing.Any]]:
        self.i += 1
        done = self.i >= self.max_steps
        obs = (
            self.observation_space.sample()
            if len(self.observations) == 0
            else self.observations.pop(0)
        )
        return obs, 0.0, done, {}


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
            rng_seed=0,
        )
        steps = list(exp.generate(random_action))
        assert len(steps) == 5 * num_episodes
        assert (
            sum([done for obs, action, reward, done, next_obs in steps]) == num_episodes
        )


def test_labels():
    task_variant = EpisodicTaskVariant(
        DummyEnv,
        num_episodes=1,
        rng_seed=0,
    )
    assert task_variant.task_label == "DummyEnv"
    assert task_variant.variant_label == "Default"

    task_variant = EpisodicTaskVariant(
        DummyEnv,
        num_episodes=1,
        task_label="TaskLabel",
        rng_seed=0,
    )
    assert task_variant.task_label == "TaskLabel"
    assert task_variant.variant_label == "Default"

    task_variant = EpisodicTaskVariant(
        DummyEnv,
        num_episodes=1,
        variant_label="VariantLabel",
        rng_seed=0,
    )
    assert task_variant.task_label == "DummyEnv"
    assert task_variant.variant_label == "VariantLabel"

    task_variant = EpisodicTaskVariant(
        DummyEnv,
        num_episodes=1,
        task_label="TaskLabel",
        variant_label="VariantLabel",
        rng_seed=0,
    )
    assert task_variant.task_label == "TaskLabel"
    assert task_variant.variant_label == "VariantLabel"


def test_validate():
    pass


def test_terminal_observations():
    task_variant = EpisodicTaskVariant(
        DummyEnv,
        num_episodes=1,
        task_label="TaskLabel",
        variant_label="VariantLabel",
        params={
            "observations": [0, 1, 2, 3, 4, 5],
            "max_steps": 3,
            "a": 1,
            "b": 3.0,
            "c": "a",
        },
        rng_seed=0,
    )
    transitions = list(task_variant.generate(random_action))
    assert len(transitions) == 3
    assert transitions[0][0] == 0
    assert transitions[0][-1] == 1
    assert transitions[1][0] == 1
    assert transitions[1][-1] == 2
    assert transitions[2][0] == 2
    assert transitions[2][-1] == 3
