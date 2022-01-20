import math
import pytest
import typing
import gym
from tella.curriculum import EpisodicTaskVariant, _where


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


@pytest.mark.parametrize("num_envs", [1, 3])
def test_num_episodes(num_envs: int):
    for num_episodes in [1, 3, 5]:
        exp = EpisodicTaskVariant(
            DummyEnv,
            num_episodes=num_episodes,
            params={"a": 1, "b": 3.0, "c": "a"},
            rng_seed=0,
        )
        exp.set_num_envs(num_envs)
        masked_transitions = sum(exp.generate(random_action), [])
        steps = [
            transition for transition in masked_transitions if transition is not None
        ]
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


@pytest.mark.parametrize("num_envs", [1, 2])
def test_generate_return_type(num_envs):
    task_variant = EpisodicTaskVariant(
        DummyEnv,
        num_episodes=3,
        params={"a": 1, "b": 3.0, "c": "a"},
        rng_seed=0,
    )
    task_variant.set_num_envs(num_envs)
    all_transitions = task_variant.generate(random_action)

    assert isinstance(all_transitions, typing.Generator)

    all_transitions = list(all_transitions)
    expected_num = math.ceil(3 / num_envs) * 5
    assert len(all_transitions) == expected_num

    for step_transitions in all_transitions:
        assert isinstance(step_transitions, typing.List)
        assert len(step_transitions) == num_envs

        for transition in step_transitions:
            if transition is not None:
                assert isinstance(transition, typing.Tuple)
                assert len(transition) == 5


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
    transitions = sum(task_variant.generate(random_action), [])
    assert len(transitions) == 3
    assert transitions[0][0] == 0
    assert transitions[0][-1] == 1
    assert transitions[1][0] == 1
    assert transitions[1][-1] == 2
    assert transitions[2][0] == 2
    assert transitions[2][-1] == 3


def test_show_rewards():
    task_variant = EpisodicTaskVariant(
        DummyEnv,
        num_episodes=3,
        params={"a": 1, "b": 3.0, "c": "a"},
        rng_seed=0,
    )
    task_variant.set_show_rewards(True)
    transitions = sum(task_variant.generate(random_action), [])
    assert len(transitions) > 0
    for obs, action, reward, done, next_obs in transitions:
        assert reward is not None


def test_hide_rewards():
    task_variant = EpisodicTaskVariant(
        DummyEnv,
        num_episodes=3,
        params={"a": 1, "b": 3.0, "c": "a"},
        rng_seed=0,
    )
    task_variant.set_show_rewards(False)
    transitions = sum(task_variant.generate(random_action), [])
    assert len(transitions) > 0
    for obs, action, reward, done, next_obs in transitions:
        assert reward is None


def test_where():
    original = [1, 1, 1, 1, 1]

    mask = [False, False, False, False, False]
    expected = [1, 1, 1, 1, 1]
    assert _where(mask, 2, original) == expected

    mask = [True, True, True, True, True]
    expected = [2, 2, 2, 2, 2]
    assert _where(mask, 2, original) == expected

    mask = [False, False, True, False, True]
    expected = [1, 1, 2, 1, 2]
    assert _where(mask, 2, original) == expected

    mask = [False, True, False, True, True]
    expected = [1, None, 1, None, None]
    assert _where(mask, None, original) == expected


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("num_episodes", [1, 3, 5])
def test_vec_env_mask(num_envs: int, num_episodes: int):
    task_variant = EpisodicTaskVariant(
        DummyEnv,
        num_episodes=num_episodes,
        params={"a": 1, "b": 3.0, "c": "a"},
        rng_seed=0,
    )
    task_variant.set_num_envs(num_envs)
    transitions = list(task_variant.generate(random_action))
    masked = [[transition is None for transition in batch] for batch in transitions]

    expected = []
    eps_remaining = num_episodes
    while eps_remaining:
        if eps_remaining < num_envs:
            batch_mask = [False] * eps_remaining + [True] * (num_envs - eps_remaining)
            eps_remaining = 0
        else:
            batch_mask = [False] * num_envs
            eps_remaining -= num_envs
        expected.extend([batch_mask] * 5)  # 5 steps per episode

    assert masked == expected
