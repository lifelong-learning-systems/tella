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

import math
import pytest
import typing
import gym
from gym.envs.classic_control import CartPoleEnv
from tella.curriculum import TaskVariant, Transition
from tella.experiment import generate_transitions, _where


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

    def seed(self, seed=None):
        self.observation_space.seed(seed)
        self.action_space.seed(seed)


def choose_action_zero(
    observations: typing.List[typing.Optional[int]],
) -> typing.List[typing.Optional[int]]:
    return [None if obs is None else 0 for obs in observations]


@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("num_episodes", [1, 3, 5])
def test_num_episodes(num_envs: int, num_episodes: int):
    exp = TaskVariant(
        DummyEnv,
        num_episodes=num_episodes,
        params={"a": 1, "b": 3.0, "c": "a"},
        rng_seed=0,
    )
    masked_transitions = sum(
        generate_transitions(exp, choose_action_zero, num_envs), []
    )
    steps = [transition for transition in masked_transitions if transition is not None]
    assert len(steps) == 5 * num_episodes
    assert sum([done for obs, action, reward, done, next_obs in steps]) == num_episodes


@pytest.mark.parametrize("num_envs", [1, 3, 8])
@pytest.mark.parametrize("num_steps", [5, 10, 100])
def test_num_steps(num_steps: int, num_envs: int):
    exp = TaskVariant(
        DummyEnv,
        num_steps=num_steps,
        params={"a": 1, "b": 3.0, "c": "a"},
        rng_seed=0,
    )
    masked_transitions = sum(
        generate_transitions(exp, choose_action_zero, num_envs), []
    )
    steps = [transition for transition in masked_transitions if transition is not None]
    assert len(steps) == num_steps
    # assert len(steps) == 5 * num_episodes
    # assert sum([done for obs, action, reward, done, next_obs in steps]) == num_episodes


def test_labels():
    task_variant = TaskVariant(
        DummyEnv,
        num_episodes=1,
        rng_seed=0,
    )
    assert task_variant.task_label == "DummyEnv"
    assert task_variant.variant_label == "Default"

    task_variant = TaskVariant(
        DummyEnv,
        num_episodes=1,
        task_label="TaskLabel",
        rng_seed=0,
    )
    assert task_variant.task_label == "TaskLabel"
    assert task_variant.variant_label == "Default"

    task_variant = TaskVariant(
        DummyEnv,
        num_episodes=1,
        variant_label="VariantLabel",
        rng_seed=0,
    )
    assert task_variant.task_label == "DummyEnv"
    assert task_variant.variant_label == "VariantLabel"

    task_variant = TaskVariant(
        DummyEnv,
        num_episodes=1,
        task_label="TaskLabel",
        variant_label="VariantLabel",
        rng_seed=0,
    )
    assert task_variant.task_label == "TaskLabel"
    assert task_variant.variant_label == "VariantLabel"


@pytest.mark.parametrize("num_envs", [1, 2])
def test_generate_return_type(num_envs):
    task_variant = TaskVariant(
        DummyEnv,
        num_episodes=3,
        params={"a": 1, "b": 3.0, "c": "a"},
        rng_seed=0,
    )
    all_transitions = generate_transitions(task_variant, choose_action_zero, num_envs)

    assert isinstance(all_transitions, typing.Generator)

    all_transitions = list(all_transitions)
    expected_num = math.ceil(3 / num_envs) * 5
    assert len(all_transitions) == expected_num

    for step_transitions in all_transitions:
        assert isinstance(step_transitions, typing.List)
        assert len(step_transitions) == num_envs

        for transition in step_transitions:
            if transition is not None:
                assert isinstance(transition, Transition)
                assert len(transition) == 5

                # check data access patterns: unpacking, indexing, and keys
                obs, action, reward, done, next_obs = transition
                assert obs is transition[0] is transition.observation
                assert action is transition[1] is transition.action
                assert reward is transition[2] is transition.reward
                assert done is transition[3] is transition.done
                assert next_obs is transition[4] is transition.next_observation


def test_terminal_observations():
    task_variant = TaskVariant(
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
    transitions = sum(
        generate_transitions(task_variant, choose_action_zero, num_envs=1), []
    )
    assert len(transitions) == 3
    assert transitions[0].observation == 0
    assert transitions[0].next_observation == 1
    assert transitions[1].observation == 1
    assert transitions[1].next_observation == 2
    assert transitions[2].observation == 2
    assert transitions[2].next_observation == 3


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


def test_single_env_mask():
    task_variant = TaskVariant(
        DummyEnv,
        num_episodes=5,
        params={"a": 1, "b": 3.0, "c": "a"},
        rng_seed=0,
    )
    transitions = sum(
        generate_transitions(task_variant, choose_action_zero, num_envs=1), []
    )
    assert not any(transition is None for transition in transitions)


@pytest.mark.parametrize("num_episodes", [1, 3, 5])
def test_vec_cartpole_env_mask(num_episodes: int):
    num_envs = 3
    task_variant = TaskVariant(
        CartPoleEnv,
        num_episodes=num_episodes,
        rng_seed=0,
    )
    transitions = list(generate_transitions(task_variant, choose_action_zero, num_envs))

    episode_id = [i for i in range(num_envs)]
    next_episode_id = num_envs
    for batch in transitions:
        for n, transition in enumerate(batch):
            if transition is not None:
                obs, action, reward, done, next_obs = transition
                if done:
                    episode_id[n] = next_episode_id
                    next_episode_id += 1
            else:
                assert episode_id[n] >= num_episodes


@pytest.mark.parametrize("num_episodes", [1, 3, 5])
def test_vec_dummy_env_mask(num_episodes: int):
    num_envs = 3
    task_rng_seed = 0
    episode_lengths = [4, 6, 7]

    class IndexedDummyEnv(DummyEnv):
        def seed(self, seed=None):
            super().seed(seed)
            # AsyncVectorEnv increments the rng seed for each env, so it can be
            #   used as an index to give each a unique, predictable max_steps
            index = seed - task_rng_seed
            self.max_steps = episode_lengths[index]

    task_variant = TaskVariant(
        IndexedDummyEnv,
        num_episodes=num_episodes,
        params={"a": 1, "b": 3.0, "c": "a"},
        rng_seed=task_rng_seed,
    )
    transitions = list(generate_transitions(task_variant, choose_action_zero, num_envs))
    masked = [[transition is None for transition in batch] for batch in transitions]

    expected = {
        1: [
            [False, True, True],
            [False, True, True],
            [False, True, True],
            [False, True, True],
        ],
        3: [
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [True, False, False],
            [True, True, False],
        ],
        5: [
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, False],
            [False, False, True],
            [True, False, True],
            [True, False, True],
            [True, False, True],
            [True, False, True],
        ],
    }

    assert masked == expected[num_episodes]


def unmasked_choose_action_zero(
    observations: typing.List[typing.Optional[int]],
) -> typing.List[typing.Optional[int]]:
    return [0 for _ in observations]


def test_ignore_unmasked_actions():
    def identical_task_variant():
        task_variant = TaskVariant(
            DummyEnv,
            num_episodes=5,
            params={"a": 1, "b": 3.0, "c": "a"},
            rng_seed=0,
        )
        return task_variant

    masked_actions_transitions = list(
        generate_transitions(identical_task_variant(), choose_action_zero, 3)
    )
    unmasked_actions_transitions = list(
        generate_transitions(identical_task_variant(), unmasked_choose_action_zero, 3)
    )

    assert masked_actions_transitions == unmasked_actions_transitions
