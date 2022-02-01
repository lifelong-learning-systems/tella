from collections import Counter
import gym.error
from unittest import mock
import pytest

# m21 curriculum depends on gym_minigrid so skip tests if not available
pytest.importorskip("gym_minigrid")

from tella._curriculums.minigrid.m21 import (
    MiniGridReducedActionSpaceWrapper,
    SimpleCrossingEnv,
    MiniGridDispersed,
)


class TestMiniGridReducedActionSpaceWrapper:
    env = SimpleCrossingEnv()

    def test_constructor(self):
        wrapper = MiniGridReducedActionSpaceWrapper(self.env, 3)
        assert wrapper.action_space.n == 3

    def test_too_many_actions(self):
        with pytest.raises(AssertionError):
            MiniGridReducedActionSpaceWrapper(self.env, 10)

    def test_with_continuous_action_space(self):
        # TODO
        pass

    def test_valid_action(self):
        wrapper = MiniGridReducedActionSpaceWrapper(self.env, 3)
        assert 2 == wrapper.action(2)

    def test_invalid_action(self):
        wrapper = MiniGridReducedActionSpaceWrapper(self.env, 3)
        with pytest.raises(gym.error.InvalidAction):
            wrapper.action(3)


def test_curriculum_default_configuration():
    curriculum = MiniGridDispersed(rng_seed=0)
    task_info = [
        (
            block.is_learning_allowed,
            variant.task_label,
            variant.variant_label,
            variant.num_episodes,
        )
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    expected_eval_episodes = 100
    expected_learn_episodes = 1000
    num_learning_episodes = Counter()
    for is_learning_allowed, task_label, variant_label, num_episodes in task_info:
        if not is_learning_allowed:
            assert num_episodes == expected_eval_episodes
        else:
            num_learning_episodes[(task_label, variant_label)] += num_episodes

    assert all(
        num_episodes == expected_learn_episodes
        for num_episodes in num_learning_episodes.values()
    )


@mock.patch(
    "builtins.open",
    mock.mock_open(
        read_data=(
            "# This is a fake yaml file to be loaded as a test config\n"
            "---\n"
            "learn:\n"
            "    default length: 999\n"
            "    CustomFetchS16T2N4: 1234\n"
            "    SimpleCrossing: 42\n"
            "num learn blocks: 5\n"
        )
    ),
)
def test_curriculum_file_configuration():
    curriculum = MiniGridDispersed(
        rng_seed=0, config_file="mocked.yml"
    )  # Filename doesn't matter here
    task_info = [
        (
            block.is_learning_allowed,
            variant.task_label,
            variant.variant_label,
            variant.num_episodes,
        )
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    expected_eval_episodes = 100
    num_learning_episodes = Counter()
    for is_learning_allowed, task_label, variant_label, num_episodes in task_info:
        if not is_learning_allowed:
            assert num_episodes == expected_eval_episodes
        else:
            num_learning_episodes[(task_label, variant_label)] += num_episodes

    for (task_label, variant_label), num_episodes in num_learning_episodes.items():
        if task_label == "SimpleCrossing":
            assert num_episodes == 42
        elif task_label + variant_label == "CustomFetchS16T2N4":
            assert num_episodes == 1234
        else:
            assert num_episodes == 999
