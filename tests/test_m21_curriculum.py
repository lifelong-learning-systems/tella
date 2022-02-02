from collections import Counter
import glob
import gym.error
from unittest import mock
import pytest

# m21 curriculum depends on gym_minigrid so skip tests if not available
pytest.importorskip("gym_minigrid")

from tella.curriculum import validate_curriculum
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


CONFIG_STEP_LIMIT = """
# This is a mocked YAML file to be loaded as a test config
---
learn:
    default unit: steps
"""


CONFIG_PER_TASK = """
# This is a mocked YAML file to be loaded as a test config
---
learn:
    default length: 999
    CustomFetchS16T2N4: 1234
    SimpleCrossing: 42
num learn blocks: 5
"""


CONFIG_FORMAT_ERROR = """
# This is a mocked YAML file to be loaded as a test config
---
learn:
    default length:
        - This is not the expected format
"""


CONFIG_VALUE_ERROR = """
# This is a mocked YAML file to be loaded as a test config
---
learn:
    SimpleCrossing: This is not an integer
"""


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
    mock.mock_open(read_data=CONFIG_STEP_LIMIT),
)
def test_curriculum_file_configuration_step_limit():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    task_info = [
        (
            block.is_learning_allowed,
            variant.task_label,
            variant.variant_label,
            variant.num_episodes,
            variant.num_steps,
        )
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    expected_eval_episodes = 100
    expected_learn_steps = 1000
    num_learning_steps = Counter()
    for (
        is_learning_allowed,
        task_label,
        variant_label,
        num_episodes,
        num_steps,
    ) in task_info:
        if not is_learning_allowed:
            assert num_episodes == expected_eval_episodes
            assert num_steps is None
        else:
            assert num_episodes is None
            num_learning_steps[(task_label, variant_label)] += num_steps

    assert all(
        num_steps == expected_learn_steps for num_steps in num_learning_steps.values()
    )


@mock.patch(
    "builtins.open",
    mock.mock_open(read_data=CONFIG_PER_TASK),
)
def test_curriculum_file_configuration_per_task():
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


def test_default_block_limits():
    curriculum = MiniGridDispersed(rng_seed=0)
    default_block_limit = {
        f"num_{curriculum.DEFAULT_BLOCK_LENGTH_UNIT}": curriculum.DEFAULT_LEARN_BLOCK_LENGTH
    }
    assert curriculum._block_limit_from_config("", "") == default_block_limit


@mock.patch(
    "builtins.open",
    mock.mock_open(read_data=CONFIG_STEP_LIMIT),
)
def test_configured_block_limits_step_limit():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    expected_block_limit = {"num_steps": curriculum.DEFAULT_LEARN_BLOCK_LENGTH}
    assert curriculum._block_limit_from_config("", "") == expected_block_limit


@mock.patch(
    "builtins.open",
    mock.mock_open(read_data=CONFIG_PER_TASK),
)
def test_configured_block_limits_per_task():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    default_block_limit = {f"num_{curriculum.DEFAULT_BLOCK_LENGTH_UNIT}": 999}

    assert curriculum._block_limit_from_config("", "") == default_block_limit

    assert curriculum._block_limit_from_config("CustomFetch", "S16T2N4") == {
        "num_episodes": 1234
    }

    assert curriculum._block_limit_from_config("SimpleCrossing", "") == {
        "num_episodes": 42
    }


@mock.patch(
    "builtins.open",
    mock.mock_open(read_data=CONFIG_FORMAT_ERROR),
)
def test_configured_block_limits_format_error():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(AssertionError):
        curriculum._block_limit_from_config("", "")


@mock.patch(
    "builtins.open",
    mock.mock_open(read_data=CONFIG_VALUE_ERROR),
)
def test_configured_block_limits_value_error():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(ValueError):
        curriculum._block_limit_from_config("SimpleCrossing", "")


def test_find_example_configs():
    example_configs = glob.glob("examples/configs/*.yml")
    expected_number = 3  # Update if configs added or removed
    assert len(example_configs) == expected_number


@pytest.mark.parametrize("config_file", glob.glob("examples/configs/*.yml"))
def test_example_configurations(config_file: str):
    curriculum = MiniGridDispersed(rng_seed=0, config_file=config_file)
    validate_curriculum(curriculum)
