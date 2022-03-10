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

from collections import Counter
import glob
import gym.error
from unittest import mock
import pytest

# m21 curriculum depends on gym_minigrid so skip tests if not available
pytest.importorskip("gym_minigrid")

from tella.curriculum import validate_curriculum, ValidationError
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
    CustomFetchS10T2N4: 1234
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


CONFIG_TYPO_ERROR = """
# This is a mocked YAML file to be loaded as a test config
---
learn:
    DoorKnob: 1234
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
        elif task_label + variant_label == "CustomFetchS10T2N4":
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

    assert curriculum._block_limit_from_config("CustomFetch", "S10T2N4") == {
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
    with pytest.raises(ValidationError) as err:
        curriculum.validate()
    assert err.match("Task default length must be a positive integer")


@mock.patch(
    "builtins.open",
    mock.mock_open(read_data=CONFIG_VALUE_ERROR),
)
def test_configured_block_limits_value_error():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(ValidationError) as err:
        curriculum.validate()
    assert err.match("Task config must be either an integer or a dictionary")


@mock.patch(
    "builtins.open",
    mock.mock_open(read_data=CONFIG_TYPO_ERROR),
)
def test_configured_block_limits_typo_error():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(ValidationError) as err:
        curriculum.validate()
    assert err.match("Unexpected task config key")


EXAMPLE_CONFIGS = glob.glob("**/examples/configs/*.yml", recursive=True)


def test_find_example_configs():
    expected_number = 3  # Update if configs added or removed
    assert len(EXAMPLE_CONFIGS) == expected_number


@pytest.mark.parametrize("config_file", EXAMPLE_CONFIGS)
def test_example_configurations(config_file: str):
    curriculum = MiniGridDispersed(rng_seed=0, config_file=config_file)
    validate_curriculum(curriculum)


@mock.patch("builtins.open", mock.mock_open(read_data="-a\n-b\n"))
def test_config_file_not_dictionary():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(ValidationError) as err:
        curriculum.validate()
    assert err.match("Configuration must be a dictionary")


@mock.patch("builtins.open", mock.mock_open(read_data="asdf: {hello: 1}"))
def test_config_file_unknown_top_level_key():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(ValidationError) as err:
        curriculum.validate()
    assert err.match("Unexpected config key")


@mock.patch("builtins.open", mock.mock_open(read_data="num learn blocks: 0"))
def test_config_file_zero_learn_blocks():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(ValidationError) as err:
        curriculum.validate()
    assert err.match("Num learn blocks must be a positive integer")


@mock.patch("builtins.open", mock.mock_open(read_data="learn: {DistShift: 0}"))
def test_config_file_zero_task_length():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(ValidationError) as err:
        curriculum.validate()
    assert err.match("Task length must be positive")


@mock.patch("builtins.open", mock.mock_open(read_data="learn: [0, 1]"))
def test_config_file_learn_not_dictionary():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(ValidationError) as err:
        curriculum.validate()
    assert err.match("Learn blocks config must be a dictionary")


@mock.patch(
    "builtins.open",
    mock.mock_open(read_data="learn: {default unit: blah}"),
)
def test_config_file_learn_invalid_unit():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(ValidationError) as err:
        curriculum.validate()
    assert err.match("Task default steps must be episodes or steps")


@mock.patch(
    "builtins.open", mock.mock_open(read_data="learn: {DistShift: {length: blah}}")
)
def test_config_file_invalid_task_length():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(ValidationError) as err:
        curriculum.validate()
    assert err.match("Task length must be a positive integer")


@mock.patch(
    "builtins.open", mock.mock_open(read_data="learn: {DistShift: {unit: blah}}")
)
def test_config_file_invalid_task_unit():
    curriculum = MiniGridDispersed(rng_seed=0, config_file="mocked.yml")
    with pytest.raises(ValidationError) as err:
        curriculum.validate()
    assert err.match("Task unit must be episodes or steps")
