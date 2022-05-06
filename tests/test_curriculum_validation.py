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

import itertools
import pytest
import sys
import typing
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv
from tella.curriculum import (
    AbstractCurriculum,
    Block,
    TaskVariant,
    simple_learn_block,
    simple_eval_block,
    validate_curriculum,
    TaskBlock,
    LearnBlock,
    validate_params,
    ValidationError,
)


class SampleCurriculum(AbstractCurriculum):
    def __init__(self, blocks: typing.Iterable[Block]) -> None:
        super().__init__(0)
        self.blocks = blocks

    def learn_blocks_and_eval_blocks(self) -> typing.Iterable[Block]:
        self.blocks, blocks = itertools.tee(self.blocks, 2)
        return blocks


class InvalidCurriculum(SampleCurriculum):
    def validate(self) -> None:
        raise ValidationError("This curriculum tests validation errors.")


def test_correct_curriculum():
    curriculum = SampleCurriculum(
        [
            simple_learn_block(
                [
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant1",
                        rng_seed=0,
                    ),
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant2",
                        rng_seed=0,
                    ),
                ]
            ),
            simple_eval_block(
                [
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        rng_seed=0,
                    )
                ]
            ),
        ]
    )
    validate_curriculum(curriculum)


def test_custom_invalid_curriculum():
    curriculum = InvalidCurriculum(
        [
            simple_learn_block(
                [
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant1",
                        rng_seed=0,
                    ),
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant2",
                        rng_seed=0,
                    ),
                ]
            ),
            simple_eval_block(
                [
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        rng_seed=0,
                    )
                ]
            ),
        ]
    )
    with pytest.raises(ValidationError):
        curriculum.validate()


def test_error_on_diff_task_labels():
    # tests if the variants for a task have the same task label
    curriculum = SampleCurriculum(
        [
            LearnBlock(
                [
                    TaskBlock(
                        "Task1",
                        [
                            TaskVariant(
                                CartPoleEnv,
                                num_episodes=1,
                                task_label="Task1",
                                variant_label="1",
                                rng_seed=0,
                            ),
                            TaskVariant(
                                CartPoleEnv,
                                num_episodes=1,
                                task_label="Task2",
                                variant_label="2",
                                rng_seed=0,
                            ),
                        ],
                    )
                ]
            ),
            simple_eval_block(
                [
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        rng_seed=0,
                    )
                ]
            ),
        ]
    )
    with pytest.raises(ValidationError) as err:
        validate_curriculum(curriculum)

    assert err.match(
        "Block #0, task block #0 had more than 1 task label found across all task variants: "
    )


def test_error_on_multiple_spaces():
    curriculum = SampleCurriculum(
        [
            simple_learn_block(
                [
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        rng_seed=0,
                    ),
                    TaskVariant(
                        MountainCarEnv,
                        num_episodes=1,
                        rng_seed=0,
                    ),
                ]
            ),
            simple_eval_block(
                [
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        rng_seed=0,
                    )
                ]
            ),
        ]
    )

    with pytest.raises(ValidationError) as err:
        validate_curriculum(curriculum)

    assert err.match(
        "All environments in a curriculum must use the same observation and action spaces."
    )


def test_warn_same_variant_labels():
    curriculum = SampleCurriculum(
        [
            simple_learn_block(
                [
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant1",
                        rng_seed=0,
                    ),
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant1",
                        rng_seed=0,
                    ),
                ]
            ),
            simple_eval_block(
                [
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        rng_seed=0,
                    )
                ]
            ),
        ]
    )
    with pytest.warns(UserWarning):
        validate_curriculum(curriculum)


def test_empty_curriculum():
    curriculum = SampleCurriculum([])

    with pytest.raises(ValidationError) as err:
        validate_curriculum(curriculum)

    assert err.match("This curriculum is empty.")


def test_empty_block():
    curriculum = SampleCurriculum([LearnBlock([])])

    with pytest.raises(ValidationError) as err:
        validate_curriculum(curriculum)

    assert err.match("Block #0 is empty.")


def test_empty_task():
    curriculum = SampleCurriculum([LearnBlock([TaskBlock("Task1", [])])])

    with pytest.raises(ValidationError) as err:
        validate_curriculum(curriculum)

    assert err.match("Block #0, task block #0 is empty.")


def test_invalid_task_params():
    if sys.version_info < (3, 9):
        pytest.skip("Python before 3.9 had a bug with inspect.")
    curriculum = SampleCurriculum(
        [
            simple_eval_block(
                [
                    TaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        params={"a": 1},
                        rng_seed=0,
                    )
                ]
            ),
        ]
    )

    with pytest.raises(ValidationError) as err:
        validate_curriculum(curriculum)

    assert err.match(
        "Invalid task variant at block #0, task block #0, task variant #0."
    )
    assert "Parameters not accepted: ['a'] in ()" in err.getrepr().chain[0][1].message


def test_validate_valid_params_function():
    def example_function(a: int, b: float, c: str):
        pass

    validate_params(example_function, ["a", "b", "c"])


def test_validate_valid_params_class():
    class ExampleClass:
        def __init__(self, a: int, b: float, c: str):
            pass

    validate_params(ExampleClass, ["a", "b", "c"])


def test_validate_invalid_params_function():
    def example_function(a: int):
        pass

    with pytest.raises(ValidationError) as err:
        validate_params(example_function, ["a", "b"])

    assert err.match(r"Parameters not accepted: \['b'\]")


def test_validate_invalid_params_class():
    class ExampleClass:
        def __init__(self, a: int):
            pass

    with pytest.raises(ValidationError) as err:
        validate_params(ExampleClass, ["a", "b"])

    assert err.match(r"Parameters not accepted: \['b'\]")


def test_validate_missing_params_function():
    def example_function(a, b):
        pass

    with pytest.raises(ValidationError) as err:
        validate_params(example_function, [])

    assert err.match(r"Missing parameters: \['a', 'b'\]")

    with pytest.raises(ValidationError) as err:
        validate_params(example_function, ["a"])

    assert err.match(r"Missing parameters: \['b'\]")


def test_validate_missing_params_class():
    class ExampleClass:
        def __init__(self, a, b):
            pass

    with pytest.raises(ValidationError) as err:
        validate_params(ExampleClass, [])

    assert err.match(r"Missing parameters: \['a', 'b'\]")

    with pytest.raises(ValidationError) as err:
        validate_params(ExampleClass, ["a"])

    assert err.match(r"Missing parameters: \['b'\]")


def test_validate_args_function():
    if sys.version_info < (3, 9):
        pytest.skip("Python before 3.9 had a bug with inspect.")

    def example_function(*args):
        pass

    with pytest.raises(ValidationError) as err:
        validate_params(example_function, ["args"])

    assert err.match(r"\*args not allowed")


def test_validate_args_class():
    if sys.version_info < (3, 9):
        pytest.skip("Python before 3.9 had a bug with inspect.")

    class ExampleClass:
        def __init__(self, *args):
            pass

    with pytest.raises(ValidationError) as err:
        validate_params(ExampleClass, ["args"])

    assert err.match(r"\*args not allowed")


def test_validate_default_function():
    def example_function(d=10):
        pass

    validate_params(example_function, [])
    validate_params(example_function, ["d"])


def test_validate_default_class():
    class ExampleClass:
        def __init__(self, d=10):
            pass

    validate_params(ExampleClass, [])
    validate_params(ExampleClass, ["d"])


def test_validate_missing_not_default_function():
    def example_function(a, d=10):
        pass

    with pytest.raises(ValidationError) as err:
        validate_params(example_function, [])

    assert err.match(r"Missing parameters: \['a'\]")


def test_validate_missing_not_default_class():
    class ExampleClass:
        def __init__(self, a, d=10):
            pass

    with pytest.raises(ValidationError) as err:
        validate_params(ExampleClass, [])

    assert err.match(r"Missing parameters: \['a'\]")


def test_validate_kwargs_function():
    def example_function(a, **kwargs):
        pass

    validate_params(example_function, ["a", "b", "c", "test_kw"])


def test_validate_kwargs_class():
    class ExampleClass:
        def __init__(self, a, **kwargs):
            pass

    validate_params(ExampleClass, ["a", "b", "c", "test_kw"])


"""
NOTE: python 3.7 fails to parse this file if this is uncommented. Leaving here for later

@pytest.mark.skipif(sys.version_info < (3, 8), reason="Position arguments require 3.8+")
def test_validate_positional():
    def function_with_params(q, /, c):
        pass

    class ClassWithParams:
        def __init__(self, q, /, c) -> None:
            pass

    for fn in [function_with_params, ClassWithParams]:
        with pytest.raises(
            ValidationError, match="Positional only arguments not allowed. Found q"
        ):
            validate_params(fn, ["q", "c"])
"""
