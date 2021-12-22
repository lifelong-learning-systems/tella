import itertools
import sys
import pytest
import typing
import gym
from tella.curriculum import (
    AbstractCurriculum,
    AbstractLearnBlock,
    AbstractEvalBlock,
    AbstractRLTaskVariant,
    EpisodicTaskVariant,
    simple_learn_block,
    simple_eval_block,
    validate_curriculum,
    TaskBlock,
    LearnBlock,
    validate_params,
)


class TestCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def __init__(
        self,
        blocks: typing.Iterable[
            typing.Union[
                "AbstractLearnBlock[AbstractRLTaskVariant]",
                "AbstractEvalBlock[AbstractRLTaskVariant]",
            ]
        ],
    ) -> None:
        self.blocks = blocks

    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        self.blocks, blocks = itertools.tee(self.blocks, 2)
        return blocks


def test_correct_curriculum():
    curriculum = TestCurriculum(
        [
            simple_learn_block(
                [
                    EpisodicTaskVariant(
                        lambda: gym.make("CartPole-v1"),
                        num_episodes=1,
                        variant_label="Variant1",
                    ),
                    EpisodicTaskVariant(
                        lambda: gym.make("CartPole-v1"),
                        num_episodes=1,
                        variant_label="Variant2",
                    ),
                ]
            ),
            simple_eval_block(
                [EpisodicTaskVariant(lambda: gym.make("CartPole-v1"), num_episodes=1)]
            ),
        ]
    )
    validate_curriculum(curriculum)


def test_simple_block_task_split():
    curriculum = TestCurriculum(
        [
            simple_learn_block(
                [
                    EpisodicTaskVariant(
                        lambda: gym.make("CartPole-v1"),
                        num_episodes=1,
                        task_label="Task1",
                    ),
                    EpisodicTaskVariant(
                        lambda: gym.make("CartPole-v1"),
                        num_episodes=1,
                        task_label="Task2",
                    ),
                ]
            ),
            simple_eval_block(
                [EpisodicTaskVariant(lambda: gym.make("CartPole-v1"), num_episodes=1)]
            ),
        ]
    )
    validate_curriculum(curriculum)


def test_error_on_diff_task_labels():
    curriculum = TestCurriculum(
        [
            LearnBlock(
                [
                    TaskBlock(
                        "Task1",
                        [
                            EpisodicTaskVariant(
                                lambda: gym.make("CartPole-v1"),
                                num_episodes=1,
                                task_label="Task1",
                            ),
                            EpisodicTaskVariant(
                                lambda: gym.make("CartPole-v1"),
                                num_episodes=1,
                                task_label="Task2",
                            ),
                        ],
                    )
                ]
            ),
            simple_eval_block(
                [EpisodicTaskVariant(lambda: gym.make("CartPole-v1"), num_episodes=1)]
            ),
        ]
    )
    with pytest.raises(ValueError) as err:
        validate_curriculum(curriculum)

    assert err.match(
        "Block #0, task block #0 had more than 1 task label "
        "found across all task variants: {'Task2', 'Task1'}"
    )


def test_warn_same_variant_labels():
    curriculum = TestCurriculum(
        [
            simple_learn_block(
                [
                    EpisodicTaskVariant(
                        lambda: gym.make("CartPole-v1"),
                        num_episodes=1,
                        variant_label="Variant1",
                    ),
                    EpisodicTaskVariant(
                        lambda: gym.make("CartPole-v1"),
                        num_episodes=1,
                        variant_label="Variant1",
                    ),
                ]
            ),
            simple_eval_block(
                [EpisodicTaskVariant(lambda: gym.make("CartPole-v1"), num_episodes=1)]
            ),
        ]
    )
    with pytest.warns(UserWarning):
        validate_curriculum(curriculum)


def test_generator_curriculum():
    curriculum = TestCurriculum(
        simple_learn_block(
            EpisodicTaskVariant(
                lambda: gym.make("CartPole-v1"),
                num_episodes=1,
                variant_label=variant_name,
            )
            for variant_name in ("Variant1", "Variant2")
        )
        for _ in range(3)
    )
    validate_curriculum(curriculum)
    # Validate twice to check if generators were exhausted
    validate_curriculum(curriculum)


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

    with pytest.raises(ValueError) as err:
        validate_params(example_function, ["a", "b"])

    assert err.match("Parameters not accepted: \['b'\]")


def test_validate_invalid_params_class():
    class ExampleClass:
        def __init__(self, a: int):
            pass

    with pytest.raises(ValueError) as err:
        validate_params(ExampleClass, ["a", "b"])

    assert err.match("Parameters not accepted: \['b'\]")


def test_validate_missing_params_function():
    def example_function(a, b):
        pass

    with pytest.raises(ValueError) as err:
        validate_params(example_function, [])

    assert err.match("Missing parameters: \['a', 'b'\]")

    with pytest.raises(ValueError) as err:
        validate_params(example_function, ["a"])

    assert err.match("Missing parameters: \['b'\]")


def test_validate_missing_params_class():
    class ExampleClass:
        def __init__(self, a, b):
            pass

    with pytest.raises(ValueError) as err:
        validate_params(ExampleClass, [])

    assert err.match("Missing parameters: \['a', 'b'\]")

    with pytest.raises(ValueError) as err:
        validate_params(ExampleClass, ["a"])

    assert err.match("Missing parameters: \['b'\]")


def test_validate_args_function():
    def example_function(*args):
        pass

    with pytest.raises(ValueError) as err:
        validate_params(example_function, ["args"])

    assert err.match("\*args not allowed")


def test_validate_args_class():
    class ExampleClass:
        def __init__(self, *args):
            pass

    with pytest.raises(ValueError) as err:
        validate_params(ExampleClass, ["args"])

    assert err.match("\*args not allowed")


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

    with pytest.raises(ValueError) as err:
        validate_params(example_function, [])

    assert err.match("Missing parameters: \['a'\]")


def test_validate_missing_not_default_class():
    class ExampleClass:
        def __init__(self, a, d=10):
            pass

    with pytest.raises(ValueError) as err:
        validate_params(ExampleClass, [])

    assert err.match("Missing parameters: \['a'\]")


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
            ValueError, match="Positional only arguments not allowed. Found q"
        ):
            validate_params(fn, ["q", "c"])
"""
