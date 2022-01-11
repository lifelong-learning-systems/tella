import itertools
import pytest
import typing
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv
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


class SampleCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def __init__(
        self,
        blocks: typing.Iterable[
            typing.Union[
                "AbstractLearnBlock[AbstractRLTaskVariant]",
                "AbstractEvalBlock[AbstractRLTaskVariant]",
            ]
        ],
    ) -> None:
        super().__init__(0)
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
    curriculum = SampleCurriculum(
        [
            simple_learn_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant1",
                    ),
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant2",
                    ),
                ]
            ),
            simple_eval_block([EpisodicTaskVariant(CartPoleEnv, num_episodes=1)]),
        ]
    )
    validate_curriculum(curriculum)


def test_error_on_diff_task_labels():
    curriculum = SampleCurriculum(
        [
            LearnBlock(
                [
                    TaskBlock(
                        "Task1",
                        [
                            EpisodicTaskVariant(
                                CartPoleEnv,
                                num_episodes=1,
                                task_label="Task1",
                            ),
                            EpisodicTaskVariant(
                                CartPoleEnv,
                                num_episodes=1,
                                task_label="Task2",
                            ),
                        ],
                    )
                ]
            ),
            simple_eval_block([EpisodicTaskVariant(CartPoleEnv, num_episodes=1)]),
        ]
    )
    with pytest.raises(ValueError) as err:
        validate_curriculum(curriculum)

    assert err.match(
        "Block #0, task block #0 had more than 1 task label found across all task variants: "
    )


def test_error_on_multiple_spaces():
    curriculum = SampleCurriculum(
        [
            simple_learn_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                    ),
                    EpisodicTaskVariant(
                        MountainCarEnv,
                        num_episodes=1,
                    ),
                ]
            ),
            simple_eval_block([EpisodicTaskVariant(CartPoleEnv, num_episodes=1)]),
        ]
    )

    with pytest.raises(ValueError) as err:
        validate_curriculum(curriculum)

    assert err.match(
        "All environments in a curriculum must use the same observation and action spaces."
    )


def test_warn_same_variant_labels():
    curriculum = SampleCurriculum(
        [
            simple_learn_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant1",
                    ),
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant1",
                    ),
                ]
            ),
            simple_eval_block([EpisodicTaskVariant(CartPoleEnv, num_episodes=1)]),
        ]
    )
    with pytest.warns(UserWarning):
        validate_curriculum(curriculum)


def test_empty_curriculum():
    curriculum = SampleCurriculum([])

    with pytest.raises(ValueError) as err:
        validate_curriculum(curriculum)

    assert err.match("This curriculum is empty.")


def test_empty_block():
    curriculum = SampleCurriculum([LearnBlock([])])

    with pytest.raises(ValueError) as err:
        validate_curriculum(curriculum)

    assert err.match("Block #0 is empty.")


def test_empty_task():
    curriculum = SampleCurriculum([LearnBlock([TaskBlock("Task1", [])])])

    with pytest.raises(ValueError) as err:
        validate_curriculum(curriculum)

    assert err.match("Block #0, task block #0 is empty.")


def test_invalid_task_params():
    curriculum = SampleCurriculum(
        [
            simple_eval_block(
                [EpisodicTaskVariant(CartPoleEnv, num_episodes=1, params={"a": 1})]
            ),
        ]
    )

    with pytest.raises(ValueError) as err:
        validate_curriculum(curriculum)

    assert err.match(
        "Invalid task variant at block #0, task block #0, task variant #0."
    )
    assert (
        err.getrepr().chain[0][1].message
        == "ValueError: Parameters not accepted: ['a'] in ()"
    )


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

    assert err.match(r"Parameters not accepted: \['b'\]")


def test_validate_invalid_params_class():
    class ExampleClass:
        def __init__(self, a: int):
            pass

    with pytest.raises(ValueError) as err:
        validate_params(ExampleClass, ["a", "b"])

    assert err.match(r"Parameters not accepted: \['b'\]")


def test_validate_missing_params_function():
    def example_function(a, b):
        pass

    with pytest.raises(ValueError) as err:
        validate_params(example_function, [])

    assert err.match(r"Missing parameters: \['a', 'b'\]")

    with pytest.raises(ValueError) as err:
        validate_params(example_function, ["a"])

    assert err.match(r"Missing parameters: \['b'\]")


def test_validate_missing_params_class():
    class ExampleClass:
        def __init__(self, a, b):
            pass

    with pytest.raises(ValueError) as err:
        validate_params(ExampleClass, [])

    assert err.match(r"Missing parameters: \['a', 'b'\]")

    with pytest.raises(ValueError) as err:
        validate_params(ExampleClass, ["a"])

    assert err.match(r"Missing parameters: \['b'\]")


def test_validate_args_function():
    def example_function(*args):
        pass

    with pytest.raises(ValueError) as err:
        validate_params(example_function, ["args"])

    assert err.match(r"\*args not allowed")


def test_validate_args_class():
    class ExampleClass:
        def __init__(self, *args):
            pass

    with pytest.raises(ValueError) as err:
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

    with pytest.raises(ValueError) as err:
        validate_params(example_function, [])

    assert err.match(r"Missing parameters: \['a'\]")


def test_validate_missing_not_default_class():
    class ExampleClass:
        def __init__(self, a, d=10):
            pass

    with pytest.raises(ValueError) as err:
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
            ValueError, match="Positional only arguments not allowed. Found q"
        ):
            validate_params(fn, ["q", "c"])
"""
