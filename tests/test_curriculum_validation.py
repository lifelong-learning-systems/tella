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
        for block in self.blocks:
            yield block


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
    with pytest.raises(AssertionError):
        invalid_task_block = TaskBlock(
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


def test_validate_params():
    def function_with_params(a: int, b: float, c: str):
        pass

    class ClassWithParams:
        def __init__(self, a: int, b: float, c: str) -> None:
            pass

    for fn in [function_with_params, ClassWithParams]:
        # all parameters there, no extra ones
        validate_params(fn, ["a", "b", "c"])
        fn(**{"a": 1, "b": 2, "c": 3})

        # extra param that isn't present in function
        with pytest.raises(ValueError, match="Parameters not accepted: \['d'\]"):
            validate_params(fn, ["a", "b", "c", "d"])
        with pytest.raises(TypeError):
            fn(**{"a": 1, "b": 2, "c": 3, "d": 4})

        # missing params
        with pytest.raises(ValueError, match="Missing parameters: \['c'\]"):
            validate_params(fn, ["a", "b"])
        with pytest.raises(ValueError, match="Missing parameters: \['b', 'c'\]"):
            validate_params(fn, ["a"])
        with pytest.raises(ValueError, match="Missing parameters: \['a', 'b', 'c'\]"):
            validate_params(fn, [])
        with pytest.raises(TypeError):
            fn(**{"a": 1, "b": 2})


def test_validate_args():
    def function_with_params(q, *args, a=2, b=3, c=4):
        print(args, a, b, c)

    class ClassWithParams:
        def __init__(self, q, *args, a=2, b=3, c=4) -> None:
            pass

    for fn in [function_with_params, ClassWithParams]:
        with pytest.raises(ValueError, match="\*args not allowed"):
            validate_params(fn, ["args"])


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


def test_validate_default():
    def function_with_params(d=10):
        pass

    class ClassWithParams:
        def __init__(self, d=10) -> None:
            pass

    for fn in [function_with_params, ClassWithParams]:
        validate_params(fn, [])
        validate_params(fn, ["d"])


def test_validate_missing_not_default():
    def function_with_params(a, d=10):
        pass

    class ClassWithParams:
        def __init__(self, a, d=10) -> None:
            pass

    for fn in [function_with_params, ClassWithParams]:
        with pytest.raises(ValueError, match="Missing parameters: \['a'\]"):
            validate_params(fn, [])


def test_validate_kwargs():
    def function_with_params(a, b, c, d=10, **kwargs):
        print(kwargs)

    class ClassWithParams:
        def __init__(self, a, b, c, d=10, **kwargs) -> None:
            pass

    for fn in [function_with_params, ClassWithParams]:
        validate_params(fn, ["a", "b", "c", "d"])
        validate_params(fn, ["a", "b", "c", "test_kw"])
        validate_params(fn, ["a", "b", "c"])
