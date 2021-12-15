import pytest
import typing
import gym
from tella.curriculum import AbstractCurriculum, AbstractLearnBlock, AbstractEvalBlock
from tella.curriculum.rl_task_variant import AbstractRLTaskVariant, EpisodicTaskVariant
from tella.curriculum.builders import simple_learn_block, simple_eval_block
from tella.validation import validate_curriculum


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


def test_error_on_diff_task_labels():
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
    with pytest.raises(ValueError):
        validate_curriculum(curriculum)


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
