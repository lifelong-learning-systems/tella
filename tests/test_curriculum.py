import itertools
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
    summarize_curriculum,
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
        rng_seed: typing.Optional[int] = None,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        self.blocks, blocks = itertools.tee(self.blocks, 2)
        return blocks


def test_simple_block_task_split():
    curriculum = TestCurriculum(
        [
            simple_learn_block(
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
                ]
            ),
            simple_eval_block([EpisodicTaskVariant(CartPoleEnv, num_episodes=1)]),
        ]
    )
    validate_curriculum(curriculum)


def test_generator_curriculum():
    curriculum = TestCurriculum(
        simple_learn_block(
            EpisodicTaskVariant(
                CartPoleEnv,
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


def test_curriculum_summary():
    curriculum = TestCurriculum(
        [
            simple_learn_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                    ),
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant",
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

    expected_summary = (
        "This curriculum has 2 blocks"
        "\n\n\tBlock 1, learning: 2 tasks"
        "\n\t\tTask 1, CartPoleEnv: 2 variants"
        "\n\t\t\tTask variant 1, CartPoleEnv - Default: 1 episode."
        "\n\t\t\tTask variant 2, CartPoleEnv - Variant: 1 episode."
        "\n\t\tTask 2, MountainCarEnv: 1 variant"
        "\n\t\t\tTask variant 1, MountainCarEnv - Default: 1 episode."
        "\n\n\tBlock 2, evaluation: 1 task"
        "\n\t\tTask 1, CartPoleEnv: 1 variant"
        "\n\t\t\tTask variant 1, CartPoleEnv - Default: 1 episode."
    )

    assert summarize_curriculum(curriculum) == expected_summary
