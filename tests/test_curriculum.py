import itertools
import typing
from gym.envs.classic_control import CartPoleEnv
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

    expected_summary = (
        "This curriculum has 3 blocks"
        "\n\n\tBlock 1, learning: 1 task"
        "\n\t\tTask 1, CartPoleEnv: 2 variants"
        "\n\t\t\tTask variant 1, CartPoleEnv - Variant1: 1 episode."
        "\n\t\t\tTask variant 2, CartPoleEnv - Variant2: 1 episode."
        "\n\n\tBlock 2, learning: 1 task"
        "\n\t\tTask 1, CartPoleEnv: 2 variants"
        "\n\t\t\tTask variant 1, CartPoleEnv - Variant1: 1 episode."
        "\n\t\t\tTask variant 2, CartPoleEnv - Variant2: 1 episode."
        "\n\n\tBlock 3, learning: 1 task"
        "\n\t\tTask 1, CartPoleEnv: 2 variants"
        "\n\t\t\tTask variant 1, CartPoleEnv - Variant1: 1 episode."
        "\n\t\t\tTask variant 2, CartPoleEnv - Variant2: 1 episode."
    )

    assert(summarize_curriculum(curriculum) == expected_summary)
