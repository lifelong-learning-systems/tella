import itertools
import typing
import numpy as np
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv
from tella.curriculum import (
    AbstractCurriculum,
    InterleavedEvalCurriculum,
    AbstractLearnBlock,
    AbstractEvalBlock,
    AbstractRLTaskVariant,
    EpisodicTaskVariant,
    TaskVariantType,
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


class ShuffledCurriculum(AbstractCurriculum[AbstractRLTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
        rng_seed: typing.Optional[int] = None,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        rng = np.random.default_rng(rng_seed)
        for n in rng.permutation(100):
            yield simple_learn_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv, num_episodes=1, task_label=f"Task{n}"
                    )
                ]
            )
        yield simple_eval_block(
            [EpisodicTaskVariant(CartPoleEnv, num_episodes=1, task_label="Task0")]
        )


def test_curriculum_no_rng_seed():
    curriculum = ShuffledCurriculum()

    first_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    second_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    assert first_call_tasks != second_call_tasks
    # In theory these could randomly result in the same order, failing the test. The probability
    #   of this is made negligible by giving ShuffledCurriculum 100 tasks (p = 1 / 100! = 1e-158).


def test_curriculum_rng_seed():
    curriculum = ShuffledCurriculum()

    first_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks(rng_seed=0)
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    second_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks(rng_seed=0)
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    assert first_call_tasks == second_call_tasks


class ShuffledInterleavedCurriculum(InterleavedEvalCurriculum[AbstractRLTaskVariant]):
    def learn_blocks(
        self,
        rng_seed: typing.Optional[int] = None,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        rng = np.random.default_rng(rng_seed)
        for n in rng.permutation(10):
            yield simple_learn_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv, num_episodes=1, task_label=f"Task{n}"
                    )
                ]
            )

    def eval_block(
        self,
        rng_seed: typing.Optional[int] = None,
    ) -> AbstractEvalBlock[TaskVariantType]:
        rng = np.random.default_rng(rng_seed)
        return simple_eval_block(
            [
                EpisodicTaskVariant(CartPoleEnv, num_episodes=1, task_label=f"Task{n}")
                for n in rng.permutation(10)
            ]
        )


def test_interleaved_rng_seed():
    curriculum = ShuffledInterleavedCurriculum()

    first_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks(rng_seed=0)
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    second_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks(rng_seed=0)
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    assert first_call_tasks == second_call_tasks
