import itertools
import typing
from unittest import mock
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv
from tella.curriculum import (
    AbstractCurriculum,
    InterleavedEvalCurriculum,
    AbstractLearnBlock,
    AbstractEvalBlock,
    AbstractRLTaskVariant,
    EpisodicTaskVariant,
    TaskVariantType,
    ValidationError,
    simple_learn_block,
    simple_eval_block,
    validate_curriculum,
    summarize_curriculum,
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


def test_simple_block_task_split():
    curriculum = SampleCurriculum(
        [
            simple_learn_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        task_label="Task1",
                        rng_seed=0,
                    ),
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        task_label="Task2",
                        rng_seed=0,
                    ),
                ]
            ),
            simple_eval_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        rng_seed=0,
                    )
                ]
            ),
        ]
    )
    validate_curriculum(curriculum)


def test_generator_curriculum():
    curriculum = SampleCurriculum(
        simple_learn_block(
            EpisodicTaskVariant(
                CartPoleEnv,
                num_episodes=1,
                variant_label=variant_name,
                rng_seed=0,
            )
            for variant_name in ("Variant1", "Variant2")
        )
        for _ in range(3)
    )
    validate_curriculum(curriculum)
    # Validate twice to check if generators were exhausted
    validate_curriculum(curriculum)


def test_curriculum_summary():
    curriculum = SampleCurriculum(
        [
            simple_learn_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        rng_seed=0,
                    ),
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        variant_label="Variant",
                        rng_seed=0,
                    ),
                    EpisodicTaskVariant(
                        MountainCarEnv,
                        num_episodes=1,
                        rng_seed=0,
                    ),
                ]
            ),
            simple_eval_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        rng_seed=0,
                    )
                ]
            ),
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
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        for n in self.rng.permutation(100):
            yield simple_learn_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        task_label=f"Task{n}",
                        rng_seed=self.rng.bit_generator.random_raw(),
                    )
                ]
            )
        yield simple_eval_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    task_label="Task0",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


def test_curriculum_diff_rng_seed():
    curriculum = ShuffledCurriculum(111111)
    first_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    curriculum = ShuffledCurriculum(222222)
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
    curriculum = ShuffledCurriculum(0)
    first_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    curriculum = ShuffledCurriculum(0)
    second_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    assert first_call_tasks == second_call_tasks


def test_curriculum_copy():
    curriculum = ShuffledCurriculum(0)

    first_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.copy().learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    second_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    assert first_call_tasks == second_call_tasks


def test_curriculum_copy_validate():
    curriculum = ShuffledCurriculum(0)
    first_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    curriculum = ShuffledCurriculum(0)
    # Validation iterates over blocks and so changes the curriculum RNG state.
    #   Copying the curriculum should not alter the state
    validate_curriculum(curriculum.copy())
    second_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    assert first_call_tasks == second_call_tasks


class ShuffledInterleavedCurriculum(InterleavedEvalCurriculum[AbstractRLTaskVariant]):
    def learn_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[AbstractRLTaskVariant]",
            "AbstractEvalBlock[AbstractRLTaskVariant]",
        ]
    ]:
        for n in self.rng.permutation(10):
            yield simple_learn_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=1,
                        task_label=f"Task{n}",
                        rng_seed=self.rng.bit_generator.random_raw(),
                    )
                ]
            )

    def eval_block(self) -> AbstractEvalBlock[TaskVariantType]:
        return simple_eval_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    task_label=f"Task{n}",
                    rng_seed=self.eval_rng_seed,
                )
                for n in range(10)
            ]
        )


def test_interleaved_rng_seed():
    curriculum = ShuffledInterleavedCurriculum(0)
    first_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    curriculum = ShuffledInterleavedCurriculum(0)
    second_call_tasks = [
        (variant.task_label, variant.variant_label)
        for block in curriculum.learn_blocks_and_eval_blocks()
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]

    assert first_call_tasks == second_call_tasks


class SampleInterleavedCurriculum(InterleavedEvalCurriculum):
    def learn_blocks(self) -> typing.Iterable[AbstractLearnBlock[TaskVariantType]]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    task_label="Task1",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    task_label="Task2",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    task_label="Task3",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )

    def eval_block(self) -> AbstractEvalBlock[TaskVariantType]:
        return simple_eval_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    task_label="Task1",
                    rng_seed=self.eval_rng_seed,
                ),
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    task_label="Task1",
                    rng_seed=self.eval_rng_seed + 1,
                ),
            ]
        )


def test_interleaved_structure():
    curriculum = SampleInterleavedCurriculum(0)
    blocks = list(curriculum.learn_blocks_and_eval_blocks())

    assert len(blocks) == 7
    assert isinstance(blocks[0], AbstractEvalBlock)
    for i in range(len(blocks)):
        if i % 2 == 0:
            assert isinstance(blocks[i], AbstractEvalBlock)
        else:
            assert isinstance(blocks[i], AbstractLearnBlock)
    assert isinstance(blocks[-1], AbstractEvalBlock)


class ConfigurableCurriculum(AbstractCurriculum[EpisodicTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[EpisodicTaskVariant]",
            "AbstractEvalBlock[EpisodicTaskVariant]",
        ]
    ]:
        num_blocks = self.config.get("num learn blocks", 1)
        num_episodes = self.config.get("num episodes", 1)
        for _ in range(num_blocks):
            yield simple_learn_block(
                [
                    EpisodicTaskVariant(
                        CartPoleEnv,
                        num_episodes=num_episodes,
                        rng_seed=self.rng.bit_generator.random_raw(),
                    )
                ]
            )
        yield simple_eval_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_episodes=1,
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


def test_curriculum_default_configuration():
    curriculum = ConfigurableCurriculum(rng_seed=0)
    task_info = [
        (
            n_block,
            block.is_learning_allowed,
            variant.task_label,
            variant.variant_label,
            variant.num_episodes,
        )
        for n_block, block in enumerate(curriculum.learn_blocks_and_eval_blocks())
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]
    expected_values = [
        (0, True, "CartPoleEnv", "Default", 1),
        (1, False, "CartPoleEnv", "Default", 1),
    ]
    assert task_info == expected_values


@mock.patch(
    "builtins.open",
    mock.mock_open(
        read_data=(
            "# This is a fake yaml file to be loaded as a test config\n"
            "---\n"
            "num learn blocks: 3\n"
            "num episodes: 10\n"
        )
    ),
)
def test_curriculum_file_configuration():
    curriculum = ConfigurableCurriculum(
        rng_seed=0, config_file="mocked.yml"
    )  # Filename doesn't matter here
    task_info = [
        (
            n_block,
            block.is_learning_allowed,
            variant.task_label,
            variant.variant_label,
            variant.num_episodes,
        )
        for n_block, block in enumerate(curriculum.learn_blocks_and_eval_blocks())
        for task in block.task_blocks()
        for variant in task.task_variants()
    ]
    expected_values = [
        (0, True, "CartPoleEnv", "Default", 10),
        (1, True, "CartPoleEnv", "Default", 10),
        (2, True, "CartPoleEnv", "Default", 10),
        (3, False, "CartPoleEnv", "Default", 1),
    ]
    assert task_info == expected_values


class SampleStepLimitCurriculum(SampleInterleavedCurriculum):
    def learn_blocks(self) -> typing.Iterable[AbstractLearnBlock[TaskVariantType]]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_steps=5000,
                    task_label="Task1",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_steps=5,
                    task_label="Task2",
                    rng_seed=self.rng.bit_generator.random_raw(),
                ),
            ]
        )
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_steps=5,
                    task_label="Task3",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


class BadDoubleLimitsCurriculum(SampleStepLimitCurriculum):
    def learn_blocks(self) -> typing.Iterable[AbstractLearnBlock[TaskVariantType]]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    num_steps=5,
                    num_episodes=5,
                    task_label="Task3",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


class BadNoLimitsCurriculum(SampleStepLimitCurriculum):
    def learn_blocks(self) -> typing.Iterable[AbstractLearnBlock[TaskVariantType]]:
        yield simple_learn_block(
            [
                EpisodicTaskVariant(
                    CartPoleEnv,
                    task_label="Task3",
                    rng_seed=self.rng.bit_generator.random_raw(),
                )
            ]
        )


def test_step_limit_curriculum():
    curriculum = SampleStepLimitCurriculum(0)
    blocks = list(curriculum.learn_blocks_and_eval_blocks())

    assert len(blocks) == 7
    assert isinstance(blocks[0], AbstractEvalBlock)
    for i in range(len(blocks)):
        if i % 2 == 0:
            assert isinstance(blocks[i], AbstractEvalBlock)
        else:
            assert isinstance(blocks[i], AbstractLearnBlock)
    assert isinstance(blocks[-1], AbstractEvalBlock)


def test_bad_limits_curriculum():
    err = 0
    try:
        curriculum = BadDoubleLimitsCurriculum(0)
        validate_curriculum(curriculum)
    except ValidationError:
        err = 1
    assert err == 1
    try:
        curriculum = BadNoLimitsCurriculum(0)
        validate_curriculum(curriculum)
    except ValidationError:
        err = 2
    assert err == 2
