import typing
from tella.curriculum import AbstractLearnBlock, AbstractEvalBlock
from tella.curriculum import (
    InterleavedEvalCurriculum,
    simple_learn_block,
    simple_eval_block,
)
from tella.curriculum import AbstractRLTaskVariant, EpisodicTaskVariant
from random_env import *


class ExampleDispersed(InterleavedEvalCurriculum[AbstractRLTaskVariant]):
    def __init__(self, num_repetitions: int):
        super().__init__()
        self.num_repetitions = num_repetitions

    def learn_blocks(
        self,
        rng_seed: int,
    ) -> typing.Iterable[AbstractLearnBlock[AbstractRLTaskVariant]]:
        task_variants = [
            EpisodicTaskVariant(Task1VariantA, num_episodes=10),
            EpisodicTaskVariant(Task2, num_episodes=10),
            EpisodicTaskVariant(Task3Variant1, num_episodes=10),
            EpisodicTaskVariant(Task4, num_episodes=10),
            EpisodicTaskVariant(Task1VariantB, num_episodes=10, params={"a": 0.1}),
            EpisodicTaskVariant(Task2, num_episodes=10, params={"b": 0.2}),
            EpisodicTaskVariant(Task3Variant2, num_episodes=10, params={"c": 0.3}),
            EpisodicTaskVariant(Task4, num_episodes=10, params={"d": 0.4}),
        ]
        rng = np.random.default_rng(rng_seed)
        for i_repetition in range(self.num_repetitions):
            rng.shuffle(task_variants)
            for task_variant in task_variants:
                # NOTE: only 1 experience in the learn block
                yield simple_learn_block([task_variant])

    def eval_block(
        self,
        rng_seed: int,
    ) -> AbstractEvalBlock[AbstractRLTaskVariant]:
        return simple_eval_block(
            [
                EpisodicTaskVariant(Task1VariantA, num_episodes=1),
                EpisodicTaskVariant(Task2, num_episodes=1),
                EpisodicTaskVariant(Task3Variant1, num_episodes=1),
                EpisodicTaskVariant(Task4, num_episodes=1),
                EpisodicTaskVariant(Task1VariantB, num_episodes=1, params={"a": 0.1}),
                EpisodicTaskVariant(Task2, num_episodes=1, params={"b": 0.2}),
                EpisodicTaskVariant(Task3Variant2, num_episodes=1, params={"c": 0.3}),
                EpisodicTaskVariant(Task4, num_episodes=1, params={"d": 0.4}),
            ]
        )


if __name__ == "__main__":
    curriculum = ExampleDispersed(num_repetitions=2)
    for i, block in enumerate(curriculum.learn_blocks_and_eval_blocks(rng_seed=0)):
        for task_block in block.task_blocks():
            for task_variant in task_block.task_variants():
                print(
                    f"Block {i}, learning_allowed={block.is_learning_allowed}, "
                    f"task_variant={task_variant}, info={task_variant.info()}"
                )
