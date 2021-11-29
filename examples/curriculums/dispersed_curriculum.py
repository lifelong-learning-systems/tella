import typing
from tella.curriculum import (
    InterleavedEvalCurriculum,
    EvalBlock,
    LearnBlock,
)
from tella.experiences.rl import LimitedEpisodesExperience, RLExperience
from random_env import *


class ExampleDispersed(InterleavedEvalCurriculum[RLExperience]):
    def __init__(self, num_repetitions: int, seed: int):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.num_repetitions = num_repetitions

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        experiences = [
            LimitedEpisodesExperience(Task1VariantA, num_episodes=10),
            LimitedEpisodesExperience(Task2, num_episodes=10),
            LimitedEpisodesExperience(Task3Variant1, num_episodes=10),
            LimitedEpisodesExperience(Task4, num_episodes=10),
            LimitedEpisodesExperience(
                Task1VariantB, num_episodes=10, params={"a": 0.1}
            ),
            LimitedEpisodesExperience(Task2, num_episodes=10, params={"b": 0.2}),
            LimitedEpisodesExperience(
                Task3Variant2, num_episodes=10, params={"c": 0.3}
            ),
            LimitedEpisodesExperience(Task4, num_episodes=10, params={"d": 0.4}),
        ]
        for i_repetition in range(self.num_repetitions):
            self.rng.shuffle(experiences)
            for task_experience in experiences:
                # NOTE: only 1 experience in the learn block
                yield LearnBlock([task_experience])

    def eval_block(self) -> EvalBlock:
        return EvalBlock(
            [
                LimitedEpisodesExperience(Task1VariantA, num_episodes=1),
                LimitedEpisodesExperience(Task2, num_episodes=1),
                LimitedEpisodesExperience(Task3Variant1, num_episodes=1),
                LimitedEpisodesExperience(Task4, num_episodes=1),
                LimitedEpisodesExperience(
                    Task1VariantB, num_episodes=1, params={"a": 0.1}
                ),
                LimitedEpisodesExperience(Task2, num_episodes=1, params={"b": 0.2}),
                LimitedEpisodesExperience(
                    Task3Variant2, num_episodes=1, params={"c": 0.3}
                ),
                LimitedEpisodesExperience(Task4, num_episodes=1, params={"d": 0.4}),
            ]
        )


if __name__ == "__main__":
    curriculum = ExampleDispersed(num_repetitions=2, seed=0)
    for i, block in enumerate(curriculum.blocks()):
        for task_experience in block.experiences():
            print(
                f"Block {i}, learning_allowed={block.is_learning_allowed()}, experience={task_experience}"
            )
