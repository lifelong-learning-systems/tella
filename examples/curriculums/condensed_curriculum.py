import typing
from tella.curriculum import (
    InterleavedEvalCurriculum,
    EvalBlock,
    LearnBlock,
)
import gym
from tella.rl_experience import RLEpisodeExperience, ActionFn, Transition
from random_env import *


class ExampleCondensed(InterleavedEvalCurriculum[ActionFn, Transition, gym.Env]):
    def __init__(self, seed: int):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def learn_blocks(self) -> typing.Iterable[LearnBlock]:
        task_experiences = [
            RLEpisodeExperience(Task1VariantA, num_episodes=10),
            RLEpisodeExperience(Task2, num_episodes=10),
            RLEpisodeExperience(Task3Variant1, num_episodes=10),
            RLEpisodeExperience(Task4, num_episodes=10),
            RLEpisodeExperience(Task1VariantB, num_episodes=10, params={"a": 0.1}),
            RLEpisodeExperience(Task2, num_episodes=10, params={"b": 0.2}),
            RLEpisodeExperience(Task3Variant2, num_episodes=10, params={"c": 0.3}),
            RLEpisodeExperience(Task4, num_episodes=10, params={"d": 0.4}),
        ]
        self.rng.shuffle(task_experiences)
        for experience in task_experiences:
            # NOTE: only 1 regime in each learn block
            yield LearnBlock([experience])

    def eval_block(self) -> EvalBlock:
        return EvalBlock(
            [
                RLEpisodeExperience(Task1VariantA, num_episodes=1),
                RLEpisodeExperience(Task2, num_episodes=1),
                RLEpisodeExperience(Task3Variant1, num_episodes=1),
                RLEpisodeExperience(Task4, num_episodes=1),
                RLEpisodeExperience(Task1VariantB, num_episodes=1, params={"a": 0.1}),
                RLEpisodeExperience(Task2, num_episodes=1, params={"b": 0.2}),
                RLEpisodeExperience(Task3Variant2, num_episodes=1, params={"c": 0.3}),
                RLEpisodeExperience(Task4, num_episodes=1, params={"d": 0.4}),
            ]
        )


if __name__ == "__main__":
    curriculum = ExampleCondensed(seed=0)
    for i, block in enumerate(curriculum.blocks()):
        for experience in block.experiences():
            print(
                f"Block {i}, learning_allowed={block.is_learning_allowed()}, experience={experience}, info={experience.info()}"
            )
