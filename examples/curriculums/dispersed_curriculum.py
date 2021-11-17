import typing
from copy import deepcopy
from tella.curriculum import Curriculum, TaskBlock
from random_env import *


class ExampleDispersed(Curriculum):
    observation_space = RandomEnv.observation_space
    action_space = RandomEnv.action_space

    def __init__(self, num_subgroups: int, seed: int):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.num_subgroups = num_subgroups

    def learning_blocks(self) -> typing.Iterable[typing.Iterable[TaskBlock]]:
        tasks = [
            TaskBlock(Task1VariantA, num_episodes=10),
            TaskBlock(Task2, num_episodes=10),
            TaskBlock(Task3Variant1, num_episodes=10),
            TaskBlock(Task4, num_episodes=10),
            TaskBlock(Task1VariantB, num_episodes=10, params={"a": 0.1}),
            TaskBlock(Task2, num_episodes=10, params={"b": 0.2}),
            TaskBlock(Task3Variant2, num_episodes=10, params={"c": 0.3}),
            TaskBlock(Task4, num_episodes=10, params={"d": 0.4}),
        ]
        for i in range(self.num_subgroups):
            self.rng.shuffle(tasks)
            for task_block in tasks:
                yield [task_block]

    def eval_block(self) -> typing.Iterable[TaskBlock]:
        tasks = [
            TaskBlock(Task1VariantA, num_episodes=1),
            TaskBlock(Task2, num_episodes=1),
            TaskBlock(Task3Variant1, num_episodes=1),
            TaskBlock(Task4, num_episodes=1),
            TaskBlock(Task1VariantB, num_episodes=1, params={"a": 0.1}),
            TaskBlock(Task2, num_episodes=1, params={"b": 0.2}),
            TaskBlock(Task3Variant2, num_episodes=1, params={"c": 0.3}),
            TaskBlock(Task4, num_episodes=1, params={"d": 0.4}),
        ]
        for task_block in tasks:
            yield task_block


if __name__ == "__main__":
    curriculum = ExampleDispersed(num_subgroups=2, seed=0)
    for i, task_blocks in enumerate(curriculum.learning_blocks()):
        for task_block in task_blocks:
            print(i, "learning", task_block)
        for task_block in curriculum.eval_block():
            print(i, "evaluating", task_block)
