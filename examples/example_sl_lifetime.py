import typing

from tella.agents.continual_learning_agent import ContinualLearningAgent, Metrics
from tella.experiences.experience import Experience
from tella.curriculum import Block, Curriculum, LearnBlock, EvalBlock
from tella.run import run
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

Batch = typing.Tuple[torch.Tensor, torch.LongTensor]


class TorchVisionExperience(Experience[int, typing.Iterable[Batch], VisionDataset]):
    def __init__(self, dataset: VisionDataset) -> None:
        self.dataset = dataset

    def validate(self) -> None:
        pass

    def info(self) -> VisionDataset:
        return self.dataset

    def generate(self, batch_size: int) -> typing.Iterable[Batch]:
        dataloader = DataLoader(self.dataset, batch_size)
        for batch in dataloader:
            yield batch


class ContinualSupervisedLearningAgent(ContinualLearningAgent[TorchVisionExperience]):
    def consume_experience(self, experience: TorchVisionExperience) -> Metrics:
        metrics = {"Accuracy": 0.5}
        for batch in experience.generate(batch_size=10):
            x, label = batch
            ...
        return metrics


class ExampleCurriculum(Curriculum[TorchVisionExperience]):
    def blocks(self) -> typing.Iterable[Block]:
        dataset = TensorDataset(
            torch.rand(10, 64, 64, 3), torch.randint(low=0, high=5, size=(10,))
        )
        yield LearnBlock([TorchVisionExperience(dataset)])
        yield EvalBlock([TorchVisionExperience(dataset)])


if __name__ == "__main__":
    import logging
    from rl_logging_agent import LoggingAgent

    logging.basicConfig(level=logging.INFO)

    agent = ContinualSupervisedLearningAgent()
    curriculum = ExampleCurriculum()

    run(agent, curriculum)
