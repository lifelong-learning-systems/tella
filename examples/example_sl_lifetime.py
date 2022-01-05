import typing

from tella.agents import ContinualLearningAgent, Metrics
from tella.curriculum import *
from tella.curriculum import simple_learn_block, simple_eval_block
from tella.experiment import run
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

Batch = typing.Tuple[torch.Tensor, torch.LongTensor]


class TorchVisionTaskVariant(
    AbstractTaskVariant[int, typing.Iterable[Batch], VisionDataset]
):
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

    @property
    def task_label(self) -> str:
        return self.dataset.__class__.name

    @property
    def variant_label(self) -> str:
        return "Default"


class ContinualSupervisedLearningAgent(ContinualLearningAgent[TorchVisionTaskVariant]):
    def learn_task_variant(self, task_variant: TorchVisionTaskVariant) -> Metrics:
        metrics = {"Accuracy": 0.5}
        for batch in task_variant.generate(batch_size=10):
            x, label = batch
            ...
        return metrics


class ExampleCurriculum(AbstractCurriculum[TorchVisionTaskVariant]):
    def learn_blocks_and_eval_blocks(
        self,
        rng_seed: typing.Optional[int] = None,
    ) -> typing.Iterable[
        typing.Union[
            "AbstractLearnBlock[TaskVariantType]", "AbstractEvalBlock[TaskVariantType]"
        ]
    ]:
        torch.manual_seed(rng_seed)
        dataset = TensorDataset(
            torch.rand(10, 64, 64, 3), torch.randint(low=0, high=5, size=(10,))
        )
        yield simple_learn_block([TorchVisionTaskVariant(dataset)])
        yield simple_eval_block([TorchVisionTaskVariant(dataset)])


if __name__ == "__main__":
    import logging
    from rl_logging_agent import LoggingAgent

    logging.basicConfig(level=logging.INFO)

    agent = ContinualSupervisedLearningAgent()
    curriculum = ExampleCurriculum()

    run(agent, curriculum)
