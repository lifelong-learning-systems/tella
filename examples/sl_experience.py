import typing
from tella.curriculum import Experience
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset


class TorchVisionExperience(
    Experience[int, typing.Tuple[torch.Tensor, torch.LongTensor]]
):
    def __init__(self, dataset: VisionDataset) -> None:
        self.dataset = dataset

    def validate(self) -> None:
        pass

    def generate(
        self, batch_size: int
    ) -> typing.Iterable[typing.Tuple[torch.Tensor, torch.LongTensor]]:
        dataloader = DataLoader(self.dataset, batch_size)
        for batch in dataloader:
            yield batch
