from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class WeightedResumableEpochSampler(ResumableEpochSampler):
    """ResumableEpochSampler that draws len(dataset) indices with replacement
    from per-index `weights` instead of a permutation.

    Seeded identically on every rank (like the parent), so accelerate's
    sampler sharding still gives each rank a disjoint slice of one list.
    """

    def __init__(self, dataset: Sized, weights: torch.Tensor, seed: int, batch_size: int, num_processes: int):
        super().__init__(dataset, seed=seed, batch_size=batch_size, num_processes=num_processes)
        if len(weights) != len(dataset):
            raise ValueError(f"weights has {len(weights)} entries, dataset has {len(dataset)}")
        self.weights = torch.as_tensor(weights, dtype=torch.float64)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.multinomial(self.weights, len(self.dataset), replacement=True, generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)
