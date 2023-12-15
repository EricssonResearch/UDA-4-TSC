import torch
import numpy as np
from _utils.enumerations import *
from transformers import Trainer
from torch.utils.data import Sampler
from typing import Iterator, List


class EqualDomainTrainSampler(Sampler[int]):
    r"""Samples elements randomly from a dataset, alternating between elements from
    the source and target domain. Will take twice the size of the smallest domain.

    Args:
        domain_labels (list of integers): a sequence of domain labels, indicating whether the sample
        at that index belongs to the source or target domain.
    """

    def __init__(self, domain_labels: List[int]):

        domain_labels = np.asarray(domain_labels)

        target_indxs = np.nonzero(domain_labels == DomainEnumInt.target.value)[0]
        source_indxs = np.nonzero(domain_labels == DomainEnumInt.source.value)[0]

        n_target_samples, n_source_samples = len(target_indxs), len(source_indxs)
        self.sampler_len = min(n_source_samples, n_target_samples) * 2

        self.target_indxs = target_indxs.tolist()
        self.source_indxs = source_indxs.tolist()

    def __iter__(self) -> Iterator[int]:
        # shuffle indexes before each epoch
        self.target_indxs = np.random.permutation(self.target_indxs).tolist()
        self.source_indxs = np.random.permutation(self.source_indxs).tolist()
        for i in range(self.sampler_len):
            if (i % 2) == 0:
                yield self.target_indxs[i // 2]
            else:
                yield self.source_indxs[i // 2]

    def __len__(self) -> int:
        return self.sampler_len


class CommonTrainer(Trainer):
    """A common trainer for HF datasets"""

    def _get_train_sampler(self) -> Sampler:
        """Will override the default train sampler"""
        domain_y = [example[AdditionalColumnEnum.domain_y] for example in self.train_dataset]
        train_sampler = EqualDomainTrainSampler(domain_y)
        return train_sampler
