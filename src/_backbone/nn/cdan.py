import torch
import numpy as np
import torch.nn as nn
from typing import Optional
from transformers import PretrainedConfig


class CDANConfig(PretrainedConfig):

    entropy_conditioning: bool
    randomized: bool
    randomized_dim: int

    def __init__(self, entropy_conditioning=True, randomized=True, randomized_dim=1024, **kwargs):
        """
        Init of the config base class for CDAN based models

        Args:

            entropy_conditioning (bool, optional): If True, use entropy-aware weight to reweight each training example.
                Default: False
            randomized (bool, optional): If True, use `randomized multi linear map`. Else, use `multi linear map`.
                Default: False
            randomized_dim (int, optional): Dimension of features after randomized.
                Default: 1024

        """
        super().__init__(**kwargs)
        self.entropy_conditioning = entropy_conditioning
        self.randomized = randomized
        self.randomized_dim = randomized_dim


class RandomizedMultiLinearMap(nn.Module):
    """Random multi linear map
    Given two inputs :math:`f` and :math:`g`, the definition is
    .. math::
        T_{\odot}(f,g) = \dfrac{1}{\sqrt{d}} (R_f f) \odot (R_g g),
    where :math:`\odot` is element-wise product, :math:`R_f` and :math:`R_g` are random matrices
    sampled only once and ﬁxed in training.
    Args:
        features_dim (int): dimension of input :math:`f`
        num_classes (int): dimension of input :math:`g`
        output_dim (int, optional): dimension of output tensor. Default: 1024
    Shape:
        - f: (minibatch, features_dim)
        - g: (minibatch, num_classes)
        - Outputs: (minibatch, output_dim)
    """

    def __init__(self, features_dim: int, num_classes: int, output_dim: Optional[int] = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    """Multi linear map
    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        batch_size = f.size(0)
        output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        res = output.view(batch_size, -1)
        return res


def entropy(predictions: torch.Tensor, reduction="none") -> torch.Tensor:
    r"""Entropy of prediction.
    The definition is:
    .. math::
        entropy(p) = - \sum_{c=1}^C p_c \log p_c
    where C is number of classes.
    Args:
        predictions (tensor): Classifier predictions. Expected to contain raw, normalized scores for each class
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output. Default: ``'mean'``
    Shape:
        - predictions: :math:`(minibatch, C)` where C means the number of classes.
        - Output: :math:`(minibatch, )` by default. If :attr:`reduction` is ``'mean'``, then scalar.
    """
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    H = H.sum(dim=1)
    if reduction == "mean":
        return H.mean()
    else:
        return H
