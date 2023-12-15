import torch
import torch.nn as nn
import torch.nn.functional as F
from _utils.enumerations import *
from typing import Type, Dict, TypeVar
from transformers import PreTrainedModel
from _backbone.nn.cdan import *
from _utils.enumerations import *
from _classifiers.hf.dann.base import HFDANN
from _backbone.nn.GradReverse import GradReverse


class CDANModelBase(PreTrainedModel):
    """
    CDAN: Conditional Domain Adversarial Network

    @article{long2018conditional,
    title={Conditional adversarial domain adaptation},
    author={Long, Mingsheng and Cao, Zhangjie and Wang, Jianmin and Jordan, Michael I},
    journal={Advances in neural information processing systems},
    volume={31},
    year={2018}
    }

    Inspired by: https://github.com/thuml/Transfer-Learning-Library/blob/master/tllib/alignment/cdan.py

    This is a base class so cannot be used directly since it needs to have backbone ...
    """

    config: CDANConfig

    def __init__(
        self,
        config: CDANConfig,
        domain_discriminator: nn.Module,
        classifier: nn.Module,
        backbone: PreTrainedModel,
        features_dim: int,
    ):
        """
        Given the config of CDAN and a backbone appply CDAN loss.
        Args:
           config (CDANConfig): the pretrainedconfig of cdan base models
           domain_discriminator (nn.Module): the domain discriminator network
           classifier (nn.Module): the source label classifier
           backbone (PreTrainedModel): the backbone of the model
           features_dim (int): the dimension of the features (e.g. equal to the out dim for the backbone)
        """
        super().__init__(config)
        self.domain_discriminator = domain_discriminator
        self.classifier = classifier
        self.backbone = backbone

        if self.config.randomized:
            assert (
                self.config.num_labels > 0 and features_dim > 0 and self.config.randomized_dim > 0
            )
            self.map = RandomizedMultiLinearMap(
                features_dim, self.config.num_labels, self.config.randomized_dim
            )
        else:
            self.map = MultiLinearMap()

        self.classifier_loss = nn.CrossEntropyLoss()

        self.tradeoff = None  # Initially set at 0 and updated during training

    def grad_reverse(self, x: torch.Tensor) -> torch.Tensor:
        return GradReverse.apply(x, self.tradeoff)

    def forward(self, mts, labels, domain_y=None) -> torch.Tensor:
        """
        - g (tensor): unnormalized classifier predictions, :math:`g^s`
        - f (tensor): feature representations, :math:`f^s`
        """
        f = self.backbone(mts)[0]
        g = self.classifier(f)
        cls_loss = self.classifier_loss(g, labels)

        dom_loss = 0.0

        if domain_y is not None:
            # Domain labels are only used during training. If evaluating the model, then skip
            g = F.softmax(g, dim=1).detach()
            h = self.grad_reverse(self.map(f, g))
            d = self.domain_discriminator(h)

            weight = 1.0 + torch.exp(-entropy(g))
            batch_size = f.size(0)
            weight: torch.Tensor = weight / torch.sum(weight) * batch_size

            # create labels to same shape as inputs
            d_label = torch.zeros_like(d)
            d_label[domain_y == DomainEnumInt.target.value, DomainEnumInt.target.value] = 1
            d_label[domain_y == DomainEnumInt.source.value, DomainEnumInt.source.value] = 1

            # get weights in case needed
            weight = weight.reshape(-1, 1) if self.config.entropy_conditioning is True else None
            # compute loss
            dom_loss = F.binary_cross_entropy_with_logits(input=d, target=d_label, weight=weight)

        total_loss = dom_loss + cls_loss
        return (total_loss, g)


THFCDAN = TypeVar("THFCDAN", bound="HFCDAN")


class HFCDAN(HFDANN):
    def __init__(
        self,
        config: Dict,
        pretrained_mdl_cls: Type[CDANModelBase],
        pretrained_cfg_cls: Type[CDANConfig],
        **kwargs,
    ):
        super().__init__(
            config=config,
            pretrained_mdl_cls=pretrained_mdl_cls,
            pretrained_cfg_cls=pretrained_cfg_cls,
            **kwargs,
        )
