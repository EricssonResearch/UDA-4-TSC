import torch
import torch.nn as nn
from transformers import PreTrainedModel
from typing import Dict, Type, TypeVar
from _classifiers.hf.common import CommonTrainer
from _classifiers.hf.base import HuggingFaceClassifier
from _datasets.base import TLDataset
from _utils.enumerations import *
from _backbone.nn.cotmix import ConditionalEntropyLoss, SupConLoss, NTXentLoss, MixConfig


THFCoTMixBase = TypeVar("THFCoTMixBase", bound="HFCoTMixBase")


class CoTMixBasemodel(PreTrainedModel):

    config: MixConfig

    def __init__(self, config: MixConfig, backbone: PreTrainedModel, classifier: nn.Module):
        """Init for the base cotmix model, config passed to mother class"""
        super().__init__(config)

        self.backbone = backbone
        self.classifier = classifier

        self.cross_entropy_loss = nn.CrossEntropyLoss()  # Loss for the original classification task
        self.entropy = (
            ConditionalEntropyLoss()
        )  # Entropy of the prediction (soft probabilities) on target domain samples
        # The following loss functions measure the similarity between predictions on natural and mixture (source dominant or target dominant) samples
        self.supervised_contrastive_loss = SupConLoss(
            self.config.temperature
        )  # Compares the predictions on source domain and source dominant samples, as a function of their class labels
        self.contrastive_loss = NTXentLoss(
            self.config.temperature, self.config.use_cosine_similarity
        )  # Compares the predictions on target domain and target dominant samples, independently of their class labels
        self.mix_ratio: float = round(
            self.config.mix_ratio, 2
        )  # percentage of the natural sample in the mixture (source or target dominant)
        self.h = self.config.temporal_shift // 2  # half  # size of the window for temporal mixture

    def forward_train(self, mts, labels, domain_y):
        # Forward pass for training

        # Source and target indices
        src_indxs = domain_y == DomainEnumInt.source.value
        trg_indxs = domain_y == DomainEnumInt.target.value

        # Separate source samples from target samples
        src_x = mts[src_indxs]
        trg_x = mts[trg_indxs]
        src_y = labels[src_indxs]

        # Forward natural source and target samples
        mts_ft = self.backbone(mts)[0]  # natural source and target features
        mts_lg = self.classifier(mts_ft)  # natural source and target logits
        total_loss = 0
        if any(src_indxs):
            src_lg = mts_lg[src_indxs]  # natural source logits
            src_loss = self.cross_entropy_loss(src_lg, src_y)  # source cross-entropy loss
            # Compute total loss
            total_loss = total_loss + self.config.beta1 * src_loss
        if any(trg_indxs):
            trg_lg = mts_lg[trg_indxs]  # natural target logits
            trg_loss = self.entropy(trg_lg)  # target entropy loss
            # Compute total loss
            total_loss = total_loss + self.config.beta2 * trg_loss

        if src_x.shape[0] == trg_x.shape[0]:
            # Compute temporal mixtures
            # Source dominant mixture
            src_d_x = self.mix_ratio * src_x + (1 - self.mix_ratio) * torch.mean(
                torch.stack([torch.roll(trg_x, -i, 2) for i in range(-self.h, self.h)], 2), 2
            )
            src_d_x = src_d_x.to(self.device)
            # Target dominant mixture
            trg_d_x = self.mix_ratio * trg_x + (1 - self.mix_ratio) * torch.mean(
                torch.stack([torch.roll(src_x, -i, 2) for i in range(-self.h, self.h)], 2), 2
            )
            trg_d_x = trg_d_x.to(self.device)

            # Forward source dominant features
            src_d_ft = self.backbone(src_d_x)[0]  # source dominant features
            src_d_lg = self.classifier(src_d_ft)  # source dominant logits
            src_concat = torch.stack([src_lg, src_d_lg], dim=1)
            src_supcon_loss = self.supervised_contrastive_loss(
                src_concat, src_y
            )  # supervised contrastive loss

            # Forward target dominant features
            trg_d_ft = self.backbone(trg_d_x)[0]  # target dominant features
            trg_d_lg = self.classifier(trg_d_ft)  # target dominant logits
            trg_con_loss = self.contrastive_loss(trg_lg, trg_d_lg)  # contrastive loss

            # Compute total loss
            total_loss = (
                total_loss + self.config.beta3 * src_supcon_loss + self.config.beta4 * trg_con_loss
            )
        return (total_loss, mts_lg)

    def forward_eval(self, mts, labels):
        # Forward pass at inference time
        mts_ft = self.backbone(mts)[0]  # natural source and target features
        mts_lg = self.classifier(mts_ft)  # natural source and target logits
        total_loss = self.cross_entropy_loss(
            mts_lg, labels
        )  # cross-entropy loss, target samples will not contribute
        return (total_loss, mts_lg)

    def forward(self, mts, labels, domain_y=None):
        if domain_y is not None:
            return self.forward_train(mts, labels, domain_y)
        else:
            return self.forward_eval(mts, labels)


class HFCoTMixBase(HuggingFaceClassifier):
    default_lr: float = 0.001

    def __init__(
        self,
        config: Dict,
        pretrained_mdl_cls: Type[CoTMixBasemodel],
        pretrained_cfg_cls: Type[MixConfig],
        **kwargs,
    ):
        super().__init__(
            config=config,
            pretrained_mdl_cls=pretrained_mdl_cls,
            pretrained_cfg_cls=pretrained_cfg_cls,
            **kwargs,
        )

        # Optimizer used in CoTMix
        optimizer = torch.optim.AdamW(
            [p for p in self.classifier.parameters() if p.requires_grad],
            lr=self.config.training.get("learning_rate", self.default_lr),
            weight_decay=self.config.training.get("weight_decay"),
        )
        self.addition_training_args["optimizers"] = (optimizer, None)

        self.trainer_class = CommonTrainer

    def fit(self, tl_dataset: TLDataset) -> THFCoTMixBase:
        new_tl_dataset = self.get_new_tl_dataset(tl_dataset)
        return super().fit(new_tl_dataset)
