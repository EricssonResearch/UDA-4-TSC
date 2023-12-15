from _classifiers.hf.dann.base import DANNmodel, HFDANN
from _backbone.nn.VRADA import VRADAConfig, VRNN, Classifier_backbone, Discriminator
from typing import Dict
from torch import nn
from _datasets.base import TLDataset
from _utils.enumerations import *


"""
VRADA model as described in https://openreview.net/pdf?id=rk9eAFcxg
"""


class VRADAhf(DANNmodel):

    config_class = VRADAConfig
    config: VRADAConfig

    def __init__(self, config: VRADAConfig):
        classifier = nn.Sequential(
            Classifier_backbone(config.Classifier_h_dim, config.VRNN_z_dim),
            nn.Linear(config.Classifier_h_dim, config.num_labels),
        )
        discriminator = Discriminator(
            config.Discrimiator_h_dim, config.VRNN_z_dim, config.num_source_domains
        )
        super().__init__(
            config=config, classifier=classifier, backbone=VRNN(config), discriminator=discriminator
        )

    def forward(self, mts, labels, domain_y=None):
        outputs = self.backbone(mts)
        kld_loss, nll_loss, z_T = outputs[0], outputs[1], outputs[2]
        class_pred = self.classifier(z_T)
        class_loss = self.classifier_loss(class_pred, labels)
        if (
            domain_y is not None
        ):  # Domain labels are only used during training. If evaluating the model, then skip
            dom_pred = self.discriminator(
                self.grad_reverse(z_T)
            )  # Gradient reversal layer applied for adversarial training
            dom_loss = self.discriminator_loss(dom_pred, domain_y)
            total_loss = (
                class_loss
                + self.config.R_loss_w * nll_loss
                + self.config.R_loss_w * kld_loss
                + dom_loss
            )
        else:
            total_loss = (
                class_loss + self.config.R_loss_w * nll_loss + self.config.R_loss_w * kld_loss
            )
        return (total_loss, class_pred)


class VRADA(HFDANN):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config=config,
            pretrained_mdl_cls=VRADAhf,
            pretrained_cfg_cls=VRADAConfig,
            **kwargs,
        )

    def inject_config_based_on_dataset(self, config: Dict, tl_dataset: TLDataset) -> Dict:
        # check if attr exist
        if InjectConfigEnum.backbone not in config:
            config[InjectConfigEnum.backbone] = {}
        config[InjectConfigEnum.backbone][InjectConfigEnum.n_input_channels] = len(
            tl_dataset.source[DataSplitEnum.train][DatasetColumnsEnum.mts][0]
        )
        return super().inject_config_based_on_dataset(config, tl_dataset)
