from typing import Dict
from _backbone.nn.inception import (
    InceptionModel,
    InceptionConfig,
    Classifier,
    Discriminator,
)
from _classifiers.hf.cdan.base import CDANModelBase
from _datasets.base import TLDataset
from _utils.enumerations import *
from _classifiers.hf.cdan.base import HFCDAN


"""
CDAN algorithm using Inception model for time series as backbone.
"""


class InceptionCDANmodel(CDANModelBase):

    config_class = InceptionConfig
    config: InceptionConfig

    def __init__(self, config: InceptionConfig):
        in_dim_disc = (
            config.randomized_dim
            if config.randomized is True
            else config.out_dim * config.num_labels
        )
        super().__init__(
            config,
            backbone=InceptionModel(config),
            domain_discriminator=Discriminator(in_dim_disc, 1),
            classifier=Classifier(config.out_dim, config.num_labels),
            features_dim=config.out_dim,
        )


class InceptionCDAN(HFCDAN):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config,
            pretrained_mdl_cls=InceptionCDANmodel,
            pretrained_cfg_cls=InceptionConfig,
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
