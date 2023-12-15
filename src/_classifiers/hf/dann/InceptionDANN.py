from typing import Dict
from _backbone.nn.inception import (
    InceptionModel,
    InceptionConfig,
    Classifier,
    Discriminator,
)
from _classifiers.hf.dann.base import DANNmodel, HFDANN
from _datasets.base import TLDataset
from _utils.enumerations import *


"""
DANN algorithm using Inception model for time series as backbone.
"""


class InceptionDANNmodel(DANNmodel):

    config_class = InceptionConfig
    config: InceptionConfig

    def __init__(self, config: InceptionConfig):
        super().__init__(
            config,
            backbone=InceptionModel(config),
            discriminator=Discriminator(config.out_dim, config.n_source_dom),
            classifier=Classifier(config.out_dim, config.num_labels),
        )


class InceptionDANN(HFDANN):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config,
            pretrained_mdl_cls=InceptionDANNmodel,
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
