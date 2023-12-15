from typing import Dict
from _backbone.nn.inception import (
    InceptionModel,
    InceptionConfig,
    Classifier,
)

from _classifiers.hf.cotmix.base import CoTMixBasemodel, HFCoTMixBase
from _datasets.base import TLDataset
from _utils.enumerations import *


class InceptionMixmodel(CoTMixBasemodel):

    config_class = InceptionConfig
    config: InceptionConfig

    def __init__(self, config: InceptionConfig):
        super().__init__(
            config,
            backbone=InceptionModel(config),
            classifier=Classifier(config.out_dim, config.num_labels),
        )


class InceptionMix(HFCoTMixBase):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config=config,
            pretrained_mdl_cls=InceptionMixmodel,
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
