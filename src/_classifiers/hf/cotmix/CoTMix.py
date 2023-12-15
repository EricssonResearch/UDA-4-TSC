from typing import Dict
from _backbone.nn.cotmix import (
    CoTMixConfig,
    CoTMixBackbone,
    Classifier,
)
from _classifiers.hf.cotmix.base import CoTMixBasemodel, HFCoTMixBase
from _datasets.base import TLDataset
from _utils.enumerations import *


class CoTMixmodel(CoTMixBasemodel):

    config_class = CoTMixConfig
    config: CoTMixConfig

    def __init__(self, config: CoTMixConfig):
        super().__init__(
            config,
            backbone=CoTMixBackbone(config),
            classifier=Classifier(config.avgpool_to, config.n_out_channels, config.num_labels),
        )


class CoTMix(HFCoTMixBase):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config=config,
            pretrained_mdl_cls=CoTMixmodel,
            pretrained_cfg_cls=CoTMixConfig,
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
