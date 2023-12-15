from typing import Dict
from _backbone.nn.codats import CoDATSConfig, Classifier, Discriminator, Feature_extractor
from _classifiers.hf.dann.base import DANNmodel, HFDANN
from _datasets.base import TLDataset
from _utils.enumerations import *


"""
CoDATS model as decribed in https://arxiv.org/pdf/2005.10996.pdf. 
"""


class CoDATSmodel(DANNmodel):

    config_class = CoDATSConfig
    config: CoDATSConfig

    def __init__(self, config: CoDATSConfig):
        super().__init__(
            config,
            backbone=Feature_extractor(config),
            discriminator=Discriminator(config.avgpool_to, config.n_source_dom),
            classifier=Classifier(config.avgpool_to, config.num_labels),
        )


class CoDATS(HFDANN):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config, pretrained_mdl_cls=CoDATSmodel, pretrained_cfg_cls=CoDATSConfig, **kwargs
        )

    def inject_config_based_on_dataset(self, config: Dict, tl_dataset: TLDataset) -> Dict:
        # check if attr exist
        if InjectConfigEnum.backbone not in config:
            config[InjectConfigEnum.backbone] = {}
        config[InjectConfigEnum.backbone][InjectConfigEnum.n_input_channels] = len(
            tl_dataset.source[DataSplitEnum.train][DatasetColumnsEnum.mts][0]
        )
        return super().inject_config_based_on_dataset(config, tl_dataset)
