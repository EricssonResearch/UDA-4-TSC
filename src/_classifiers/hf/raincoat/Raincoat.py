from typing import Dict
from _backbone.nn.raincoat import (
    RaincoatConfig,
    RaincoatBackbone,
    Classifier,
)
from _classifiers.hf.raincoat.base import RaincoatBasemodel, HFRaincoatBase


"""Raincoat model for time series classification as described in 'Domain Adaptation for Time Series Under Feature and Label Shifts'

@inproceedings{he2023domain,
title = {Domain Adaptation for Time Series Under Feature and Label Shifts},
author = {He, Huan and Queen, Owen and Koker, Teddy and Cuevas, Consuelo and Tsiligkaridis, Theodoros and Zitnik, Marinka},
booktitle = {https://arxiv.org/abs/2302.03133},
year      = {2023}
}

Code taken from the official [implementation](https://github.com/mims-harvard/Raincoat/tree/91db75f41d74e0f18c3f6ef50f884a7fbb78b60f)
"""


class RaincoatModel(RaincoatBasemodel):

    config_class = RaincoatConfig
    config: RaincoatConfig

    def __init__(self, config: RaincoatConfig):
        super().__init__(
            config,
            backbone=RaincoatBackbone(config),
            classifier=Classifier(
                config.averagepool_to * config.n_out_channels + config.n_fourier_modes,
                config.num_labels,
            ),
        )


class Raincoat(HFRaincoatBase):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config=config,
            pretrained_mdl_cls=RaincoatModel,
            pretrained_cfg_cls=RaincoatConfig,
            **kwargs,
        )
