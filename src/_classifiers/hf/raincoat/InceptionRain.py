from typing import Dict
from _backbone.nn.inception import (
    InceptionConfig,
    InceptionModel,
    Classifier,
)
from _classifiers.hf.raincoat.base import RaincoatBasemodel, HFRaincoatBase


"""
Inception backbone for RainCoat
"""


class InceptionRainModel(RaincoatBasemodel):

    config_class = InceptionConfig
    config: InceptionConfig

    def __init__(self, config: InceptionConfig):
        super().__init__(
            config,
            backbone=InceptionModel(config),
            classifier=Classifier(
                config.out_dim + config.n_fourier_modes,
                config.num_labels,
            ),
        )


class InceptionRain(HFRaincoatBase):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config=config,
            pretrained_mdl_cls=InceptionRainModel,
            pretrained_cfg_cls=InceptionConfig,
            **kwargs,
        )
