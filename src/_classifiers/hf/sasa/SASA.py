from typing import Dict
from _backbone.nn.sasa import SASABackbone, SASAConfig
from _classifiers.hf.sasa.base import SASABasemodel, HFSASABase


"""
Code taken from https://github.com/DMIRLAB-Group/SASA-pytorch/blob/main/SASA_Model.py
"""


class SASAModel(SASABasemodel):

    config_class = SASAConfig
    config: SASAConfig

    def __init__(self, config: SASAConfig):
        super().__init__(
            config,
            backbone=SASABackbone(config),
        )


class SASA(HFSASABase):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config=config,
            pretrained_mdl_cls=SASAModel,
            pretrained_cfg_cls=SASAConfig,
            **kwargs,
        )
