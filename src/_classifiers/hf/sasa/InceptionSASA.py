import copy
import torch
from typing import Dict
from _backbone.nn.inception import InceptionConfig, InceptionModel
from _classifiers.hf.sasa.base import SASABasemodel, HFSASABase
from _utils.enumerations import *

"""
Inception backbone for SASA
"""


class IncpetionBackboneSASA(InceptionModel):
    def forward(self, x: torch.Tensor):
        x = torch.transpose(x, 2, 1)
        res = super().forward(x)[0]

        return ((None, (res, None)), None)


class InceptionSASAModel(SASABasemodel):

    config_class = InceptionConfig
    config: InceptionConfig

    def __init__(self, config: InceptionConfig):
        backbone_conf = copy.deepcopy(config)
        backbone_conf.n_input_channels = 1
        super().__init__(
            config,
            backbone=IncpetionBackboneSASA(backbone_conf),
        )


class InceptionSASA(HFSASABase):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config=config,
            pretrained_mdl_cls=InceptionSASAModel,
            pretrained_cfg_cls=InceptionConfig,
            **kwargs,
        )
