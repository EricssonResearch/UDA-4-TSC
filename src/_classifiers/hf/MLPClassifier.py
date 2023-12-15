from _classifiers.hf.base import HuggingFaceClassifier
from _datasets.base import TLDataset
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
)
from typing import Dict, TypeVar
import torch
from _backbone.nn.mlp import MLPModel, MLPConfig
from _utils.enumerations import *


# I belive this class ought to be in the same package as _backbone.nn.MLPModel to follow huggingface's archi
class MLPForTimeSeriesClassification(PreTrainedModel):
    config_class = MLPConfig
    num_labels: int
    backbone: PreTrainedModel
    classifer: torch.nn.Module
    loss_fct: torch.nn.CrossEntropyLoss

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.backbone = MLPModel(config)
        self.classifer = torch.nn.Linear(config.out_dim, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, labels, mts):
        x = self.backbone(mts)[0]
        logits = self.classifer(x)
        loss = self.loss_fct(logits, labels)

        return (loss, logits)  # required by huggingface


TMLPClassifier = TypeVar("TMLPClassifier", bound="MLPClassifier")


class MLPClassifier(HuggingFaceClassifier):
    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config=config,
            pretrained_mdl_cls=MLPForTimeSeriesClassification,
            pretrained_cfg_cls=MLPConfig,
            **kwargs,
        )

    def inject_config_based_on_dataset(self, config: Dict, tl_dataset: TLDataset) -> Dict:
        # check if attr exist
        if InjectConfigEnum.backbone not in config:
            config[InjectConfigEnum.backbone] = {}
        # add num channels
        config[InjectConfigEnum.backbone][InjectConfigEnum.n_input_channels] = len(
            tl_dataset.source[DataSplitEnum.train][DatasetColumnsEnum.mts][0]
        )
        # add length of the time series
        config[InjectConfigEnum.backbone][InjectConfigEnum.input_len] = len(
            tl_dataset.source[DataSplitEnum.train][DatasetColumnsEnum.mts][0][0]
        )
        return super().inject_config_based_on_dataset(config, tl_dataset)
