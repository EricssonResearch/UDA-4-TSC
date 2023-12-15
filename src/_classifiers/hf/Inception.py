from _classifiers.hf.base import HuggingFaceClassifier
from _datasets.base import TLDataset
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    TrainerCallback,
)
from typing import Dict, TypeVar
import torch
from _backbone.nn.inception import InceptionModel, InceptionConfig
from _utils.enumerations import *


# I belive this class ought to be in the same package as _backbone.nn.MLPModel to follow huggingface's archi
class InceptionForTimeSeriesClassification(PreTrainedModel):
    config_class = InceptionConfig
    num_labels: int
    backbone: InceptionModel
    fc: torch.nn.Module
    loss_fct: torch.nn.CrossEntropyLoss

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.backbone = InceptionModel(config)
        self.fc = torch.nn.Linear(self.config.out_dim, self.num_labels)
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, labels, mts):
        x = self.backbone(mts)[0]
        logits = self.fc(x)
        loss = self.loss_fct(logits, labels)

        return (loss, logits)  # required by huggingface


TInception = TypeVar("TInception", bound="Inception")


class Inception(HuggingFaceClassifier):

    default_lr: float = 0.001
    scheduler_lr_config: Dict = {
        "mode": "min",
        "factor": 0.5,
        "patience": 50,
        "min_lr": 0.0001,
    }

    def __init__(self, config: Dict, **kwargs):
        super().__init__(
            config=config,
            pretrained_mdl_cls=InceptionForTimeSeriesClassification,
            pretrained_cfg_cls=InceptionConfig,
            **kwargs,
        )

        # these are fixed params here to reproduce Inception
        self.config.training["lr_scheduler_type"] = "constant"
        self.config.training["logging_strategy"] = "epoch"

        # fixed to reproduce inception
        optimizer = torch.optim.AdamW(
            self.classifier.parameters(),
            lr=self.config.training.get("learning_rate", self.default_lr),
            weight_decay=self.config.training.get("weight_decay", 0.0),
        )

        # update if needed from config
        self.scheduler_lr_config.update(self.config.scheduler_lr)
        scheduler_lr = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **self.scheduler_lr_config
        )

        class CallbackReduceLROnPlateau(TrainerCallback):
            # a hack since ReduceLROnPlateau is not in huggingface
            # https://github.com/huggingface/transformers/issues/16503

            scheduler_lr: torch.optim.lr_scheduler.ReduceLROnPlateau
            cur_lr: float

            def __init__(self, scheduler_lr: torch.optim.lr_scheduler.ReduceLROnPlateau, **kwargs):
                super().__init__(**kwargs)
                self.scheduler_lr = scheduler_lr
                self.cur_lr = self.scheduler_lr.optimizer.param_groups[0]["lr"]

            def on_step_begin(self, args, state, control, **kwargs):
                # for some reason we need to force the lr at first epcoh
                self.scheduler_lr.optimizer.param_groups[0]["lr"] = self.cur_lr

            def on_log(self, args, state, control, logs, **kwargs):
                if "loss" in logs:
                    # make sure they are using the correct lr
                    self.scheduler_lr.optimizer.param_groups[0]["lr"] = self.cur_lr
                    # perform the scheduler step
                    self.scheduler_lr.step(logs["loss"])
                    # update the current lr
                    self.cur_lr = self.scheduler_lr.optimizer.param_groups[0]["lr"]
                    # update the logs to match the used lr
                    logs["learning_rate"] = self.cur_lr

        # force the use of adam optimizer
        self.addition_training_args["optimizers"] = (optimizer, None)
        # force the use of ReduceLROnPlateau scheduler
        self.addition_training_args["callbacks"] = [CallbackReduceLROnPlateau(scheduler_lr)]

    def inject_config_based_on_dataset(self, config: Dict, tl_dataset: TLDataset) -> Dict:
        # check if attr exist
        if InjectConfigEnum.backbone not in config:
            config[InjectConfigEnum.backbone] = {}
        config[InjectConfigEnum.backbone][InjectConfigEnum.n_input_channels] = len(
            tl_dataset.source[DataSplitEnum.train][DatasetColumnsEnum.mts][0]
        )
        return super().inject_config_based_on_dataset(config, tl_dataset)
