import math
from transformers import PreTrainedModel, TrainerCallback
import torch
import torch.nn as nn
from pydantic import BaseModel
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, TypeVar, Type
from _utils.enumerations import *
from _datasets.base import TLDataset
from _classifiers.hf.base import HuggingFaceClassifier
from _backbone.nn.GradReverse import GradReverse
from _backbone.nn.codats import DannConfig
from _classifiers.hf.common import CommonTrainer

"""
DANN model according to DANN paper: https://arxiv.org/pdf/1505.07818.pdf
"""

THFDANN = TypeVar("THFDANN", bound="HFDANN")


class HFDANNConfig(BaseModel, extra="allow"):
    max_epochs_lr_scheduler: int  # a number of ficional epochs to create the lr scheduler


class DANNmodel(PreTrainedModel):
    """
    Implementation of Adversarial Training from DANN.
    Not intended as a standalone class.
    """

    config: DannConfig

    def __init__(
        self,
        config: DannConfig,
        classifier: nn.Module,
        discriminator: nn.Module,
        backbone: PreTrainedModel,
    ):
        """Init for the base dann model, config passed to mother class"""
        super().__init__(config)
        self.classifier = classifier
        self.discriminator = discriminator
        self.backbone = backbone

        # Objective functions for the source task classifier and the discriminator
        self.classifier_loss = nn.CrossEntropyLoss()
        self.discriminator_loss = nn.CrossEntropyLoss()

        self.tradeoff = None  # Initially set at None and updated during training

    def grad_reverse(self, x: torch.Tensor) -> torch.Tensor:
        return GradReverse.apply(x, self.tradeoff)

    def forward(self, mts, labels, domain_y=None):
        features = self.backbone(mts)[0]
        class_pred = self.classifier(features)
        class_loss = self.classifier_loss(class_pred, labels)
        if (
            domain_y is not None
        ):  # Domain labels are only used during training. If evaluating the model, then skip
            dom_pred = self.discriminator(
                self.grad_reverse(features)
            )  # Gradient reversal layer applied for adversarial training
            dom_loss = self.discriminator_loss(dom_pred, domain_y)
            return (
                self.config.class_loss_w * class_loss + self.config.disc_loss_w * dom_loss,
                class_pred,
            )
        else:
            return (class_loss, class_pred)


class HFDANN(HuggingFaceClassifier):

    # Parameters set according to [DANN](https://arxiv.org/pdf/1505.07818.pdf)
    default_lr: float = 0.0001
    schedulers_config: Dict = {"alpha": 10, "beta": 0.75, "gamma": 10}

    def __init__(
        self,
        config: Dict,
        pretrained_mdl_cls: Type[DANNmodel],
        pretrained_cfg_cls: Type[DannConfig],
        **kwargs,
    ):
        super().__init__(
            config=config,
            pretrained_mdl_cls=pretrained_mdl_cls,
            pretrained_cfg_cls=pretrained_cfg_cls,
            **kwargs,
        )

        # Adam optimizer used in CoDATS
        self.optimizer = torch.optim.AdamW(
            [p for p in self.classifier.parameters() if p.requires_grad],
            lr=self.config.training.get("learning_rate", self.default_lr),
            weight_decay=self.config.training.get("weight_decay", 0.0),
        )
        # Learning Rate scheduler is implemented in 'fit'

        self.trainer_class = CommonTrainer

    def get_scheduler_total_num_train_steps(self, tl_dataset: TLDataset) -> int:
        # Compute total number of training steps
        max_epochs_lr_scheduler = HFDANNConfig(**self.config.dict()).max_epochs_lr_scheduler
        size_of_train = len(tl_dataset.source[DataSplitEnum.train])
        batch_size = self.config.training.get("per_device_train_batch_size", 8)
        total_num_training_steps = math.ceil(size_of_train / batch_size) * max_epochs_lr_scheduler
        return total_num_training_steps

    def fit(self, tl_dataset: TLDataset) -> THFDANN:
        # get the dataset for dann
        new_tl_dataset = self.get_new_tl_dataset(tl_dataset)

        # LR scheduler
        total_num_training_steps = self.get_scheduler_total_num_train_steps(new_tl_dataset)

        # Callback to reduce the tradeoff during training

        class CallbackUpdateTradeoff(TrainerCallback):
            # Increase the tradeoff between source classification task and domain losses

            def __init__(self, classifier: DANNmodel, gamma: int, max_steps: int):
                self.classifier = classifier
                self.gamma = gamma
                self.max_steps = max_steps

            def on_train_begin(self, args, state, control, **kwargs):
                self.classifier.tradeoff = 0.0

            def on_step_begin(self, args, state, control, **kwargs):
                p = state.global_step / self.max_steps
                self.classifier.tradeoff = 2 / (1 + math.exp(-self.gamma * p)) - 1

        self.addition_training_args["callbacks"] = [
            CallbackUpdateTradeoff(
                self.classifier, self.schedulers_config["gamma"], total_num_training_steps
            )
        ]

        # Function to compute the LR as a function of the curr step index (from the DANN paper)
        def dann_lambda(current_step: int):
            """Will return the lr at the current step"""
            p = current_step / total_num_training_steps
            return (
                1.0 / (1.0 + p * self.schedulers_config["alpha"]) ** self.schedulers_config["beta"]
            )

        # Use Adam optimizer with DANN's lr scheduler
        self.addition_training_args["optimizers"] = (
            self.optimizer,
            LambdaLR(self.optimizer, dann_lambda),
        )

        return super().fit(new_tl_dataset)
