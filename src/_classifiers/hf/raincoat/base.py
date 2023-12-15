import torch
from transformers import PreTrainedModel
from typing import Type, TypeVar, Dict
from _classifiers.hf.base import HuggingFaceClassifier
from _datasets.base import TLDataset
from _classifiers.hf.common import CommonTrainer
from _utils.enumerations import *
from _backbone.nn.raincoat import RaincoatConfigBase, Tf_encoder, SinkhornDistance
from pytorch_metric_learning.losses import ContrastiveLoss

THFRaincoatBase = TypeVar("THFRaincoatBase", bound="HFRaincoatBase")


class RaincoatBasemodel(PreTrainedModel):
    """
    Abstract base model for raincoat based models.
    Not intended as a standalone class.
    """

    config: RaincoatConfigBase

    def __init__(
        self, config: RaincoatConfigBase, backbone: PreTrainedModel, classifier: torch.nn.Module
    ):
        super().__init__(config)
        self.backbone = backbone

        self.tf_encoder = Tf_encoder(config)
        self.classifier = classifier

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.sinkhorn_loss = SinkhornDistance(eps=1e-3, max_iter=1000, reduction="sum")
        self.constrastive_loss = ContrastiveLoss(
            pos_margin=0.5
        )  # This function categorizes samples within the same class as similar, and labels the distance between them accordingly

    def forward_train(self, mts, labels, domain_y):
        # Forward pass for training

        # Initialize the loss
        total_loss = 0

        # Source and target indices
        src_indxs = domain_y == DomainEnumInt.source.value
        trg_indxs = domain_y == DomainEnumInt.target.value

        # Encode the input into temporal and frequency features
        # The first output contains concatenated temporal and frequency features
        # The second output contains the rfft of the mts, multiplied by a matrix
        features, _ = self.tf_encoder(mts, self.backbone(mts)[0])

        # Compute class prediction
        logits = self.classifier(features)
        # Compute classification loss
        total_loss += self.config.class_loss_w * self.cross_entropy_loss(logits, labels)

        # Separate source and target domain features
        src_feat = features[src_indxs]
        trg_feat = features[trg_indxs]

        if any(src_indxs):
            # The following contrastive loss is not mentioned in the paper, but it's part of the official implementation
            # Compute contrastive loss of the source
            src_labels = labels[src_indxs]
            total_loss += self.config.cont_loss_w * self.constrastive_loss(src_feat, src_labels)

            if any(trg_indxs):
                # Compute the cost for the optimal transport plan using the Sinkhorn algorithm
                ot_cost, _, _ = self.sinkhorn_loss(src_feat, trg_feat)
                total_loss += self.config.sink_loss_w * ot_cost

        return (total_loss, logits)

    def forward_eval(self, mts, labels):
        # Forward pass at inference time
        features, _ = self.tf_encoder(mts, self.backbone(mts)[0])  # source and target features
        logits = self.classifier(features)  # source and target logits
        total_loss = self.cross_entropy_loss(
            logits, labels
        )  # cross-entropy loss, target samples will not contribute
        return (total_loss, logits)

    def forward(self, mts, labels, domain_y=None):
        if domain_y is not None:
            return self.forward_train(mts, labels, domain_y)
        else:
            return self.forward_eval(mts, labels)


class HFRaincoatBase(HuggingFaceClassifier):
    default_lr: float = 0.0005
    default_wd: float = 0.0001

    def __init__(
        self,
        config: Dict,
        pretrained_mdl_cls: Type[RaincoatBasemodel],
        pretrained_cfg_cls: Type[RaincoatConfigBase],
        **kwargs,
    ):
        super().__init__(
            config=config,
            pretrained_mdl_cls=pretrained_mdl_cls,
            pretrained_cfg_cls=pretrained_cfg_cls,
            **kwargs,
        )

        # Optimizer used in Raincoat
        optimizer = torch.optim.AdamW(
            [p for p in self.classifier.parameters() if p.requires_grad],
            lr=self.config.training.get("learning_rate", self.default_lr),
            weight_decay=self.config.training.get("weight_decay", self.default_wd),
        )
        self.addition_training_args["optimizers"] = (optimizer, None)

        self.trainer_class = CommonTrainer

    def fit(self, tl_dataset: TLDataset) -> THFRaincoatBase:
        new_tl_dataset = self.get_new_tl_dataset(tl_dataset)
        return super().fit(new_tl_dataset)

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
