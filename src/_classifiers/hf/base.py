import torch
from datasets import concatenate_datasets, DatasetDict
from _classifiers.base import TLClassifier
from transformers import (
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    AutoModel,
    AutoConfig,
    AutoModelForImageClassification,
    PretrainedConfig,
)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from transformers.integrations import TensorBoardCallback, TrainerCallback
from typing import Dict, TypeVar, Type, Union, Any
from _datasets.base import TLDataset
from _scoring.base import Scorer
from _utils.enumerations import *
from pydantic import BaseModel
import pathlib
import time
import numpy as np
from torch import nn

THuggingFaceClassifier = TypeVar("THuggingFaceClassifier", bound="HuggingFaceClassifier")


PER_DEVICE_EVAL_BATCH_SIZE = 32
NUM_CHECKPOINTS = 20


class TimeBudgetStoppingCallback(TrainerCallback):

    """
    A callback that will check the time passed since training, if reached limit, will stop the training.
    """

    def __init__(self, time_limit: int, do_checkpoint: bool, num_train_epochs: int):
        """
        time_limit: in seconds (default: `None` means nothing happens)
        do_checkpoint: whether or not to save model
        """
        self.start_time = None
        self.time_limit = time_limit
        self.do_checkpoint = do_checkpoint
        num_checkpoints = NUM_CHECKPOINTS
        # the checkpointing epochs this will correspond to following
        # [1, 2, 3, 5, 7, 12, 19, 30, 49, 79, 128, 207, 336, 546, 886, 1439, 2336, 3793, 6159, 10000]
        self.checkpoint_epochs = set(
            np.ceil(np.logspace(0, np.log10(num_train_epochs), num=num_checkpoints))
            .astype(int)
            .tolist()
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        if (self.do_checkpoint is True) and (epoch in self.checkpoint_epochs):
            control.should_save = True
            control.should_evaluate = True

    def on_step_end(self, args, state, control, **kwargs):
        """
        Will send a signal to stop training if time limit reached
        """
        # do nothing if timi limit is not set
        if self.time_limit is None:
            return None

        # check if time limit reached
        if (time.time() - self.start_time) >= self.time_limit:
            # set the training to stop
            control.should_training_stop = True
            # make sure to save when model ends if do_checkpoint is True
            control.should_save = self.do_checkpoint
            control.should_evaluate = self.do_checkpoint

        return None


class HFConfig(BaseModel, extra="allow"):
    backbone: Dict
    classifier: Dict
    training: Dict
    device: TorchDeviceEnum = TorchDeviceEnum.gpu
    scheduler_lr: Dict = {}
    do_checkpoint: bool = False  # whether or not to save the model
    time_limit: int  # in seconds


class HuggingFaceClassifier(TLClassifier):
    """
    This is a base class that covers common functionalities to all huggingface models.
    """

    classifier: PreTrainedModel
    trainer_class: Type[Trainer]
    device: torch.device
    pretrained_mdl_cls: Type[PreTrainedModel]
    pretrained_cfg_cls: Type[PretrainedConfig]
    addition_training_args: Dict = {}
    last_checkpoint_folder: str = f"checkpoint-{CheckpointLoadEnum.last.value}"

    def __init__(
        self,
        config: Dict,
        pretrained_mdl_cls: Type[PreTrainedModel],
        pretrained_cfg_cls: Type[PretrainedConfig],
        **kwargs,
    ):
        """
        config: Represents the dictionnary

        """
        super().__init__(config=config, **kwargs)
        # make sure config is the one expected by huggingface
        self.config: HFConfig = HFConfig(**config)
        self.fill_fixed_nn_training_config()
        self.pretrained_cfg_cls = pretrained_cfg_cls
        self.pretrained_mdl_cls = pretrained_mdl_cls
        self.classifier = pretrained_mdl_cls(
            config=pretrained_cfg_cls(**self.config.backbone, **self.config.classifier)
        )
        self.trainer_class = Trainer
        self.register_model()

        # Use cuda if available
        if (self.config.device == TorchDeviceEnum.gpu) and (not torch.cuda.is_available()):
            raise Exception("Cuda is selected yet not avaialble")

        self.device = torch.device(
            TorchDeviceEnum.gpu
            if (torch.cuda.is_available() and (self.config.device == TorchDeviceEnum.gpu))
            else TorchDeviceEnum.cpu
        )

        # Assign to device
        self.classifier = self.classifier.to(self.device)

        print("Device: ", self.classifier.device)

    def fill_fixed_nn_training_config(self) -> None:
        """Fill some static training args common for all neural nets by default."""
        self.config.training["save_strategy"] = self.config.training.get("save_strategy", "no")
        self.config.training["evaluation_strategy"] = self.config.training.get(
            "evaluation_strategy", "no"
        )
        self.config.training["logging_strategy"] = self.config.training.get(
            "logging_strategy", "epoch"
        )
        self.config.training["per_device_eval_batch_size"] = self.config.training.get(
            "per_device_eval_batch_size", PER_DEVICE_EVAL_BATCH_SIZE
        )

    def inject_config_based_on_dataset(self, config: Dict, tl_dataset: TLDataset) -> Dict:
        # check if num_labels exists
        if InjectConfigEnum.classifier not in config:
            config[InjectConfigEnum.classifier] = {}
        config[InjectConfigEnum.classifier][InjectConfigEnum.num_labels] = (
            tl_dataset.source[DataSplitEnum.train].features[DatasetColumnsEnum.labels].num_classes
        )
        return config

    def fit(self, tl_dataset: TLDataset) -> THuggingFaceClassifier:
        """This function will fit on tl_dataset.source.train and eval on both tl_dataset.source.test and tl_dataset.target.test"""
        # get training args
        training_args = TrainingArguments(
            **self.config.training,
            output_dir=self.train_dir,
            no_cuda=True if self.config.device == TorchDeviceEnum.cpu else False,
            report_to="tensorboard",
        )
        # get metrics
        compute_metrics = Scorer(tl_dataset=tl_dataset)

        # get train dataset
        train_dataset = tl_dataset.source[DataSplitEnum.train]
        # get eval dataset which is the test set
        # however not the real test set since the function
        # get_tl_dataset_with_val_instead_of_test should take
        # care of setting it either train or split from train
        eval_dataset = {
            DomainNameEnum.source.value: tl_dataset.source[DataSplitEnum.test],
            DomainNameEnum.target.value: tl_dataset.target[DataSplitEnum.test],
        }

        # create trainier
        self.trainer = self.trainer_class(
            self.classifier,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            args=training_args,
            **self.addition_training_args,
        )

        # add callback for logging with tensorboard
        self.trainer.add_callback(TensorBoardCallback)
        # add callback for time budget
        self.trainer.add_callback(
            TimeBudgetStoppingCallback(
                time_limit=self.config.time_limit,
                do_checkpoint=self.config.do_checkpoint,
                num_train_epochs=training_args.num_train_epochs,
            )
        )
        # start training
        res_train = self.trainer.train()

        print("Train output", res_train)

        return self

    def get_new_tl_dataset(self, tl_dataset: TLDataset) -> TLDataset:
        """
        Will first add for each attribute wether it is source or target
        Finally returns a new tl_dataset with:
        - new.source.train: old.source.train(labeled) + old.target.train(unlabeled)
        - new.source.test: old.source.test
        - new.target.train: old.target.train
        - new.target.test: old.target.test
        """
        # Add domain labels to train the discriminator
        def add_domain_labels(example: Dict, domain_label: int) -> Dict:
            example[AdditionalColumnEnum.domain_y] = domain_label
            return example

        # Add domain labels to the source domain
        new_source_train = tl_dataset.source[DataSplitEnum.train].map(
            lambda example: add_domain_labels(example, DomainEnumInt.source.value)
        )

        new_source_test = tl_dataset.source[DataSplitEnum.test].map(
            lambda example: add_domain_labels(example, DomainEnumInt.source.value)
        )

        # Add domain labels to the target domain
        new_target_train = tl_dataset.target[DataSplitEnum.train].map(
            lambda example: add_domain_labels(example, DomainEnumInt.target.value)
        )

        # Modify target domain classification task labels, so that they are not use during training.
        # See https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
        def modify_label(example: Dict, new_label: int) -> Dict:
            example[DatasetColumnsEnum.labels] = new_label
            return example

        new_target_train = new_target_train.map(
            lambda example: modify_label(example, nn.CrossEntropyLoss().ignore_index)
        )

        # Concatenate both source and target domain datasets, so that they are both used for training.
        new_source_train = concatenate_datasets([new_source_train, new_target_train], axis=0)

        new_source = DatasetDict(
            {
                DataSplitEnum.train.value: new_source_train,
                DataSplitEnum.test.value: new_source_test,
            }
        )

        new_target = DatasetDict(
            {
                DataSplitEnum.train.value: new_target_train,
                DataSplitEnum.test.value: tl_dataset.target[DataSplitEnum.test],
            }
        )

        # create the new tl dataset with the new source and the new target
        new_tl_dataset = TLDataset(
            dataset_name=tl_dataset.dataset_name,
            preprocessing_pipeline=tl_dataset.preprocessing_pipeline,
            source=new_source,
            target=new_target,
            gmm_root_dir=tl_dataset.gmm_root_dir,
        )

        return new_tl_dataset

    def predict(
        self, tl_dataset: TLDataset
    ) -> Dict[DomainNameEnum, Dict[DataSplitEnum, EvalPrediction]]:
        """
        Potentially same function for all HuggingFace based classifiers
        """
        # init the empty result dict
        res: Dict = {}

        # get training args
        training_args = TrainingArguments(
            output_dir=self.train_dir,
            **self.config.training,
            no_cuda=True if self.config.device == TorchDeviceEnum.cpu else False,
        )

        # create trainier which makes predicting easier
        self.trainer = self.trainer_class(
            self.classifier,
            args=training_args,
            **self.addition_training_args,
        )

        # init res dict for source and target
        res[DomainNameEnum.source.value] = {}
        res[DomainNameEnum.target.value] = {}

        # loop through splits in source
        for split in tl_dataset.source:
            # perform the prediction
            preds = self.trainer.predict(tl_dataset.source[split])

            # extract info from preds
            y_pred = preds.predictions
            y_source = preds.label_ids

            # y_pred contains logits
            # apply softmax to make them proba
            y_pred = torch.nn.functional.softmax(torch.FloatTensor(y_pred), dim=1).numpy()

            # add prediction
            res[DomainNameEnum.source.value][split] = EvalPrediction(
                predictions=y_pred, label_ids=y_source
            )

        # loop through splits in target
        for split in tl_dataset.target:
            # perform the prediction
            preds = self.trainer.predict(tl_dataset.target[split])

            # extract info from preds
            y_pred = preds.predictions
            y_target = preds.label_ids

            # y_pred contains logits
            # apply softmax to make them proba
            y_pred = torch.nn.functional.softmax(torch.FloatTensor(y_pred), dim=1).numpy()

            # add prediction
            res[DomainNameEnum.target.value][split] = EvalPrediction(
                predictions=y_pred, label_ids=y_target
            )

        return res

    def save_model(self):
        """
        This function save last checkpoint for now since other checkpoints are saved during training.
        """
        model_path = f"{self.train_dir}/{self.last_checkpoint_folder}"
        self.classifier.save_pretrained(model_path)

    def register_model(self) -> None:
        """
        To be able to use save_pretrained and from_pretrained functions offered by huggingface
        we need to register the model.
        For now using AutoModelForImageClassification but we can imagine improving with
        an explicit AutoModelForTimeSeriesClassification
        """
        # register model to load / save later
        AutoConfig.register(self.classifier.config.model_type, self.pretrained_cfg_cls)
        AutoModel.register(self.pretrained_cfg_cls, type(self.classifier.backbone))
        AutoModelForImageClassification.register(self.pretrained_cfg_cls, self.pretrained_mdl_cls)

    def load_model(
        self,
        checkpoint: Union[int, CheckpointLoadEnum],
        metric_key: str = None,
        best: BestCheckpointEnum = None,
        on_domain: DomainNameEnum = None,
        **args,
    ) -> None:
        """
        This function should load the model based on the checkpoint
        """
        checkpoint_str = str(checkpoint)
        if checkpoint == CheckpointLoadEnum.best:
            # get the best epoch
            # load the tensorboard event file
            event_file = list(
                pathlib.Path(f"{self.train_dir}/runs").glob("*/*events.out.tfevents*")
            )[0]
            event_acc = EventAccumulator(str(event_file))
            event_acc.Reload()

            # get the metrics
            _, step_nums, vals = zip(*event_acc.Scalars(f"eval/{on_domain.value}_{metric_key}"))

            # get argmax or argmin
            if best == BestCheckpointEnum.maximum:
                best_idx = np.argmax(vals)
            elif best == BestCheckpointEnum.minimum:
                best_idx = np.argmin(vals)

            # get the argmax (best epoch)
            checkpoint_str = str(step_nums[best_idx])

        # load model with selected checkpoint str
        model_path = f"{self.train_dir}/checkpoint-{checkpoint_str}"

        print("Loading Model:", model_path)

        # load the model using the huggingface api
        self.classifier = AutoModelForImageClassification.from_pretrained(model_path)
