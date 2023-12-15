from _datasets.base import TLDataset
from _classifiers.base import TLClassifier
import numpy as np
import pandas as pd
from datasets import Dataset
from typing import TypeVar, Dict, Tuple
from transformers import EvalPrediction
from sklearn.base import ClassifierMixin
from _utils.enumerations import *
import os
import pickle as pkl
from pydantic import BaseModel
from _classifiers.sk.utils import transform_dataset_to_X_Y_numpy


TSKClassifier = TypeVar("TSKClassifier", bound="SKClassifier")


class SKConfig(BaseModel):
    classifier: Dict


class SKClassifier(TLClassifier):

    classifier: ClassifierMixin
    model_best_fname: str = "model_best.pkl"

    def __init__(self, config: Dict, classifier: ClassifierMixin, **kwargs):
        super().__init__(config, **kwargs)
        # make sure the config respects the expected sk ones
        self.config = SKConfig(**config)
        # create the classifier
        self.classifier = classifier(**self.config.classifier)

    @staticmethod
    def transform_dataset_to_X_Y_numpy(dataset: Dataset) -> Tuple[np.array, np.array]:
        return transform_dataset_to_X_Y_numpy(dataset)

    def save_model(self) -> None:
        """
        Dumping in pickle the final model.
        """
        # create the directory if it does not exists
        os.makedirs(self.train_dir, exist_ok=True)

        # dump the file in the created dir
        pkl_file = f"{self.train_dir}/{self.model_best_fname}"
        with open(pkl_file, "wb") as f:
            pkl.dump(self.classifier, f)

    def load_model(self, **args) -> None:
        """
        Loading from pickle the final model.
        """
        pkl_file = f"{self.train_dir}/{self.model_best_fname}"
        with open(pkl_file, "rb") as f:
            self.classifier = pkl.load(f)

    def pre_predict_source(
        self, x_source: np.array, y_source: np.array
    ) -> Tuple[np.array, np.array]:
        return x_source, y_source

    def pre_predict_target(
        self, x_target: np.array, y_target: np.array
    ) -> Tuple[np.array, np.array]:
        return x_target, y_target

    def predict(
        self, tl_dataset: TLDataset
    ) -> Dict[DomainNameEnum, Dict[DataSplitEnum, EvalPrediction]]:
        # init the empty result dict
        res: Dict = {}

        # init res dict for source and target
        res[DomainNameEnum.source.value] = {}
        res[DomainNameEnum.target.value] = {}

        # loop through splits in source
        for split in tl_dataset.source:
            # transform for numpy x y
            x_source, y_source = self.transform_dataset_to_X_Y_numpy(tl_dataset.source[split])

            # pre_predict_source
            x_source, y_source = self.pre_predict_source(x_source, y_source)

            # perform the prediction
            y_pred = self.classifier.predict_proba(x_source)

            # add prediction
            res[DomainNameEnum.source.value][split] = EvalPrediction(
                predictions=y_pred, label_ids=y_source
            )

        # loop through splits in target
        for split in tl_dataset.target:
            # transform for numpy x y
            x_target, y_target = self.transform_dataset_to_X_Y_numpy(tl_dataset.target[split])

            # pre_predict_target
            x_target, y_target = self.pre_predict_target(x_target, y_target)

            # perform the prediction
            y_pred = self.classifier.predict_proba(x_target)

            # add prediction
            res[DomainNameEnum.target.value][split] = EvalPrediction(
                predictions=y_pred, label_ids=y_target
            )

        return res
