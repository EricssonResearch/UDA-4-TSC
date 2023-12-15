from _classifiers.sk.base import SKClassifier
from _datasets.base import TLDataset
from datasets import Dataset
import numpy as np
from tslearn.utils import to_time_series_dataset
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
from typing import Dict, TypeVar, Tuple, List
from _utils.enumerations import DataSplitEnum, DatasetColumnsEnum, DomainNameEnum
from transformers import EvalPrediction
import os, pickle as pkl
from sklearn.utils.validation import NotFittedError


TKNNDTW = TypeVar("TKNNDTW", bound="KNNDTW")


class KNNDTW(SKClassifier):

    classifier: KNeighborsTimeSeriesClassifier

    def __init__(self, config: Dict, **kwargs):
        super().__init__(config, KNeighborsTimeSeriesClassifier, **kwargs)

    @staticmethod
    def transform_dataset_to_X_Y_numpy(dataset: Dataset) -> Tuple[List, List]:
        x = np.array(dataset[DatasetColumnsEnum.mts])
        y = np.array(dataset[DatasetColumnsEnum.labels])
        return x, y

    def save_model(self) -> None:
        """
        By default the whole dataset is saved as model for the nearest neighbor.
        Therefore we would like to avoid that by saving the model without saving
        the whole dataset avoid wasting time and space. Which will require refitting
        when loading the model.
        """
        # create the directory if it does not exists
        os.makedirs(self.train_dir, exist_ok=True)
        # dump the config as pkl instead of the whole model
        pkl_file = f"{self.train_dir}/{self.model_best_fname}"
        with open(pkl_file, "wb") as f:
            pkl.dump(self.config, f)

    def load_model(self, **args) -> None:
        """
        Instead of loading the model we need to load config and init the model and fit it.
        """
        pkl_file = f"{self.train_dir}/{self.model_best_fname}"
        with open(pkl_file, "rb") as f:
            self.config = pkl.load(f)
            self.classifier = KNeighborsTimeSeriesClassifier(**self.config.classifier)

    def predict(
        self, tl_dataset: TLDataset
    ) -> Dict[DomainNameEnum, Dict[DataSplitEnum, EvalPrediction]]:
        """
        Since the loaded model is not fitted to avoid saving the whole dataset on disk.
        We need to fit before prediction, then call prediction of mother class.
        """
        try:
            self.classifier._is_fitted()
            # continue since already fitted
        except NotFittedError:
            # not fitted need to fit
            x_source, y_source = self.transform_dataset_to_X_Y_numpy(
                tl_dataset.source[DataSplitEnum.train]
            )

            x_source, y_source = self.pre_predict_source(x_source, y_source)

            self.classifier.fit(x_source, y_source)

        return super().predict(tl_dataset)

    def pre_predict_source(
        self, x_source: np.array, y_source: np.array
    ) -> Tuple[np.array, np.array]:
        # need to transpose since for tslearn channels are axis=2 and
        # for our pipeline channels are axis=1
        # note axis=0 is always the batch dimension
        return to_time_series_dataset(x_source).transpose([0, 2, 1]), y_source

    def pre_predict_target(
        self, x_target: np.array, y_target: np.array
    ) -> Tuple[np.array, np.array]:
        return self.pre_predict_source(x_target, y_target)

    def fit(self, tl_dataset: TLDataset) -> TKNNDTW:
        print("No need to fit since nearest neighbor is a lazy classifier")

        return self
