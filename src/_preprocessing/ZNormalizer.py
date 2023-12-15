from _preprocessing.Preprocessor import Preprocessor
from _utils.enumerations import DatasetColumnsEnum
from datasets import Dataset
from typing import TypeVar, Dict
import numpy as np


TZNormalizer = TypeVar("TZNormalizer", bound="ZNormalizer")


class ZNormalizer(Preprocessor):

    verbose: bool

    def fit(self, X: Dataset) -> TZNormalizer:
        if self.verbose is True:
            print("Fitting ZNormalizer ...")
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        # define the mapping function that znormalize all columns
        def map_function(example: Dict) -> Dict:
            x = np.array(example[DatasetColumnsEnum.mts])
            std = x.std(axis=1, keepdims=True)
            std[std == 0] = 1.0  # avoid divide by zero
            x = (x - x.mean(axis=1, keepdims=True)) / std
            example[DatasetColumnsEnum.mts] = x.tolist()
            return example

        return dataset.map(map_function)
