from _preprocessing.Preprocessor import Preprocessor
from typing import TypeVar, Dict
import numpy as np
from _utils.enumerations import DatasetColumnsEnum
from datasets import Dataset


TSubSample = TypeVar("TSubSample", bound="SubSample")


class SubSample(Preprocessor):

    target_length: int
    new_idx: int = None

    class Config:
        arbitrary_types_allowed = True

    def fit(self, X: Dataset) -> TSubSample:
        print("Fitting subsample")
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        cur_length = len(dataset[0][DatasetColumnsEnum.mts][0])
        old_idx = np.arange(cur_length)
        new_idx = np.sort(np.random.permutation(old_idx)[: self.target_length])

        # define a function that maps a single example in the dataset
        def map_function(example: Dict) -> Dict:
            mts = example[DatasetColumnsEnum.mts]
            for dim in range(len(mts)):
                mts[dim] = np.array(mts[dim])[new_idx].tolist()
            example[DatasetColumnsEnum.mts] = mts
            return example

        # apply the mapping function on the whole dataset
        return dataset.map(map_function)
