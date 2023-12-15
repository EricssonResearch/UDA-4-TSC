from _preprocessing.Preprocessor import Preprocessor
from typing import TypeVar, Dict
import numpy as np
from _utils.enumerations import DatasetColumnsEnum
from datasets import Dataset


TRemoveNaN = TypeVar("TRemoveNaN", bound="RemoveNaN")


class RemoveNaN(Preprocessor):
    class Config:
        arbitrary_types_allowed = True

    def fit(self, X: Dataset) -> TRemoveNaN:
        print("Removing NaNs")
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        # define a function that maps a single example in the dataset
        def map_function(example: Dict) -> Dict:
            mts = example[DatasetColumnsEnum.mts]
            for dim in range(len(mts)):
                mts[dim] = np.array(mts[dim])[np.isfinite(mts[dim])].tolist()
            example[DatasetColumnsEnum.mts] = mts
            return example

        # apply the mapping function on the whole dataset
        return dataset.map(map_function)
