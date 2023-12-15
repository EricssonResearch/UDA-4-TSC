from _preprocessing.Preprocessor import Preprocessor
from typing import TypeVar, Dict
import numpy as np
from _utils.enumerations import DatasetColumnsEnum
from datasets import Dataset


TTimeMatchNormalize = TypeVar("TTimeMatchNormalize", bound="TimeMatchNormalize")


class TimeMatchNormalize(Preprocessor):

    min_value: int
    max_value: int

    class Config:
        arbitrary_types_allowed = True

    def fit(self, X: Dataset) -> TTimeMatchNormalize:
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        # define a function that maps a single example in the dataset
        def map_function(example: Dict) -> Dict:
            mts = np.asarray(example[DatasetColumnsEnum.mts])
            mts = (
                np.clip(mts, a_min=self.min_value, a_max=self.max_value).astype(np.float32)
                / self.max_value
            )
            example[DatasetColumnsEnum.mts] = mts.tolist()
            return example

        # apply the mapping function on the whole dataset
        return dataset.map(map_function)
