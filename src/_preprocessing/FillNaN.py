from _preprocessing.Preprocessor import Preprocessor
from typing import TypeVar, Dict, Union
import numpy as np
import pandas as pd
from _utils.enumerations import DatasetColumnsEnum
from datasets import Dataset
from enum import Enum


TFillNaN = TypeVar("TFillNaN", bound="FillNaN")


class ValueEnums(str, Enum):
    mean: str = "mean"


class FillNaN(Preprocessor):

    method: str = None
    value: Union[int, float, str]

    class Config:
        arbitrary_types_allowed = True

    def fit(self, X: Dataset) -> TFillNaN:
        print("Filling NaNs")
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        # define a function that maps a single example in the dataset
        def map_function(example: Dict) -> Dict:
            mts = pd.DataFrame(np.array(example[DatasetColumnsEnum.mts]).transpose())
            if self.value == ValueEnums.mean:
                for col in mts.columns:
                    m = mts[col].mean()
                    mts[col] = mts[col].apply(lambda x: m if pd.isna(x) else x)
            else:
                mts = mts.fillna(value=self.value, method=self.method, axis=0)

            # assert all nan removed
            assert not mts.isna().values.any(), "Not all NaN were removed"

            example[DatasetColumnsEnum.mts] = mts.values.transpose().tolist()
            return example

        # apply the mapping function on the whole dataset
        return dataset.map(map_function)
