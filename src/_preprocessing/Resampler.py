from _preprocessing.Preprocessor import Preprocessor
from typing import TypeVar, Dict
import numpy as np
from _utils.enumerations import DatasetColumnsEnum
from datasets import Dataset
from tslearn.preprocessing import TimeSeriesResampler


TResampler = TypeVar("TResampler", bound="Resampler")


class Resampler(Preprocessor):
    """Will resample the time series in a linear fashion to have a length equal to target_length"""

    target_length: int
    tsr: TimeSeriesResampler = None

    class Config:
        arbitrary_types_allowed = True

    def fit(self, X: Dataset) -> TResampler:
        print("Fitting Resampler")
        self.tsr = TimeSeriesResampler(sz=self.target_length)
        return self

    def transform(self, dataset: Dataset) -> Dataset:

        # define a function that maps a single example in the dataset
        def map_function(example: Dict) -> Dict:
            mts = example[DatasetColumnsEnum.mts]
            n_channels = len(mts)
            ts_len = len(mts[0])
            mts = np.array(mts).reshape(1, ts_len, n_channels)
            mts: np.ndarray = self.tsr.fit_transform(mts)[0]
            mts = mts.transpose().tolist()
            example[DatasetColumnsEnum.mts] = mts
            return example

        # apply the mapping function on the whole dataset
        return dataset.map(map_function)
