from _preprocessing.Preprocessor import Preprocessor
from datasets import Dataset
from typing import TypeVar, Dict
import numpy as np
from _utils.enumerations import DatasetColumnsEnum

TGaussianPadding = TypeVar("TGaussianPadding", bound="GaussianPadding")


class GaussianPadding(Preprocessor):

    verbose: bool
    padding_to: int
    std: float = 0.1

    class Config:
        arbitrary_types_allowed = True

    def fit(self, X: Dataset) -> TGaussianPadding:
        if self.verbose is True:
            print("Fitting GaussianPadding ...")
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        # define the mapping function that pads all channels
        def map_function(example: Dict) -> Dict:
            mts = example[DatasetColumnsEnum.mts]
            ts_len = len(mts[0])

            if ts_len == self.padding_to:
                return example
            else:
                n_channels = len(mts)
                pad = np.random.normal(scale=self.std, size=(n_channels, self.padding_to - ts_len))
                mts_np = np.asarray(mts)
                padded_sequence = np.concatenate((mts_np, pad), axis=1)
                example[DatasetColumnsEnum.mts] = padded_sequence.tolist()
                return example

        return dataset.map(map_function)
