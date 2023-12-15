from _preprocessing.Preprocessor import Preprocessor
from datasets import Dataset
from typing import TypeVar, Dict
import numpy as np
from _utils.enumerations import DatasetColumnsEnum

TTruncate = TypeVar("TTruncate", bound="Truncate")


class Truncate(Preprocessor):

    verbose: bool
    truncate_to: int

    class Config:
        arbitrary_types_allowed = True

    def fit(self, X: Dataset) -> TTruncate:
        if self.verbose is True:
            print("Fitting Truncate ...")
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        # define the mapping function that truncates all channels
        def map_function(example: Dict) -> Dict:
            mts = example[DatasetColumnsEnum.mts]
            ts_len = len(mts[0])
            if ts_len > self.truncate_to:
                mts_np = np.asarray(mts)
                truncated_sequence = mts_np[:, : self.truncate_to]
                example[DatasetColumnsEnum.mts] = truncated_sequence.tolist()
                return example
            else:
                return example

        return dataset.map(map_function)
