from _preprocessing.Preprocessor import Preprocessor
from datasets import Dataset
from typing import TypeVar, Dict
import numpy as np
from scipy.interpolate import interp1d
from _utils.enumerations import DatasetColumnsEnum

TInterpolate = TypeVar("TInterpolate", bound="Interpolate")


class Interpolate(Preprocessor):

    verbose: bool
    interpolate_to: int = 64

    class Config:
        arbitrary_types_allowed = True

    def fit(self, X: Dataset) -> TInterpolate:
        if self.verbose is True:
            print("Fitting Interpolate ...")
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        # define the mapping function that interpolates all channels
        def map_function(example: Dict) -> Dict:
            mts = example[DatasetColumnsEnum.mts]
            np_mts = np.asarray(mts).transpose()
            ts_len = np_mts.shape[0]
            interpolator = interp1d(np.arange(0, ts_len), np_mts, axis=0, kind="linear")
            mts_interp = interpolator(np.linspace(0, ts_len - 1, self.interpolate_to))
            example[DatasetColumnsEnum.mts] = mts_interp.transpose().tolist()
            return example

        return dataset.map(map_function)
